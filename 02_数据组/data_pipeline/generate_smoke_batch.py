"""generate_smoke_batch.py — D3 批量生成 smoke episode（数据组）。

复用环境组交接的 TrajectoryRecorderV2 / OvercookedMultiEnv / ScriptedPolicy，用
可复现的随机动作脚本批量生成多个 episode。**不涉及六个训练模型**（BC/SP/PBT/FCP/
MEP/COLE）——这是刻意的范围限制：这些 checkpoint 是 TF1.15/SB1 格式，需要逐个
适配 policy_adapter.py 才能安全加载，属于独立风险项，本批次不引入。

因此本批次的 `policy_id` 统一标记为 "random_scripted"，只提供 (episode, seed)
两个拆分轴，不提供 policy 维度的多样性——这是需要在 quality_report.json 中如实
报告的已知限制。

用法（cole-platform conda 环境）：
    python generate_smoke_batch.py --layout-id simple --num-episodes 60 \
        --horizon 400 --seed-start 1000 --output-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

COLE_PLATFORM_ROOT = Path(
    r"C:\Users\36724\PycharmProjects\WelcomeScreen\Overcooked2Agent\COLE-Platform"
)
for path in (
    COLE_PLATFORM_ROOT,
    COLE_PLATFORM_ROOT / "overcookedgym" / "human_aware_rl",
    COLE_PLATFORM_ROOT / "overcookedgym" / "human_aware_rl" / "overcooked_ai",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from overcookedgym.environment_interfaces import (  # noqa: E402
    ScriptedPolicy,
    ScriptExhaustedError,
    TrajectoryRecorderV2,
)
from overcookedgym.overcooked import OvercookedMultiEnv  # noqa: E402

NUM_ACTIONS = 6  # overcooked_ai Action space: up/down/left/right/stay/interact


def physical_observations(observations, ego_agent_idx):
    if ego_agent_idx == 0:
        return observations
    if ego_agent_idx == 1:
        return observations[1], observations[0]
    raise ValueError("ego_agent_idx must be 0 or 1")


def random_action_script(rng: np.random.RandomState, horizon: int):
    return [int(a) for a in rng.randint(0, NUM_ACTIONS, size=horizon)]


def run_one_episode(layout_id, seed, human_index, ego_agent_idx, episode_id, horizon, output_dir: Path):
    trajectory_path = output_dir / f"{episode_id}.trajectory.jsonl"
    manifest_path = output_dir / f"{episode_id}.manifest.json"
    started = time.time()
    manifest = {
        "episode_id": episode_id,
        "layout_id": layout_id,
        "seed": seed,
        "human_index": human_index,
        "ego_agent_idx": ego_agent_idx,
        "horizon": horizon,
        "policy_id": "random_scripted",
        "status": "invalid_episode",
        "termination_reason": None,
        "steps_recorded": 0,
        "trajectory_path": str(trajectory_path.resolve()),
    }
    try:
        # 两个独立的 RNG 流（human / cole），种子由 episode seed 派生，保证可复现且互不相关。
        human_rng = np.random.RandomState(seed * 2 + 1)
        cole_rng = np.random.RandomState(seed * 2 + 2)
        human_policy = ScriptedPolicy(random_action_script(human_rng, horizon))
        cole_policy = ScriptedPolicy(random_action_script(cole_rng, horizon))

        env = OvercookedMultiEnv(layout_id, ego_agent_idx=ego_agent_idx)
        recorder = TrajectoryRecorderV2(env, layout_id, seed, human_index=human_index)
        observations = recorder.reset(episode_id)
        human_policy.reset()
        cole_policy.reset()
        done = False
        for _ in range(horizon):
            by_index = physical_observations(observations, env.ego_agent_idx)
            human_action = human_policy.act(by_index[human_index])
            cole_action = cole_policy.act(by_index[1 - human_index])
            observations, _, done, _ = recorder.step(human_action, cole_action)
            if done:
                break
        trajectory = recorder.to_dict(validate=True)
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_path.write_text(json.dumps(trajectory, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest["steps_recorded"] = len(trajectory["step_records"][0])
        manifest["status"] = "valid"
        manifest["termination_reason"] = "environment_done" if done else "horizon_reached"
    except ScriptExhaustedError as exc:
        manifest["termination_reason"] = f"script_exhausted: {exc}"
    except Exception as exc:  # noqa: BLE001 — record and re-raise-free, batch continues
        manifest["termination_reason"] = f"{type(exc).__name__}: {exc}"
    manifest["elapsed_seconds"] = time.time() - started
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-id", default="simple")
    parser.add_argument("--num-episodes", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifests = []
    for i in range(args.num_episodes):
        seed = args.seed_start + i
        episode_id = f"smoke-{args.layout_id}-{seed}"
        # human_index / ego_agent_idx 交替，覆盖两种视角，避免拆分数据只反映单一视角。
        human_index = i % 2
        ego_agent_idx = i % 2
        manifest = run_one_episode(
            args.layout_id, seed, human_index, ego_agent_idx, episode_id, args.horizon, args.output_dir
        )
        manifests.append(manifest)
        status = manifest["status"]
        print(f"[{i+1}/{args.num_episodes}] {episode_id}: {status} steps={manifest['steps_recorded']}")

    valid = sum(1 for m in manifests if m["status"] == "valid")
    print(f"done: {valid}/{len(manifests)} valid episodes written to {args.output_dir}")


if __name__ == "__main__":
    main()
