"""generate_scripted_chef_batch.py — D3 确定性贪心脚本批量生成（数据组）。

不使用六个训练模型（BC/SP/PBT/FCP/MEP/COLE）。这里手工推导了 `simple.layout`
网格上一套确定性走位+interact动作序列（"贪心厨师"），保证覆盖全部五类任务事件
（含 soup_plated / soup_delivered 这两个纯随机策略几乎不可能触发的稀有事件）。

layout `simple.layout` 网格（0-index, x=列, y=行）：
    y=0: X  X  P  X  X
    y=1: O  _  _  2  O
    y=2: X  1  _  _  X
    y=3: X  D  X  S  X
玩家1（focal chef）出生在 (1,2)；洋葱台在 (0,1)/(4,1)；锅在 (2,0)；
盘子台在 (1,3)；出餐口在 (3,3)。玩家2固定 STAY，不参与，只作为队友占位。

这是刻意的范围限制（非训练模型行为），必须在 quality_report.json 中如实报告：
本批次不代表真实训练智能体的行为分布，仅用于验证 D3 拆分/smoke/quality_report
全流程在有真实五类事件覆盖时的正确性。

用法（cole-platform conda 环境）：
    python generate_scripted_chef_batch.py --num-episodes 20 --horizon 400 \
        --seed-start 2000 --output-dir <dir>
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

NORTH, SOUTH, EAST, WEST, STAY, INTERACT = 0, 1, 2, 3, 4, 5

# 从出生点 (1,2) 走到锅正前方 (2,1) 朝北
INIT = [NORTH, EAST, NORTH]

# 单趟取洋葱+下锅（起止都在 (2,1) 朝北、空手）
ONION_CYCLE = [WEST, INTERACT, EAST, NORTH, INTERACT]

# 等待煮熟（cook_time=20，留 5 步余量）
WAIT_COOK = [STAY] * 25

# 取盘子 + 装盘（起止：(2,1)朝北 -> 装盘后仍在 (2,1)朝北，手持 soup）
GET_PLATE_AND_PLATE_SOUP = [SOUTH, WEST, SOUTH, INTERACT, NORTH, EAST, NORTH, INTERACT]

# 送餐（(2,1)朝北 -> (3,2)朝南，出餐后空手）
DELIVER = [SOUTH, EAST, SOUTH, INTERACT]

# 送餐后从 (3,2) 走回 (2,1) 朝北，形成闭环
RETURN_TO_POT = [WEST, NORTH]

RECIPE = ONION_CYCLE * 3 + WAIT_COOK + GET_PLATE_AND_PLATE_SOUP + DELIVER + RETURN_TO_POT


def build_chef_script(horizon: int, jitter: int = 0):
    script = [STAY] * jitter + list(INIT)
    while len(script) + len(RECIPE) <= horizon:
        script.extend(RECIPE)
    while len(script) < horizon:
        script.append(STAY)
    return script[:horizon]


def physical_observations(observations, ego_agent_idx):
    if ego_agent_idx == 0:
        return observations
    if ego_agent_idx == 1:
        return observations[1], observations[0]
    raise ValueError("ego_agent_idx must be 0 or 1")


def run_one_episode(layout_id, seed, episode_id, horizon, jitter, output_dir: Path):
    trajectory_path = output_dir / f"{episode_id}.trajectory.jsonl"
    manifest_path = output_dir / f"{episode_id}.manifest.json"
    started = time.time()
    manifest = {
        "episode_id": episode_id,
        "layout_id": layout_id,
        "seed": seed,
        "human_index": 0,
        "ego_agent_idx": 0,
        "horizon": horizon,
        "policy_id": "scripted_chef_greedy",
        "status": "invalid_episode",
        "termination_reason": None,
        "steps_recorded": 0,
        "trajectory_path": str(trajectory_path.resolve()),
    }
    try:
        chef_policy = ScriptedPolicy(build_chef_script(horizon, jitter))
        idle_policy = ScriptedPolicy([STAY] * horizon)

        env = OvercookedMultiEnv(layout_id, ego_agent_idx=0)
        recorder = TrajectoryRecorderV2(env, layout_id, seed, human_index=0)
        observations = recorder.reset(episode_id)
        chef_policy.reset()
        idle_policy.reset()
        done = False
        for _ in range(horizon):
            by_index = physical_observations(observations, env.ego_agent_idx)
            human_action = chef_policy.act(by_index[0])
            cole_action = idle_policy.act(by_index[1])
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
    except Exception as exc:  # noqa: BLE001
        manifest["termination_reason"] = f"{type(exc).__name__}: {exc}"
    manifest["elapsed_seconds"] = time.time() - started
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-id", default="simple")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument("--max-jitter", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifests = []
    for i in range(args.num_episodes):
        seed = args.seed_start + i
        rng = np.random.RandomState(seed)
        jitter = int(rng.randint(0, args.max_jitter + 1))
        episode_id = f"chef-{args.layout_id}-{seed}"
        manifest = run_one_episode(args.layout_id, seed, episode_id, args.horizon, jitter, args.output_dir)
        manifests.append(manifest)
        print(f"[{i+1}/{args.num_episodes}] {episode_id}: {manifest['status']} "
              f"steps={manifest['steps_recorded']} jitter={jitter}")

    valid = sum(1 for m in manifests if m["status"] == "valid")
    print(f"done: {valid}/{len(manifests)} valid episodes written to {args.output_dir}")


if __name__ == "__main__":
    main()
