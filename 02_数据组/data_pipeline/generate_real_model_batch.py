"""generate_real_model_batch.py — D5 真实模型对战批量生成（数据组）。

加载 COLE-Platform 提供的 TF SavedModel 格式策略（SP/PBT/FCP/MEP/COLE），复用
`pantheonrl/tf_utils.py::get_agent_from_saved_model`（环境组代码库自带的加载器，
未做任何修改），在 random1 布局上做自对弈（self-play：双方用同一 checkpoint），
产生真实模型行为轨迹，写入 trajectory_schema_v2 格式。

**已知限制（范围裁剪，需写入 quality_report.json）**：
- BC（behavioural cloning）模型未接入。其加载路径（human_aware_rl/imitation/
  behavioural_cloning.py::load_bc_model_from_path）依赖 stable-baselines GAIL
  的 `.load()`，与本项目 requirements 中的 stable-baselines3/torch 生态不完全
  对齐，需要单独适配和验证，风险高于本轮时间预算，故本轮只接入 SP/PBT/FCP/MEP/
  COLE 共 5 个模型。
- 本轮为自对弈（human 和队友用同一 checkpoint），不是"5 个模型互相对战"的全
  交叉矩阵；交叉对战（如 SP vs COLE）留作后续扩展，当前先验证单模型自对弈链路
  的正确性与事件触发率。
- 复用环境组 `tf_utils.get_agent_from_saved_model` 时发现该模型的 TF 计算图固定
  batch 维度为 30（sim_threads=30 硬编码进 checkpoint），因此每步推理都要 pad
  到 30 份重复输入、只取第 0 份输出——这是环境组遗留实现细节，不是本组引入的
  bug，此处原样复用不做修改。

用法（cole-platform 环境）：
    python generate_real_model_batch.py --model SP --num-episodes 5 \
        --horizon 400 --seed-start 3000 --output-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

COLE_PLATFORM_ROOT = Path(
    r"C:\Users\36724\PycharmProjects\WelcomeScreen\Overcooked2Agent\COLE-Platform"
)
MODELS_ROOT = Path(r"C:\Users\36724\PycharmProjects\WelcomeScreen\Overcooked2Agent\models\random1")
for path in (
    COLE_PLATFORM_ROOT,
    COLE_PLATFORM_ROOT / "overcookedgym" / "human_aware_rl",
    COLE_PLATFORM_ROOT / "overcookedgym" / "human_aware_rl" / "overcooked_ai",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from overcookedgym.environment_interfaces import TrajectoryRecorderV2  # noqa: E402
from overcookedgym.overcooked import OvercookedMultiEnv  # noqa: E402
from overcooked_ai_py.mdp.actions import Action  # noqa: E402
from pantheonrl.tf_utils import get_agent_from_saved_model  # noqa: E402

SUPPORTED_MODELS = ("SP", "PBT", "FCP", "MEP", "COLE")
SIM_THREADS = 30  # 与 checkpoint 固定的 batch 维度一致（见模块 docstring 说明）


def action_tuple_to_index(action_tuple) -> int:
    return Action.ACTION_TO_INDEX[action_tuple]


def load_model_agent(model_name: str, mdp, agent_index: int):
    model_dir = MODELS_ROOT / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"model dir not found: {model_dir}")
    agent = get_agent_from_saved_model(str(model_dir), SIM_THREADS)
    agent.set_mdp(mdp)
    agent.set_agent_index(agent_index)
    return agent


def run_one_episode(model_name, layout_id, seed, human_index, episode_id, horizon, output_dir: Path):
    trajectory_path = output_dir / f"{episode_id}.trajectory.jsonl"
    manifest_path = output_dir / f"{episode_id}.manifest.json"
    started = time.time()
    manifest = {
        "episode_id": episode_id,
        "layout_id": layout_id,
        "seed": seed,
        "human_index": human_index,
        "ego_agent_idx": 0,
        "horizon": horizon,
        "policy_id": f"real_model_selfplay_{model_name}",
        "status": "invalid_episode",
        "termination_reason": None,
        "steps_recorded": 0,
        "trajectory_path": str(trajectory_path.resolve()),
    }
    try:
        env = OvercookedMultiEnv(layout_id, ego_agent_idx=0)
        recorder = TrajectoryRecorderV2(env, layout_id, seed, human_index=human_index)
        recorder.reset(episode_id)

        agent0 = load_model_agent(model_name, env.mdp, agent_index=0)
        agent1 = load_model_agent(model_name, env.mdp, agent_index=1)
        agent0.reset()
        agent1.reset()

        done = False
        for _ in range(horizon):
            raw_state = env.base_env.state
            action0 = action_tuple_to_index(agent0.action(raw_state))
            action1 = action_tuple_to_index(agent1.action(raw_state))
            if human_index == 0:
                human_action, cole_action = action0, action1
            else:
                human_action, cole_action = action1, action0
            _, _, done, _ = recorder.step(human_action, cole_action)
            if done:
                break

        trajectory = recorder.to_dict(validate=True)
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_path.write_text(json.dumps(trajectory, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest["steps_recorded"] = len(trajectory["step_records"][0])
        manifest["status"] = "valid"
        manifest["termination_reason"] = "environment_done" if done else "horizon_reached"
    except Exception as exc:  # noqa: BLE001
        manifest["termination_reason"] = f"{type(exc).__name__}: {exc}"
    manifest["elapsed_seconds"] = time.time() - started
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--layout-id", default="random1")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed-start", type=int, default=3000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifests = []
    for i in range(args.num_episodes):
        seed = args.seed_start + i
        episode_id = f"real-{args.model.lower()}-{args.layout_id}-{seed}"
        human_index = i % 2
        manifest = run_one_episode(
            args.model, args.layout_id, seed, human_index, episode_id, args.horizon, args.output_dir
        )
        manifests.append(manifest)
        print(f"[{i+1}/{args.num_episodes}] {episode_id}: {manifest['status']} "
              f"steps={manifest['steps_recorded']} reason={manifest['termination_reason']}")

    valid = sum(1 for m in manifests if m["status"] == "valid")
    print(f"done: {valid}/{len(manifests)} valid episodes written to {args.output_dir}")


if __name__ == "__main__":
    main()
