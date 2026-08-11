"""Run one programmatic Overcooked episode and write trajectory v2 + manifest."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (
    REPO_ROOT,
    REPO_ROOT / "overcookedgym" / "human_aware_rl",
    REPO_ROOT / "overcookedgym" / "human_aware_rl" / "overcooked_ai",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from overcookedgym.environment_interfaces import (
    ConstantPolicy,
    ScriptedPolicy,
    ScriptExhaustedError,
    TrajectoryRecorderV2,
)
from overcookedgym.overcooked import OvercookedMultiEnv


def parse_actions(text):
    if text is None:
        return None
    try:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError:
        raise ValueError("action scripts must be comma-separated integers")


def build_policy(script, constant):
    actions = parse_actions(script)
    return ScriptedPolicy(actions) if actions is not None else ConstantPolicy(constant)


def physical_observations(observations, ego_agent_idx):
    if ego_agent_idx == 0:
        return observations
    if ego_agent_idx == 1:
        return observations[1], observations[0]
    raise ValueError("ego_agent_idx must be 0 or 1")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-id", default="simple")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--human-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--ego-agent-idx", type=int, choices=(0, 1), default=0)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--human-script")
    parser.add_argument("--cole-script")
    parser.add_argument("--human-constant", type=int, default=4)
    parser.add_argument("--cole-constant", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.horizon <= 0:
        parser.error("--horizon must be positive")

    trajectory_path = args.output_dir / "{}.trajectory.jsonl".format(args.episode_id)
    manifest_path = args.output_dir / "{}.manifest.json".format(args.episode_id)
    started = time.time()
    manifest = {
        "episode_id": args.episode_id,
        "layout_id": args.layout_id,
        "seed": args.seed,
        "human_index": args.human_index,
        "ego_agent_idx": args.ego_agent_idx,
        "horizon": args.horizon,
        "status": "invalid_episode",
        "termination_reason": None,
        "steps_recorded": 0,
        "trajectory_path": str(trajectory_path.resolve()),
        "controller_enabled": False,
        "models_loaded": False,
        "training_run": False,
    }
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        focal_policy = build_policy(args.human_script, args.human_constant)
        cole_policy = build_policy(args.cole_script, args.cole_constant)
        env = OvercookedMultiEnv(args.layout_id, ego_agent_idx=args.ego_agent_idx)
        recorder = TrajectoryRecorderV2(
            env, args.layout_id, args.seed, human_index=args.human_index
        )
        observations = recorder.reset(args.episode_id)
        focal_policy.reset()
        cole_policy.reset()
        done = False
        for _ in range(args.horizon):
            by_index = physical_observations(observations, env.ego_agent_idx)
            human_action = focal_policy.act(by_index[args.human_index])
            cole_action = cole_policy.act(by_index[1 - args.human_index])
            observations, _, done, _ = recorder.step(human_action, cole_action)
            if done:
                break
        trajectory = recorder.to_dict(validate=True)
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_path.write_text(
            json.dumps(trajectory, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        manifest["steps_recorded"] = len(trajectory["step_records"][0])
        manifest["status"] = "valid"
        manifest["termination_reason"] = "environment_done" if done else "horizon_reached"
    except ScriptExhaustedError as exc:
        manifest["termination_reason"] = "script_exhausted: {}".format(exc)
    except Exception as exc:
        manifest["termination_reason"] = "{}: {}".format(type(exc).__name__, exc)
        manifest["elapsed_seconds"] = time.time() - started
        write_json(manifest_path, manifest)
        raise
    manifest["elapsed_seconds"] = time.time() - started
    write_json(manifest_path, manifest)
    print("single episode: {}".format(manifest["status"].upper()))
    print("episode_id={} steps={} reason={}".format(
        args.episode_id, manifest["steps_recorded"], manifest["termination_reason"]
    ))
    print("trajectory={}".format(trajectory_path if trajectory_path.exists() else "NOT_WRITTEN"))
    print("manifest={}".format(manifest_path))
    print("No COLE model was loaded and no training was run.")
    return 0 if manifest["status"] == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
