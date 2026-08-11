"""Run a real OvercookedMultiEnv episode with and without the v2 recorder.

This script uses fixed stay actions and never loads or trains a policy.
"""

import argparse
import json
import os
import platform
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

from overcookedgym.environment_interfaces import TrajectoryRecorderV2
from overcookedgym.overcooked import OvercookedMultiEnv


def observations_equal(left, right):
    return len(left) == len(right) and all(
        np.array_equal(left_item, right_item)
        for left_item, right_item in zip(left, right)
    )


def state_dict(env):
    return env.base_env.state.to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layout-id", default="simple")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode-id", default="real-env-smoke-001")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    started = time.time()
    raw_env = OvercookedMultiEnv(args.layout_id, ego_agent_idx=0)
    wrapped_base = OvercookedMultiEnv(args.layout_id, ego_agent_idx=0)
    recorder = TrajectoryRecorderV2(
        wrapped_base,
        layout_id=args.layout_id,
        seed=args.seed,
        human_index=0,
    )

    raw_reset = raw_env.multi_reset()
    wrapped_reset = recorder.reset(args.episode_id)
    reset_equal = observations_equal(raw_reset, wrapped_reset)
    initial_state_equal = state_dict(raw_env) == state_dict(wrapped_base)

    step_results = []
    timestep = 0
    while True:
        raw_result = raw_env.multi_step(4, 4)
        wrapped_result = recorder.step(4, 4, final_ai_action=4)
        obs_equal = observations_equal(raw_result[0], wrapped_result[0])
        reward_equal = raw_result[1] == wrapped_result[1]
        done_equal = raw_result[2] == wrapped_result[2]
        info_equal = raw_result[3] == wrapped_result[3]
        state_equal = state_dict(raw_env) == state_dict(wrapped_base)
        step_results.append(
            {
                "timestep": timestep,
                "observation_equal": obs_equal,
                "reward_equal": reward_equal,
                "done_equal": done_equal,
                "info_equal": info_equal,
                "state_equal": state_equal,
            }
        )
        if not all((obs_equal, reward_equal, done_equal, info_equal, state_equal)):
            raise AssertionError("wrapped environment diverged at timestep {}".format(timestep))
        if raw_result[2]:
            break
        timestep += 1

    trajectory = recorder.to_dict(validate=True)
    final_state_equal = state_dict(raw_env) == state_dict(wrapped_base)
    events_empty = all(
        not events for episode in trajectory["ep_events"] for events in episode
    )
    report = {
        "status": "PASS",
        "layout_id": args.layout_id,
        "seed": args.seed,
        "episode_id": args.episode_id,
        "joint_action": [4, 4],
        "num_steps": len(step_results),
        "reset_observation_equal": reset_equal,
        "initial_state_equal": initial_state_equal,
        "final_state_equal": final_state_equal,
        "all_step_comparisons_pass": all(
            all(
                step[key]
                for key in (
                    "observation_equal",
                    "reward_equal",
                    "done_equal",
                    "info_equal",
                    "state_equal",
                )
            )
            for step in step_results
        ),
        "events_empty_for_all_stay_episode": events_empty,
        "steps": step_results,
    }
    if not all(
        (
            reset_equal,
            initial_state_equal,
            final_state_equal,
            report["all_step_comparisons_pass"],
            events_empty,
        )
    ):
        raise AssertionError("real environment smoke verification failed")

    with (output_dir / "sample_episode.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
    with (output_dir / "wrapped_vs_unwrapped_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    elapsed = time.time() - started
    log_lines = [
        "status=PASS",
        "command={}".format(" ".join(sys.argv)),
        "working_directory={}".format(os.getcwd()),
        "python={}".format(sys.version.replace("\n", " ")),
        "platform={}".format(platform.platform()),
        "layout_id={}".format(args.layout_id),
        "seed={}".format(args.seed),
        "episode_id={}".format(args.episode_id),
        "joint_action=[4, 4]",
        "num_steps={}".format(len(step_results)),
        "elapsed_seconds={:.6f}".format(elapsed),
        "reset_observation_equal={}".format(reset_equal),
        "initial_state_equal={}".format(initial_state_equal),
        "final_state_equal={}".format(final_state_equal),
        "all_step_comparisons_pass={}".format(
            report["all_step_comparisons_pass"]
        ),
        "events_empty_for_all_stay_episode={}".format(events_empty),
        "cole_model_loaded=False",
        "training_run=False",
        "exit_code=0",
    ]
    (logs_dir / "real_env_smoke.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )
    print("\n".join(log_lines))


if __name__ == "__main__":
    main()
