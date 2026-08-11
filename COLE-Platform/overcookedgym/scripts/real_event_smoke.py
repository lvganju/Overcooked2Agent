"""Generate one real-environment positive example for each v2 task event."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


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


EXPECTED_EVENTS = [
    "ingredient_acquired",
    "ingredient_put_in_pot",
    "plate_acquired",
    "soup_plated",
    "soup_delivered",
]


def append(actions, action, count=1):
    actions.extend([action] * count)


def scripted_human_actions():
    """Player 0 completes one onion soup on the repository's simple layout."""
    actions = []
    # Reach the left onion dispenser and acquire the first onion.
    actions.extend([0, 3, 5])
    # Put the first onion into the pot.
    actions.extend([2, 0, 5])
    # Acquire and place onions two and three.
    for _ in range(2):
        actions.extend([3, 5, 2, 0, 5])
    # Wait for the full soup to cook.
    append(actions, 4, 20)
    # Walk to the dish dispenser and acquire a plate.
    actions.extend([1, 3, 1, 5])
    # Return to the pot and plate the ready soup.
    actions.extend([0, 2, 0, 5])
    # Walk to the serving location and deliver.
    actions.extend([1, 2, 1, 5])
    return actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode-id", default="real-event-smoke-001")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = OvercookedMultiEnv("simple", ego_agent_idx=0)
    recorder = TrajectoryRecorderV2(env, "simple", args.seed, human_index=0)
    recorder.reset(args.episode_id)
    for human_action in scripted_human_actions():
        _, _, done, _ = recorder.step(
            human_action=human_action,
            cole_action=4,
            final_ai_action=4,
        )
        if done:
            raise AssertionError("episode ended before scripted delivery")

    trajectory = recorder.to_dict(validate=True)
    records = trajectory["step_records"][0]
    final_state = trajectory["ep_final_states"][0]
    audit_rows = []
    for index, record in enumerate(records):
        state_after = records[index + 1]["state"] if index + 1 < len(records) else final_state
        for event in record["events"]:
            audit_rows.append(
                {
                    "source": "real_overcooked_multi_env",
                    "audit_label": "positive",
                    "episode_id": record["episode_id"],
                    "layout_id": record["layout_id"],
                    "seed": record["seed"],
                    "timestep": record["timestep"],
                    "human_index": record["human_index"],
                    "joint_action": trajectory["ep_actions"][0][index],
                    "team_reward": record["team_reward"],
                    "event": event,
                    "state_before": record["state"],
                    "state_after": state_after,
                }
            )

    counts = Counter(row["event"]["event_type"] for row in audit_rows)
    missing = [event_type for event_type in EXPECTED_EVENTS if counts[event_type] < 1]
    if missing:
        raise AssertionError("missing real event examples: {}".format(missing))

    with (output_dir / "event_audit_real_positive.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in audit_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "sample_event_episode.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(trajectory, ensure_ascii=False) + "\n")

    summary = {
        "status": "PASS",
        "episode_id": args.episode_id,
        "layout_id": "simple",
        "seed": args.seed,
        "num_steps": len(records),
        "event_counts": dict(counts),
        "all_required_event_types_present": not missing,
        "cole_model_loaded": False,
        "training_run": False,
    }
    with (logs_dir / "real_event_smoke.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
