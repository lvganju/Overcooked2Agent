"""Replay recorded transitions through the original MDP and compare state/reward."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (
    REPO_ROOT,
    REPO_ROOT / "overcookedgym" / "human_aware_rl",
    REPO_ROOT / "overcookedgym" / "human_aware_rl" / "overcooked_ai",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcookedgym.environment_interfaces.trajectory_schema_v2 import (
    state_to_dict,
    validate_trajectory,
)


REWARD_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}


def load_documents(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return [json.loads(text)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    args = parser.parse_args()

    checked = 0
    for document_index, trajectory in enumerate(load_documents(args.trajectory)):
        validate_trajectory(trajectory)
        for episode_index, records in enumerate(trajectory["step_records"]):
            layout_id = records[0]["layout_id"]
            mdp = OvercookedGridworld.from_layout_name(
                layout_name=layout_id, rew_shaping_params=REWARD_SHAPING_PARAMS
            )
            final_state = trajectory["ep_final_states"][episode_index]
            for timestep, record in enumerate(records):
                before = OvercookedState.from_dict(record["state"])
                joint_ids = trajectory["ep_actions"][episode_index][timestep]
                joint_action = tuple(Action.INDEX_TO_ACTION[action] for action in joint_ids)
                after, sparse_reward, shaped_reward = mdp.get_state_transition(
                    before, joint_action
                )
                expected_after = (
                    records[timestep + 1]["state"]
                    if timestep + 1 < len(records)
                    else final_state
                )
                if state_to_dict(after) != expected_after:
                    raise AssertionError(
                        "state mismatch at document {} episode {} timestep {}".format(
                            document_index, episode_index, timestep
                        )
                    )
                expected_reward = sparse_reward + shaped_reward
                if expected_reward != record["team_reward"]:
                    raise AssertionError(
                        "reward mismatch at document {} episode {} timestep {}: {} != {}".format(
                            document_index,
                            episode_index,
                            timestep,
                            expected_reward,
                            record["team_reward"],
                        )
                    )
                checked += 1
    print("episode replay: PASS")
    print("transitions_checked={}".format(checked))
    print("Original MDP transition and reward logic were used unchanged.")
    print("No COLE model was loaded and no training was run.")


if __name__ == "__main__":
    main()
