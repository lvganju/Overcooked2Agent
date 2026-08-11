"""Schema helpers for environment trajectory format v2.

The v2 format keeps the existing ``ep_states``, ``ep_actions`` and
``ep_rewards`` arrays and adds dones, events, metadata, and a per-step view.
"""

import copy
import json
import numbers
from typing import Any, Dict, Iterable, Mapping


SCHEMA_VERSION = "trajectory_schema_v2"
ACTION_ID_TO_NAME = {
    0: "up",
    1: "down",
    2: "right",
    3: "left",
    4: "stay",
    5: "interact",
}
EVENT_TYPES = {
    "ingredient_acquired",
    "ingredient_put_in_pot",
    "plate_acquired",
    "soup_plated",
    "soup_delivered",
}
STEP_FIELDS = {
    "episode_id",
    "timestep",
    "layout_id",
    "seed",
    "human_index",
    "state",
    "human_action",
    "cole_action",
    "final_ai_action",
    "team_reward",
    "done",
    "events",
}


def state_to_dict(state: Any) -> Dict[str, Any]:
    """Return a detached JSON-compatible representation of a real state."""
    if hasattr(state, "to_dict"):
        value = state.to_dict()
    elif isinstance(state, Mapping):
        value = dict(state)
    else:
        raise TypeError("state must expose to_dict() or be a mapping")
    # The round trip also normalizes tuples to JSON arrays and prevents callers
    # from mutating the recorded state after the transition.
    return json.loads(json.dumps(value))


def validate_action(action: Any, field_name: str = "action") -> None:
    if isinstance(action, bool) or not isinstance(action, int):
        raise TypeError("{} must be an integer".format(field_name))
    if action not in ACTION_ID_TO_NAME:
        raise ValueError("{} must be in range 0..5".format(field_name))


def validate_event(event: Mapping[str, Any]) -> None:
    required = {"event_type", "timestep", "agent_index", "details"}
    missing = required.difference(event)
    if missing:
        raise ValueError("event missing fields: {}".format(sorted(missing)))
    if event["event_type"] not in EVENT_TYPES:
        raise ValueError("unknown event_type: {}".format(event["event_type"]))
    if not isinstance(event["timestep"], int) or event["timestep"] < 0:
        raise ValueError("event timestep must be a non-negative integer")
    if event["agent_index"] not in (0, 1):
        raise ValueError("event agent_index must be 0 or 1")
    if not isinstance(event["details"], Mapping):
        raise TypeError("event details must be a mapping")


def validate_step_record(record: Mapping[str, Any]) -> None:
    missing = STEP_FIELDS.difference(record)
    if missing:
        raise ValueError("step record missing fields: {}".format(sorted(missing)))
    if not isinstance(record["episode_id"], str) or not record["episode_id"]:
        raise ValueError("episode_id must be a non-empty string")
    if not isinstance(record["timestep"], int) or record["timestep"] < 0:
        raise ValueError("timestep must be a non-negative integer")
    if not isinstance(record["layout_id"], str) or not record["layout_id"]:
        raise ValueError("layout_id must be a non-empty string")
    if record["human_index"] not in (0, 1):
        raise ValueError("human_index must be 0 or 1")
    if isinstance(record["seed"], bool) or not isinstance(record["seed"], int):
        raise TypeError("seed must be an integer")
    validate_action(record["human_action"], "human_action")
    validate_action(record["cole_action"], "cole_action")
    validate_action(record["final_ai_action"], "final_ai_action")
    if not isinstance(record["done"], bool):
        raise TypeError("done must be bool")
    if isinstance(record["team_reward"], bool) or not isinstance(
        record["team_reward"], numbers.Real
    ):
        raise TypeError("team_reward must be numeric")
    if not isinstance(record["events"], list):
        raise TypeError("events must be a list")
    for event in record["events"]:
        validate_event(event)
    # Raises when a value cannot be represented in a trajectory JSON file.
    json.dumps(record)


def validate_trajectory(trajectory: Mapping[str, Any]) -> None:
    if trajectory.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or missing schema_version")
    episode_keys = (
        "ep_states",
        "ep_actions",
        "ep_rewards",
        "ep_dones",
        "ep_events",
        "step_records",
    )
    for key in episode_keys:
        if not isinstance(trajectory.get(key), list):
            raise TypeError("{} must be a list of episodes".format(key))
    lengths = [len(trajectory[key]) for key in episode_keys]
    if len(set(lengths)) != 1:
        raise ValueError("trajectory episode arrays have different lengths")
    final_states = trajectory.get("ep_final_states")
    if not isinstance(final_states, list) or len(final_states) != lengths[0]:
        raise ValueError("ep_final_states must contain one final state per episode")
    for episode_index in range(lengths[0]):
        step_lengths = [len(trajectory[key][episode_index]) for key in episode_keys]
        if len(set(step_lengths)) != 1:
            raise ValueError("episode {} arrays have different lengths".format(episode_index))
        records = trajectory["step_records"][episode_index]
        if not records:
            raise ValueError("episode {} must contain at least one step".format(episode_index))
        episode_id = records[0]["episode_id"]
        metadata = (
            records[0]["layout_id"],
            records[0]["seed"],
            records[0]["human_index"],
        )
        for timestep, record in enumerate(records):
            validate_step_record(record)
            if record["timestep"] != timestep:
                raise ValueError("episode {} timestep is not contiguous".format(episode_index))
            if record["episode_id"] != episode_id:
                raise ValueError("episode {} contains multiple episode_ids".format(episode_index))
            if (record["layout_id"], record["seed"], record["human_index"]) != metadata:
                raise ValueError("episode {} metadata changes within episode".format(episode_index))
            if record["state"] != trajectory["ep_states"][episode_index][timestep]:
                raise ValueError("step record state differs from ep_states")
            expected_action = [None, None]
            expected_action[record["human_index"]] = record["human_action"]
            expected_action[1 - record["human_index"]] = record["final_ai_action"]
            if expected_action != trajectory["ep_actions"][episode_index][timestep]:
                raise ValueError("step record actions differ from ep_actions")
            if record["team_reward"] != trajectory["ep_rewards"][episode_index][timestep]:
                raise ValueError("step record reward differs from ep_rewards")
            if record["done"] != trajectory["ep_dones"][episode_index][timestep]:
                raise ValueError("step record done differs from ep_dones")
            if record["events"] != trajectory["ep_events"][episode_index][timestep]:
                raise ValueError("step record events differ from ep_events")
        if any(record["done"] for record in records[:-1]):
            raise ValueError("done=True may only appear on the final recorded step")
        if not isinstance(final_states[episode_index], Mapping):
            raise TypeError("each ep_final_states entry must be a state mapping")
    json.dumps(trajectory)


def detached(value: Any) -> Any:
    """Return a defensive copy for recorder exports."""
    return copy.deepcopy(value)
