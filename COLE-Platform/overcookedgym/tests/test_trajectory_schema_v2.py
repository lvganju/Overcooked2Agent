import json
import pathlib
import sys
import unittest

OVERCOOKEDGYM_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(OVERCOOKEDGYM_DIR) not in sys.path:
    sys.path.insert(0, str(OVERCOOKEDGYM_DIR))

from environment_interfaces.trajectory_recorder import TrajectoryRecorderV2
from environment_interfaces.trajectory_schema_v2 import (
    ACTION_ID_TO_NAME,
    validate_trajectory,
)


class Obj(object):
    def __init__(self, name, position, state=None):
        self.name = name
        self.position = tuple(position)
        self.state = state

    def to_dict(self):
        return {"name": self.name, "position": self.position, "state": self.state}


class Player(object):
    def __init__(self, position, orientation, held_object=None):
        self.position = tuple(position)
        self.orientation = tuple(orientation)
        self.held_object = held_object

    def to_dict(self):
        return {
            "position": self.position,
            "orientation": self.orientation,
            "held_object": None if self.held_object is None else self.held_object.to_dict(),
        }


class State(object):
    def __init__(self, players):
        self.players = tuple(players)
        self.objects = {}
        self.order_list = None

    def to_dict(self):
        return {
            "players": [player.to_dict() for player in self.players],
            "objects": [],
            "order_list": self.order_list,
        }


class FakeMdp(object):
    num_items_for_soup = 3
    soup_cooking_time = 20

    def get_terrain_type_at_pos(self, pos):
        return "O" if tuple(pos) == (2, 1) else " "

    def get_pot_locations(self):
        return []

    def get_serving_locations(self):
        return []


class BaseEnv(object):
    def __init__(self):
        self.state = None


class FakeOvercookedMultiEnv(object):
    def __init__(self):
        self.mdp = FakeMdp()
        self.base_env = BaseEnv()
        self.ego_agent_idx = 1
        self.last_call = None
        self.reset_calls = 0

    def multi_reset(self):
        self.reset_calls += 1
        self.base_env.state = State(
            [Player((1, 1), (1, 0)), Player((4, 1), (-1, 0))]
        )
        return ("human-observation", "ai-observation")

    def multi_step(self, ego_action, alt_action):
        self.last_call = (ego_action, alt_action)
        self.base_env.state = State(
            [
                Player((1, 1), (1, 0), Obj("onion", (1, 1))),
                Player((4, 1), (-1, 0)),
            ]
        )
        return ("next-ego", "next-alt"), (3, 3), False, {"original": True}


class TrajectorySchemaTests(unittest.TestCase):
    def test_action_mapping_is_unchanged(self):
        self.assertEqual(
            {0: "up", 1: "down", 2: "right", 3: "left", 4: "stay", 5: "interact"},
            ACTION_ID_TO_NAME,
        )

    def test_recorder_wraps_reset_and_step_without_changing_return_values(self):
        env = FakeOvercookedMultiEnv()
        recorder = TrajectoryRecorderV2(env, "simple", seed=7, human_index=0)
        reset_observation = recorder.reset("episode-test")
        self.assertEqual(("human-observation", "ai-observation"), reset_observation)

        result = recorder.step(human_action=5, cole_action=4)
        self.assertEqual(
            (("next-ego", "next-alt"), (3, 3), False, {"original": True}),
            result,
        )
        # ego_agent_idx=1, so the AI index-1 action is the ego action and the
        # human index-0 action is the alt action.
        self.assertEqual((4, 5), env.last_call)

        trajectory = recorder.to_dict()
        validate_trajectory(trajectory)
        record = trajectory["step_records"][0][0]
        self.assertEqual(3, record["team_reward"])
        self.assertEqual(4, record["cole_action"])
        self.assertEqual(4, record["final_ai_action"])
        self.assertEqual("ingredient_acquired", record["events"][0]["event_type"])
        self.assertEqual([5, 4], trajectory["ep_actions"][0][0])
        self.assertEqual("onion", trajectory["ep_final_states"][0]["players"][0]["held_object"]["name"])
        json.dumps(trajectory)

    def test_validator_rejects_non_contiguous_timestep(self):
        env = FakeOvercookedMultiEnv()
        recorder = TrajectoryRecorderV2(env, "simple", seed=7, human_index=0)
        recorder.reset("episode-continuity")
        recorder.step(5, 4)
        trajectory = recorder.to_dict()
        trajectory["step_records"][0][0]["timestep"] = 2
        with self.assertRaises(ValueError):
            validate_trajectory(trajectory)

    def test_final_ai_action_can_be_recorded_separately(self):
        env = FakeOvercookedMultiEnv()
        recorder = TrajectoryRecorderV2(env, "simple", seed=7, human_index=0)
        recorder.reset("episode-override")
        recorder.step(human_action=5, cole_action=2, final_ai_action=4)
        record = recorder.to_dict()["step_records"][0][0]
        self.assertEqual(2, record["cole_action"])
        self.assertEqual(4, record["final_ai_action"])
        self.assertEqual((4, 5), env.last_call)

    def test_step_requires_reset(self):
        recorder = TrajectoryRecorderV2(
            FakeOvercookedMultiEnv(), "simple", seed=0, human_index=0
        )
        with self.assertRaises(RuntimeError):
            recorder.step(4, 4)

    def test_non_shared_reward_is_rejected_not_guessed(self):
        with self.assertRaises(ValueError):
            TrajectoryRecorderV2._team_reward((1, 2))


if __name__ == "__main__":
    unittest.main()
