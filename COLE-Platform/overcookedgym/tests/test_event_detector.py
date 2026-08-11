import copy
import pathlib
import sys
import unittest

OVERCOOKEDGYM_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(OVERCOOKEDGYM_DIR) not in sys.path:
    sys.path.insert(0, str(OVERCOOKEDGYM_DIR))

from environment_interfaces.event_detector import EventDetector


class Obj(object):
    def __init__(self, name, position, state=None):
        self.name = name
        self.position = tuple(position)
        self.state = state

    def to_dict(self):
        return {"name": self.name, "position": self.position, "state": self.state}


class Player(object):
    def __init__(self, position=(1, 1), orientation=(1, 0), held_object=None):
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
    def __init__(self, players, objects=None, order_list=None):
        self.players = tuple(players)
        self.objects = objects or {}
        self.order_list = order_list

    def to_dict(self):
        return {
            "players": [player.to_dict() for player in self.players],
            "objects": [obj.to_dict() for obj in self.objects.values()],
            "order_list": self.order_list,
        }


class FakeMdp(object):
    num_items_for_soup = 3
    soup_cooking_time = 20

    def __init__(self):
        self.terrain = {(2, 1): "X", (4, 1): "P", (6, 1): "S"}

    def get_terrain_type_at_pos(self, pos):
        return self.terrain.get(tuple(pos), " ")

    def get_pot_locations(self):
        return [(4, 1)]

    def get_serving_locations(self):
        return [(6, 1)]


def two_players(first, second=None):
    return [first, second or Player(position=(8, 1), orientation=(-1, 0))]


class EventDetectorTests(unittest.TestCase):
    def setUp(self):
        self.mdp = FakeMdp()
        self.detector = EventDetector(self.mdp)

    def event_types(self, before, after, team_reward=0):
        return [
            event["event_type"]
            for event in self.detector.detect(before, after, 7, team_reward=team_reward)
        ]

    def test_ingredient_acquired_requires_held_object_change(self):
        self.mdp.terrain[(2, 1)] = "O"
        before = State(two_players(Player()))
        onion = Obj("onion", (1, 1))
        after = State(two_players(Player(held_object=onion)))
        events = self.detector.detect(before, after, 7)
        self.assertEqual(["ingredient_acquired"], [event["event_type"] for event in events])
        self.assertEqual("onion", events[0]["details"]["ingredient"])
        self.assertEqual("dispenser", events[0]["details"]["source"])

        unchanged = self.detector.detect(before, copy.deepcopy(before), 8)
        self.assertEqual([], unchanged)

    def test_ingredient_put_in_pot_requires_matching_pot_delta(self):
        onion = Obj("onion", (3, 1))
        before = State(two_players(Player(position=(3, 1), held_object=onion)))
        soup = Obj("soup", (4, 1), ("onion", 1, 0))
        after = State(two_players(Player(position=(3, 1))), {(4, 1): soup})
        events = self.detector.detect(before, after, 3)
        self.assertEqual(["ingredient_put_in_pot"], [event["event_type"] for event in events])
        self.assertEqual(0, events[0]["details"]["count_before"])
        self.assertEqual(1, events[0]["details"]["count_after"])

        no_pot_change = State(two_players(Player(position=(3, 1))))
        self.assertEqual([], self.detector.detect(before, no_pot_change, 4))

    def test_plate_acquired(self):
        self.mdp.terrain[(2, 1)] = "D"
        before = State(two_players(Player()))
        after = State(two_players(Player(held_object=Obj("dish", (1, 1)))))
        self.assertEqual(["plate_acquired"], self.event_types(before, after))

    def test_soup_plated_requires_ready_pot_to_disappear(self):
        dish = Obj("dish", (3, 1))
        ready = Obj("soup", (4, 1), ("onion", 3, 20))
        before = State(two_players(Player(position=(3, 1), held_object=dish)), {(4, 1): ready})
        plated = Obj("soup", (3, 1), ("onion", 3, 20))
        after = State(two_players(Player(position=(3, 1), held_object=plated)))
        events = self.detector.detect(before, after, 9)
        self.assertEqual(["soup_plated"], [event["event_type"] for event in events])

        not_ready = State(
            two_players(Player(position=(3, 1), held_object=dish)),
            {(4, 1): Obj("soup", (4, 1), ("onion", 3, 19))},
        )
        self.assertEqual([], self.detector.detect(not_ready, after, 10))

    def test_soup_delivered_requires_serving_and_valid_order_delta(self):
        soup = Obj("soup", (5, 1), ("onion", 3, 20))
        before = State(two_players(Player(position=(5, 1), held_object=soup)), order_list=["onion", "any"])
        after = State(two_players(Player(position=(5, 1))), order_list=["any"])
        events = self.detector.detect(before, after, 11, team_reward=20)
        self.assertEqual(["soup_delivered"], [event["event_type"] for event in events])
        self.assertEqual(["onion", "any"], events[0]["details"]["orders_before"])

        unchanged_orders = State(two_players(Player(position=(5, 1))), order_list=["onion", "any"])
        self.assertEqual([], self.detector.detect(before, unchanged_orders, 12))

    def test_infinite_order_delivery_is_state_and_terrain_based(self):
        soup = Obj("soup", (5, 1), ("tomato", 3, 20))
        before = State(two_players(Player(position=(5, 1), held_object=soup)), order_list=None)
        after = State(two_players(Player(position=(5, 1))), order_list=None)
        self.assertEqual(["soup_delivered"], self.event_types(before, after, team_reward=20))

    def test_soup_delivery_requires_positive_reward_evidence(self):
        soup = Obj("soup", (5, 1), ("onion", 3, 20))
        before = State(two_players(Player(position=(5, 1), held_object=soup)), order_list=["onion"])
        after = State(two_players(Player(position=(5, 1))), order_list=[])
        self.assertEqual([], self.detector.detect(before, after, 13, team_reward=0))

    def test_same_state_never_emits_event(self):
        state = State(two_players(Player()))
        self.assertEqual([], self.detector.detect(state, copy.deepcopy(state), 0))


if __name__ == "__main__":
    unittest.main()
