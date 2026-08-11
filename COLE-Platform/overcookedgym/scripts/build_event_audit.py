"""Build and verify the 5x positive/negative task event audit set."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (
    REPO_ROOT,
    REPO_ROOT / "overcookedgym" / "human_aware_rl",
    REPO_ROOT / "overcookedgym" / "human_aware_rl" / "overcooked_ai",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from overcooked_ai_py.mdp.overcooked_mdp import (
    ObjectState,
    OvercookedGridworld,
    OvercookedState,
    PlayerState,
)
from overcookedgym.environment_interfaces import EventDetector
from overcookedgym.environment_interfaces.trajectory_schema_v2 import state_to_dict


EVENT_TYPES = [
    "ingredient_acquired",
    "ingredient_put_in_pot",
    "plate_acquired",
    "soup_plated",
    "soup_delivered",
]


def obj(name, position, extra=None):
    return ObjectState(name, position, extra)


def make_state(held=None, position=(1, 1), orientation=(-1, 0), objects=None, orders=None):
    if held is not None:
        held.position = position
    players = [
        PlayerState(position, orientation, held),
        PlayerState((3, 1), (1, 0), None),
    ]
    return OvercookedState(players, objects or {}, all_orders=orders)


def detect(detector, before, after, timestep, reward):
    return detector.detect(before, after, timestep, team_reward=reward)


def audit_row(source, label, target, reason, before, after, reward, events, index):
    return {
        "audit_id": "{}-{}-{:02d}".format(target, label, index),
        "source": source,
        "audit_label": label,
        "target_event_type": target,
        "reason": reason,
        "layout_id": "simple",
        "team_reward": reward,
        "observed_events": events,
        "state_before": state_to_dict(before),
        "state_after": state_to_dict(after),
    }


def positive_transition(event_type, variant):
    if event_type == "ingredient_acquired":
        before = make_state()
        after = make_state(obj("onion", (1, 1)))
        return before, after, 0, "empty hand becomes onion"
    if event_type == "ingredient_put_in_pot":
        pot = (2, 0)
        count_before = variant % 2
        before_objects = {}
        if count_before:
            before_objects[pot] = obj("soup", pot, ("onion", count_before, 0))
        after_objects = {
            pot: obj("soup", pot, ("onion", count_before + 1, 0))
        }
        before = make_state(
            obj("onion", (2, 1)), position=(2, 1), orientation=(0, -1), objects=before_objects
        )
        after = make_state(None, position=(2, 1), orientation=(0, -1), objects=after_objects)
        return before, after, 0, "held onion disappears and faced pot count increases"
    if event_type == "plate_acquired":
        before = make_state(position=(1, 2), orientation=(0, 1))
        after = make_state(
            obj("dish", (1, 2)), position=(1, 2), orientation=(0, 1)
        )
        return before, after, 0, "empty hand becomes dish"
    if event_type == "soup_plated":
        pot = (2, 0)
        ready = obj("soup", pot, ("onion", 3, 20))
        before = make_state(
            obj("dish", (2, 1)),
            position=(2, 1),
            orientation=(0, -1),
            objects={pot: ready},
        )
        after = make_state(
            obj("soup", (2, 1), ("onion", 3, 20)),
            position=(2, 1),
            orientation=(0, -1),
        )
        return before, after, 0, "dish becomes ready soup and faced pot empties"
    if event_type == "soup_delivered":
        before = make_state(
            obj("soup", (3, 2), ("onion", 3, 20)),
            position=(3, 2),
            orientation=(0, 1),
            orders=None,
        )
        after = make_state(None, position=(3, 2), orientation=(0, 1), orders=None)
        return before, after, 20, "ready soup disappears at serving with positive reward"
    raise ValueError(event_type)


def negative_transitions(event_type):
    pot = (2, 0)
    ready = obj("soup", pot, ("onion", 3, 20))
    if event_type == "ingredient_acquired":
        return [
            (make_state(), make_state(), 0, "no held-object change"),
            (make_state(), make_state(obj("dish", (1, 1))), 0, "acquired dish, not ingredient"),
            (make_state(obj("onion", (1, 1))), make_state(obj("onion", (1, 1))), 0, "already held ingredient"),
            (make_state(), make_state(None, position=(1, 2)), 0, "movement without acquisition"),
            (make_state(obj("dish", (1, 1))), make_state(obj("onion", (1, 1))), 0, "hand was not empty before"),
        ]
    if event_type == "ingredient_put_in_pot":
        return [
            (make_state(obj("onion", (2, 1)), position=(2, 1), orientation=(0, -1)), make_state(None, position=(2, 1), orientation=(0, -1)), 0, "ingredient disappears without pot delta"),
            (make_state(obj("onion", (2, 1)), position=(2, 1), orientation=(0, -1), objects={pot: obj("soup", pot, ("onion", 1, 0))}), make_state(None, position=(2, 1), orientation=(0, -1), objects={pot: obj("soup", pot, ("onion", 1, 0))}), 0, "pot count does not increase"),
            (make_state(obj("onion", (2, 1)), position=(2, 1), orientation=(0, -1)), make_state(None, position=(2, 1), orientation=(0, -1), objects={pot: obj("soup", pot, ("tomato", 1, 0))}), 0, "pot ingredient type does not match"),
            (make_state(obj("onion", (2, 1)), position=(2, 1), orientation=(0, -1)), make_state(obj("onion", (2, 1)), position=(2, 1), orientation=(0, -1), objects={pot: obj("soup", pot, ("onion", 1, 0))}), 0, "ingredient remains held"),
            (make_state(obj("onion", (1, 1)), position=(1, 1), orientation=(0, 1)), make_state(None, position=(1, 1), orientation=(0, 1), objects={pot: obj("soup", pot, ("onion", 1, 0))}), 0, "agent is not facing changed pot"),
        ]
    if event_type == "plate_acquired":
        return [
            (make_state(), make_state(), 0, "no held-object change"),
            (make_state(), make_state(obj("onion", (1, 1))), 0, "acquired ingredient, not plate"),
            (make_state(obj("dish", (1, 1))), make_state(None), 0, "plate was dropped"),
            (make_state(obj("onion", (1, 1))), make_state(obj("dish", (1, 1))), 0, "hand was not empty before"),
            (make_state(), make_state(objects={(2, 3): obj("dish", (2, 3))}), 0, "dish appears in world, not in hand"),
        ]
    if event_type == "soup_plated":
        return [
            (make_state(obj("dish", (2, 1)), position=(2, 1), orientation=(0, -1), objects={pot: ready}), make_state(obj("soup", (2, 1), ("onion", 3, 20)), position=(2, 1), orientation=(0, -1), objects={pot: ready}), 0, "ready pot does not empty"),
            (make_state(obj("dish", (2, 1)), position=(2, 1), orientation=(0, -1), objects={pot: obj("soup", pot, ("onion", 3, 19))}), make_state(obj("soup", (2, 1), ("onion", 3, 19)), position=(2, 1), orientation=(0, -1)), 0, "soup is not ready"),
            (make_state(None, position=(2, 1), orientation=(0, -1), objects={pot: ready}), make_state(obj("soup", (2, 1), ("onion", 3, 20)), position=(2, 1), orientation=(0, -1)), 0, "agent did not hold dish before"),
            (make_state(obj("dish", (2, 1)), position=(2, 1), orientation=(0, -1), objects={pot: ready}), make_state(None, position=(2, 1), orientation=(0, -1)), 0, "agent does not hold soup after"),
            (make_state(obj("dish", (1, 1)), position=(1, 1), orientation=(0, 1), objects={pot: ready}), make_state(obj("soup", (1, 1), ("onion", 3, 20)), position=(1, 1), orientation=(0, 1)), 0, "agent is not facing emptied pot"),
        ]
    if event_type == "soup_delivered":
        return [
            (make_state(obj("soup", (3, 2), ("onion", 3, 20)), position=(3, 2), orientation=(0, 1)), make_state(None, position=(3, 2), orientation=(0, 1)), 0, "no positive reward evidence"),
            (make_state(obj("soup", (2, 2), ("onion", 3, 20)), position=(2, 2), orientation=(0, 1)), make_state(None, position=(2, 2), orientation=(0, 1)), 20, "agent is not facing serving"),
            (make_state(obj("soup", (3, 2), ("onion", 3, 19)), position=(3, 2), orientation=(0, 1)), make_state(None, position=(3, 2), orientation=(0, 1)), 20, "soup is not fully cooked"),
            (make_state(obj("soup", (3, 2), ("onion", 3, 20)), position=(3, 2), orientation=(0, 1), orders=["onion"]), make_state(None, position=(3, 2), orientation=(0, 1), orders=["onion"]), 20, "finite order list does not shrink"),
            (make_state(obj("soup", (3, 2), ("onion", 3, 20)), position=(3, 2), orientation=(0, 1), orders=["tomato"]), make_state(None, position=(3, 2), orientation=(0, 1), orders=[]), 20, "delivered soup does not match current order"),
        ]
    raise ValueError(event_type)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-positive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    mdp = OvercookedGridworld.from_layout_name("simple")
    detector = EventDetector(mdp)
    real_rows = [
        json.loads(line)
        for line in Path(args.real_positive).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    real_by_type = defaultdict(list)
    for row in real_rows:
        real_by_type[row["event"]["event_type"]].append(row)

    output_rows = []
    for event_type in EVENT_TYPES:
        positives = []
        for row in real_by_type[event_type]:
            copied = dict(row)
            copied["target_event_type"] = event_type
            copied["reason"] = "real environment state transition"
            copied["observed_events"] = [row["event"]]
            copied["audit_id"] = "{}-positive-{:02d}".format(event_type, len(positives))
            positives.append(copied)
        while len(positives) < 5:
            before, after, reward, reason = positive_transition(event_type, len(positives))
            events = detect(detector, before, after, len(positives), reward)
            if event_type not in [event["event_type"] for event in events]:
                raise AssertionError("constructed positive failed: {}".format(event_type))
            positives.append(
                audit_row("constructed_real_state_classes", "positive", event_type, reason, before, after, reward, events, len(positives))
            )
        output_rows.extend(positives[:5])

        negatives = []
        for index, (before, after, reward, reason) in enumerate(negative_transitions(event_type)):
            events = detect(detector, before, after, index, reward)
            if event_type in [event["event_type"] for event in events]:
                raise AssertionError("constructed negative emitted target: {} {}".format(event_type, reason))
            negatives.append(
                audit_row("constructed_real_state_classes", "negative", event_type, reason, before, after, reward, events, index)
            )
        if len(negatives) != 5:
            raise AssertionError("expected five negatives for {}".format(event_type))
        output_rows.extend(negatives)

    counts = Counter((row["target_event_type"], row["audit_label"]) for row in output_rows)
    expected_counts = {
        (event_type, label): 5
        for event_type in EVENT_TYPES
        for label in ("positive", "negative")
    }
    if counts != expected_counts:
        raise AssertionError("audit distribution mismatch: {}".format(counts))
    for event_type in EVENT_TYPES:
        real_count = sum(
            1
            for row in output_rows
            if row["target_event_type"] == event_type
            and row["audit_label"] == "positive"
            and row["source"] == "real_overcooked_multi_env"
        )
        if real_count < 1:
            raise AssertionError("{} lacks a real environment positive".format(event_type))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("event audit: PASS")
    print("rows={}".format(len(output_rows)))
    for event_type in EVENT_TYPES:
        print("{} positive=5 negative=5".format(event_type))


if __name__ == "__main__":
    main()
