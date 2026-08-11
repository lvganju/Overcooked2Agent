"""State-delta task event detection for Overcooked.

No event in this module is inferred from an action. Detection uses player-held
objects, world objects, orders, and MDP terrain before and after a transition.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Position = Tuple[int, int]


def _position(value: Any) -> Position:
    return tuple(value)


def _players(state: Any) -> Sequence[Any]:
    return state.players


def _held(player: Any) -> Optional[Any]:
    return getattr(player, "held_object", None)


def _objects(state: Any) -> Dict[Position, Any]:
    objects = state.objects
    if isinstance(objects, dict):
        return {_position(pos): obj for pos, obj in objects.items()}
    return {_position(obj.position): obj for obj in objects}


def _soup_data(obj: Any) -> Tuple[str, int, int]:
    soup_type, num_items, cook_time = obj.state
    return str(soup_type), int(num_items), int(cook_time)


class EventDetector(object):
    """Detect the five environment-team task events from two real states."""

    EVENT_ORDER = {
        "ingredient_acquired": 0,
        "ingredient_put_in_pot": 1,
        "plate_acquired": 2,
        "soup_plated": 3,
        "soup_delivered": 4,
    }

    def __init__(self, mdp: Any):
        required = (
            "get_terrain_type_at_pos",
            "get_pot_locations",
            "get_serving_locations",
            "num_items_for_soup",
            "soup_cooking_time",
        )
        missing = [name for name in required if not hasattr(mdp, name)]
        if missing:
            raise TypeError("mdp missing event-detection attributes: {}".format(missing))
        self.mdp = mdp

    def detect(
        self,
        state_before: Any,
        state_after: Any,
        timestep: int,
        team_reward: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return events proven by the before/after state delta.

        The method intentionally accepts no action argument, preventing action-
        based event guesses by construction.
        """
        before_players = _players(state_before)
        after_players = _players(state_after)
        if len(before_players) != len(after_players):
            raise ValueError("player count changed across transition")

        events = []
        for agent_index, (before_player, after_player) in enumerate(
            zip(before_players, after_players)
        ):
            events.extend(
                self._detect_for_player(
                    agent_index,
                    before_player,
                    after_player,
                    state_before,
                    state_after,
                    timestep,
                    team_reward,
                )
            )
        return sorted(
            events,
            key=lambda event: (
                event["agent_index"],
                self.EVENT_ORDER[event["event_type"]],
            ),
        )

    def _detect_for_player(
        self,
        agent_index: int,
        before_player: Any,
        after_player: Any,
        state_before: Any,
        state_after: Any,
        timestep: int,
        team_reward: Optional[float],
    ) -> List[Dict[str, Any]]:
        before_held = _held(before_player)
        after_held = _held(after_player)
        events = []

        if before_held is None and after_held is not None:
            if after_held.name in ("onion", "tomato"):
                details = {"ingredient": after_held.name}
                source = self._acquisition_source(before_player, state_before, after_held.name)
                if source is not None:
                    details["source"] = source
                events.append(self._event("ingredient_acquired", timestep, agent_index, details))
            elif after_held.name == "dish":
                details = {}
                source = self._acquisition_source(before_player, state_before, "dish")
                if source is not None:
                    details["source"] = source
                events.append(self._event("plate_acquired", timestep, agent_index, details))

        pot_pos = self._facing_feature(before_player)
        if pot_pos in set(map(_position, self.mdp.get_pot_locations())):
            before_pot = _objects(state_before).get(pot_pos)
            after_pot = _objects(state_after).get(pot_pos)
            put_details = self._ingredient_put_details(before_held, after_held, before_pot, after_pot, pot_pos)
            if put_details is not None:
                events.append(self._event("ingredient_put_in_pot", timestep, agent_index, put_details))
            plated_details = self._soup_plated_details(before_held, after_held, before_pot, after_pot, pot_pos)
            if plated_details is not None:
                events.append(self._event("soup_plated", timestep, agent_index, plated_details))

        delivered_details = self._soup_delivered_details(
            before_player,
            before_held,
            after_held,
            state_before,
            state_after,
            team_reward,
        )
        if delivered_details is not None:
            events.append(self._event("soup_delivered", timestep, agent_index, delivered_details))
        return events

    def _facing_feature(self, player: Any) -> Position:
        x, y = _position(player.position)
        dx, dy = _position(player.orientation)
        return x + dx, y + dy

    def _acquisition_source(self, player: Any, state_before: Any, object_name: str) -> Optional[str]:
        feature_pos = self._facing_feature(player)
        obj = _objects(state_before).get(feature_pos)
        if obj is not None and obj.name == object_name:
            return "counter"
        terrain = self.mdp.get_terrain_type_at_pos(feature_pos)
        dispenser = {"onion": "O", "tomato": "T", "dish": "D"}.get(object_name)
        if terrain == dispenser:
            return "dispenser"
        return None

    def _ingredient_put_details(self, before_held, after_held, before_pot, after_pot, pot_pos):
        if before_held is None or before_held.name not in ("onion", "tomato"):
            return None
        if after_held is not None:
            return None
        if after_pot is None or after_pot.name != "soup":
            return None
        after_type, after_count, _ = _soup_data(after_pot)
        if after_type != before_held.name:
            return None
        if before_pot is None:
            before_count = 0
        elif before_pot.name == "soup":
            before_type, before_count, _ = _soup_data(before_pot)
            if before_type != after_type:
                return None
        else:
            return None
        if after_count != before_count + 1:
            return None
        return {
            "ingredient": after_type,
            "pot_position": list(pot_pos),
            "count_before": before_count,
            "count_after": after_count,
        }

    def _soup_plated_details(self, before_held, after_held, before_pot, after_pot, pot_pos):
        if before_held is None or before_held.name != "dish":
            return None
        if after_held is None or after_held.name != "soup":
            return None
        if before_pot is None or before_pot.name != "soup" or after_pot is not None:
            return None
        before_data = _soup_data(before_pot)
        after_data = _soup_data(after_held)
        if before_data != after_data:
            return None
        soup_type, num_items, cook_time = before_data
        if num_items != self.mdp.num_items_for_soup or cook_time < self.mdp.soup_cooking_time:
            return None
        return {
            "soup_type": soup_type,
            "pot_position": list(pot_pos),
            "num_items": num_items,
            "cook_time": cook_time,
        }

    def _soup_delivered_details(
        self,
        before_player,
        before_held,
        after_held,
        state_before,
        state_after,
        team_reward,
    ):
        if before_held is None or before_held.name != "soup" or after_held is not None:
            return None
        soup_type, num_items, cook_time = _soup_data(before_held)
        if num_items != self.mdp.num_items_for_soup or cook_time < self.mdp.soup_cooking_time:
            return None
        serving_pos = self._facing_feature(before_player)
        if serving_pos not in set(map(_position, self.mdp.get_serving_locations())):
            return None
        if team_reward is None or isinstance(team_reward, bool) or team_reward <= 0:
            # State alone cannot distinguish a rewarded delivery in layouts
            # with an unlimited (None) order list. Reward evidence is required
            # for one consistent rule across finite and unlimited orders.
            return None

        before_orders = getattr(state_before, "order_list", None)
        after_orders = getattr(state_after, "order_list", None)
        if before_orders is None:
            if after_orders is not None:
                return None
        else:
            if not before_orders:
                return None
            current_order = before_orders[0]
            if current_order not in ("any", soup_type):
                return None
            if list(after_orders) != list(before_orders[1:]):
                return None
        return {
            "soup_type": soup_type,
            "serving_position": list(serving_pos),
            "orders_before": None if before_orders is None else list(before_orders),
            "orders_after": None if after_orders is None else list(after_orders),
            "team_reward": team_reward,
        }

    @staticmethod
    def _event(event_type: str, timestep: int, agent_index: int, details: Dict[str, Any]):
        return {
            "event_type": event_type,
            "timestep": timestep,
            "agent_index": agent_index,
            "details": details,
        }
