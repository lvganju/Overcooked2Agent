"""Non-invasive trajectory recorder for ``OvercookedMultiEnv``."""

import copy
import uuid
from typing import Any, Dict, Optional

from .event_detector import EventDetector
from .trajectory_schema_v2 import (
    SCHEMA_VERSION,
    detached,
    state_to_dict,
    validate_action,
    validate_trajectory,
)


class TrajectoryRecorderV2(object):
    """Wrap an existing OvercookedMultiEnv without changing its behavior."""

    def __init__(self, env: Any, layout_id: str, seed: int, human_index: int):
        required = ("multi_reset", "multi_step", "base_env", "ego_agent_idx", "mdp")
        missing = [name for name in required if not hasattr(env, name)]
        if missing:
            raise TypeError("env is not OvercookedMultiEnv-compatible; missing {}".format(missing))
        if human_index not in (0, 1):
            raise ValueError("human_index must be 0 or 1")
        if not isinstance(layout_id, str) or not layout_id:
            raise ValueError("layout_id must be a non-empty string")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self.env = env
        self.layout_id = layout_id
        self.seed = seed
        self.human_index = human_index
        self.event_detector = EventDetector(env.mdp)
        self.trajectory = self._empty_trajectory()
        self._episode_index = None
        self._episode_id = None
        self._timestep = 0

    @staticmethod
    def _empty_trajectory() -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "ep_states": [],
            "ep_actions": [],
            "ep_rewards": [],
            "ep_dones": [],
            "ep_events": [],
            "ep_final_states": [],
            "step_records": [],
        }

    def reset(self, episode_id: Optional[str] = None):
        """Call the wrapped reset and begin a new recorded episode."""
        observation = self.env.multi_reset()
        self._episode_id = episode_id or str(uuid.uuid4())
        if not isinstance(self._episode_id, str) or not self._episode_id:
            raise ValueError("episode_id must be a non-empty string")
        for key in ("ep_states", "ep_actions", "ep_rewards", "ep_dones", "ep_events", "step_records"):
            self.trajectory[key].append([])
        self.trajectory["ep_final_states"].append(state_to_dict(self.env.base_env.state))
        self._episode_index = len(self.trajectory["step_records"]) - 1
        self._timestep = 0
        return observation

    def step(self, human_action: int, cole_action: int, final_ai_action: Optional[int] = None):
        """Execute one real environment step and record its state delta."""
        if self._episode_index is None:
            raise RuntimeError("reset() must be called before step()")
        validate_action(human_action, "human_action")
        validate_action(cole_action, "cole_action")
        if final_ai_action is None:
            final_ai_action = cole_action
        validate_action(final_ai_action, "final_ai_action")

        indexed_actions = [None, None]
        indexed_actions[self.human_index] = human_action
        indexed_actions[1 - self.human_index] = final_ai_action
        if self.env.ego_agent_idx == 0:
            ego_action, alt_action = indexed_actions
        elif self.env.ego_agent_idx == 1:
            alt_action, ego_action = indexed_actions
        else:
            raise ValueError("wrapped env ego_agent_idx must be 0 or 1")

        state_before = copy.deepcopy(self.env.base_env.state)
        observation, rewards, done, info = self.env.multi_step(ego_action, alt_action)
        state_after = copy.deepcopy(self.env.base_env.state)
        team_reward = self._team_reward(rewards)
        done = bool(done)
        events = self.event_detector.detect(
            state_before, state_after, self._timestep, team_reward=team_reward
        )
        state_dict = state_to_dict(state_before)
        joint_action = [int(indexed_actions[0]), int(indexed_actions[1])]

        record = {
            "episode_id": self._episode_id,
            "timestep": self._timestep,
            "layout_id": self.layout_id,
            "seed": self.seed,
            "human_index": self.human_index,
            "state": state_dict,
            "human_action": int(human_action),
            "cole_action": int(cole_action),
            "final_ai_action": int(final_ai_action),
            "team_reward": team_reward,
            "done": done,
            "events": events,
        }
        idx = self._episode_index
        self.trajectory["ep_states"][idx].append(state_dict)
        self.trajectory["ep_actions"][idx].append(joint_action)
        self.trajectory["ep_rewards"][idx].append(team_reward)
        self.trajectory["ep_dones"][idx].append(done)
        self.trajectory["ep_events"][idx].append(events)
        self.trajectory["ep_final_states"][idx] = state_to_dict(state_after)
        self.trajectory["step_records"][idx].append(record)
        self._timestep += 1
        return observation, rewards, done, info

    @staticmethod
    def _team_reward(rewards: Any):
        if isinstance(rewards, (tuple, list)):
            if len(rewards) != 2 or rewards[0] != rewards[1]:
                raise ValueError("shared team reward requires two equal agent rewards")
            return rewards[0]
        return rewards

    def to_dict(self, validate: bool = True) -> Dict[str, Any]:
        result = detached(self.trajectory)
        if validate:
            validate_trajectory(result)
        return result

    def clear(self) -> None:
        self.trajectory = self._empty_trajectory()
        self._episode_index = None
        self._episode_id = None
        self._timestep = 0

    def __getattr__(self, name: str) -> Any:
        """Delegate non-recorder attributes to the wrapped environment."""
        return getattr(self.env, name)
