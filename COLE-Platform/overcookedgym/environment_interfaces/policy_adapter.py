"""Small, model-agnostic policy adapters for environment data collection."""

import numbers

from .trajectory_schema_v2 import validate_action


class ScriptExhaustedError(RuntimeError):
    """Raised when a finite scripted policy has no action for the next step."""


def normalize_action(value):
    """Normalize a scalar or common ``predict`` result to an action id in 0..5."""
    if isinstance(value, tuple):
        if not value:
            raise TypeError("policy returned an empty tuple")
        value = value[0]
    if hasattr(value, "shape") and hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            raise TypeError("policy action array must contain exactly one value")
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError("policy action must resolve to one integer")
    action = int(value)
    validate_action(action, "policy action")
    return action


class PolicyAdapter(object):
    """Expose a uniform ``act(observation) -> int`` interface.

    The wrapped object may implement Stable-Baselines-style ``predict`` or be
    directly callable. Loading checkpoints remains the caller's responsibility.
    """

    def __init__(self, policy, deterministic=True):
        if policy is None:
            raise TypeError("policy is required")
        self.policy = policy
        self.deterministic = deterministic

    def reset(self):
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def act(self, observation):
        predict = getattr(self.policy, "predict", None)
        if callable(predict):
            try:
                result = predict(observation, deterministic=self.deterministic)
            except TypeError:
                result = predict(observation)
        elif callable(self.policy):
            result = self.policy(observation)
        else:
            raise TypeError("policy must be callable or expose predict()")
        return normalize_action(result)


class ScriptedPolicy(object):
    """Finite programmatic action sequence for reproducible environment runs."""

    def __init__(self, actions):
        self.actions = tuple(normalize_action(action) for action in actions)
        if not self.actions:
            raise ValueError("scripted policy requires at least one action")
        self.reset()

    def reset(self):
        self._index = 0

    def act(self, observation=None):
        del observation
        if self._index >= len(self.actions):
            raise ScriptExhaustedError(
                "scripted policy exhausted after {} actions".format(len(self.actions))
            )
        action = self.actions[self._index]
        self._index += 1
        return action


class ConstantPolicy(ScriptedPolicy):
    """Non-exhausting fixed action policy, useful for deterministic baselines."""

    def __init__(self, action):
        self.action = normalize_action(action)

    def reset(self):
        pass

    def act(self, observation=None):
        del observation
        return self.action
