import pathlib
import sys
import unittest


OVERCOOKEDGYM_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(OVERCOOKEDGYM_DIR) not in sys.path:
    sys.path.insert(0, str(OVERCOOKEDGYM_DIR))

from environment_interfaces.policy_adapter import (
    ConstantPolicy,
    PolicyAdapter,
    ScriptedPolicy,
    ScriptExhaustedError,
    normalize_action,
)


class PredictPolicy(object):
    def __init__(self):
        self.deterministic = None

    def predict(self, observation, deterministic=False):
        self.deterministic = deterministic
        return 2, "unused-state"


class PolicyAdapterTests(unittest.TestCase):
    def test_all_original_action_ids_are_accepted(self):
        self.assertEqual(list(range(6)), [normalize_action(i) for i in range(6)])

    def test_invalid_actions_are_rejected(self):
        for value in (-1, 6, 1.5, True, "4"):
            with self.assertRaises((TypeError, ValueError)):
                normalize_action(value)

    def test_predict_tuple_is_supported_and_deterministic(self):
        policy = PredictPolicy()
        self.assertEqual(2, PolicyAdapter(policy).act("observation"))
        self.assertTrue(policy.deterministic)

    def test_callable_policy_is_supported(self):
        self.assertEqual(5, PolicyAdapter(lambda observation: 5).act(None))

    def test_scripted_policy_exhaustion_is_explicit_and_resettable(self):
        policy = ScriptedPolicy([0, 5])
        self.assertEqual(0, policy.act())
        self.assertEqual(5, policy.act())
        with self.assertRaises(ScriptExhaustedError):
            policy.act()
        policy.reset()
        self.assertEqual(0, policy.act())

    def test_constant_policy_does_not_exhaust(self):
        policy = ConstantPolicy(4)
        self.assertEqual([4, 4, 4], [policy.act(), policy.act(), policy.act()])


if __name__ == "__main__":
    unittest.main()
