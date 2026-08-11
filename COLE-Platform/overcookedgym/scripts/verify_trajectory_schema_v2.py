"""Run environment-interface verification without loading models or training."""

import importlib.util
import pathlib
import sys
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def verify_original_action_mapping():
    actions_path = (
        REPO_ROOT
        / "overcookedgym"
        / "human_aware_rl"
        / "overcooked_ai"
        / "overcooked_ai_py"
        / "mdp"
        / "actions.py"
    )
    spec = importlib.util.spec_from_file_location("cole_original_actions", str(actions_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = [(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0), "interact"]
    actual = list(module.Action.INDEX_TO_ACTION)
    if actual != expected:
        raise AssertionError("original action mapping changed: {!r}".format(actual))
    return actual


def main():
    mapping = verify_original_action_mapping()
    tests_dir = REPO_ROOT / "overcookedgym" / "tests"
    suite = unittest.defaultTestLoader.discover(str(tests_dir), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("Verified original action mapping 0..5: {}".format(mapping))
    print("No COLE model was loaded and no training was run.")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
