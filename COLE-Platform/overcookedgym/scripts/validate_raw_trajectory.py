"""Validate trajectory_schema_v2 JSON/JSONL files without loading a model."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from overcookedgym.environment_interfaces.trajectory_schema_v2 import validate_trajectory


def load_documents(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return [json.loads(text)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    args = parser.parse_args()

    documents = load_documents(args.trajectory)
    if not documents:
        raise ValueError("trajectory file contains no documents")
    episodes = 0
    steps = 0
    events = 0
    for index, trajectory in enumerate(documents):
        try:
            validate_trajectory(trajectory)
        except Exception as exc:
            raise ValueError("document {} failed validation: {}".format(index, exc))
        episodes += len(trajectory["step_records"])
        steps += sum(len(episode) for episode in trajectory["step_records"])
        events += sum(
            len(step_events)
            for episode in trajectory["ep_events"]
            for step_events in episode
        )
    print("raw trajectory validation: PASS")
    print("documents={} episodes={} steps={} events={}".format(
        len(documents), episodes, steps, events
    ))
    print("No COLE model was loaded and no training was run.")


if __name__ == "__main__":
    main()
