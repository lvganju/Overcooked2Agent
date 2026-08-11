"""Run the environment-group acceptance checks without models or training."""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVENT_TYPES = (
    "ingredient_acquired",
    "ingredient_put_in_pot",
    "plate_acquired",
    "soup_plated",
    "soup_delivered",
)


def run_check(name, command):
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    result = {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "output": completed.stdout,
        "passed": completed.returncode == 0,
    }
    if not result["passed"]:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def validate_event_audit(path):
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = {
        "audit_id",
        "source",
        "audit_label",
        "target_event_type",
        "reason",
        "state_before",
        "state_after",
        "team_reward",
        "observed_events",
    }
    if len(rows) != 50:
        raise ValueError("event audit must contain exactly 50 rows")
    if any(not required.issubset(row) for row in rows):
        raise ValueError("event audit row is missing required fields")
    counts = Counter((row["target_event_type"], row["audit_label"]) for row in rows)
    real_positive_counts = Counter(
        row["target_event_type"]
        for row in rows
        if row["audit_label"] == "positive"
        and row["source"] == "real_overcooked_multi_env"
    )
    for event_type in EVENT_TYPES:
        if counts[(event_type, "positive")] != 5:
            raise ValueError("{} must have five positives".format(event_type))
        if counts[(event_type, "negative")] != 5:
            raise ValueError("{} must have five negatives".format(event_type))
        if real_positive_counts[event_type] < 1:
            raise ValueError("{} lacks a real environment positive".format(event_type))
    for row in rows:
        observed_types = {event["event_type"] for event in row["observed_events"]}
        target = row["target_event_type"]
        if row["audit_label"] == "positive" and target not in observed_types:
            raise ValueError("positive {} does not contain target event".format(row["audit_id"]))
        if row["audit_label"] == "negative" and target in observed_types:
            raise ValueError("negative {} contains target event".format(row["audit_id"]))
    return {
        "name": "event_audit",
        "passed": True,
        "rows": len(rows),
        "distribution": {
            event_type: {
                "positive": counts[(event_type, "positive")],
                "negative": counts[(event_type, "negative")],
                "real_environment_positive": real_positive_counts[event_type],
            }
            for event_type in EVENT_TYPES
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--handoff-dir",
        type=Path,
        default=REPO_ROOT / "platform_data_handoff_v2",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    handoff_dir = args.handoff_dir.resolve()
    report_path = args.report or handoff_dir / "logs" / "environment_handoff_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    scripts = REPO_ROOT / "overcookedgym" / "scripts"
    checks = []
    try:
        checks.append(run_check(
            "action_mapping_and_unit_tests",
            [python, "-B", str(scripts / "verify_trajectory_schema_v2.py")],
        ))
        checks.append(run_check(
            "raw_trajectory",
            [python, "-B", str(scripts / "validate_raw_trajectory.py"),
             str(handoff_dir / "sample_event_episode.jsonl")],
        ))
        checks.append(run_check(
            "episode_replay",
            [python, "-B", str(scripts / "replay_episode.py"),
             str(handoff_dir / "sample_event_episode.jsonl")],
        ))
        checks.append(validate_event_audit(handoff_dir / "event_audit.jsonl"))
        report = {
            "status": "PASS",
            "stage": "E4",
            "checks": checks,
            "models_loaded": False,
            "training_run": False,
        }
    except Exception as exc:
        report = {
            "status": "FAIL",
            "stage": "E4",
            "checks": checks,
            "error": str(exc),
            "models_loaded": False,
            "training_run": False,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("environment handoff validation: FAIL")
        print(str(exc))
        print("report={}".format(report_path))
        return 1

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("environment handoff validation: PASS")
    print("checks={}".format(len(checks)))
    print("event_audit_rows=50")
    print("report={}".format(report_path))
    print("No COLE model was loaded and no training was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
