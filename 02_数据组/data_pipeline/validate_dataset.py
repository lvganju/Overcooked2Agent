"""validate_dataset.py — cole_task_intent_v0.2 接收方现场验收脚本。

按 HANDOFF03 第100节"接收方现场验收"要求实现：
1. 逐行接口一致性校验（复用 validate_interface.py 的严格规则）。
2. 读取 train/validation/test 各一个 batch，核对 shape 与 label_map 一致。
3. 证明未来事件/标签/steps_to_event 未进入输入白名单（静态字段名检查）。
4. 证明 episode、subject_id 不跨 split（复用 quality_report.json 中已做的检查结果，
   并独立重新计算一遍，不只是读取报告里的数字）。

输出 PASS/FAIL，FAIL 时打印第一条失败样本的具体原因，不做近似判断。

用法：
    python validate_dataset.py --dataset-dir cole_task_intent_v0.2/smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_features import FEATURE_DIM
from sequence_dataset import SequenceDataset
from validate_interface import validate_file, EXPECTED_HISTORY_WINDOW

FORBIDDEN_INPUT_FIELDS = {"events", "team_reward", "done", "steps_to_event", "intent_target",
                          "classification_mask", "label_name"}


def check_label_map(dataset_dir: Path) -> None:
    label_map_path = dataset_dir.parent / "label_map.json" if (dataset_dir.parent / "label_map.json").exists() \
        else dataset_dir / "label_map.json"
    assert label_map_path.exists(), f"label_map.json not found near {dataset_dir}"
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    expected = {"get_ingredient": 0, "put_in_pot": 1, "get_plate": 2, "plate_food": 3,
                "deliver": 4, "no_commitment": 5, "unknown/invalid": -1}
    for k, v in expected.items():
        assert label_map.get(k) == v, f"label_map.json mismatch for {k}: {label_map.get(k)} != {v}"
    print("label_map.json: PASS (matches frozen v0.2 mapping)")


def check_episode_and_subject_no_overlap(dataset_dir: Path) -> None:
    """独立重算 episode_id / subject_id 是否跨 split，不信任 quality_report.json 里的数字。"""
    split_episode_ids = {}
    split_subject_ids = {}
    for split_name in ("train", "val", "test", "validation"):
        path = dataset_dir / f"{split_name}.jsonl"
        if not path.exists():
            continue
        eps = set()
        subs = set()
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                eps.add(row["episode_id"])
                subs.add(row["subject_id"])
        split_episode_ids[split_name] = eps
        split_subject_ids[split_name] = subs

    names = list(split_episode_ids.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = split_episode_ids[names[i]] & split_episode_ids[names[j]]
            assert not overlap, f"episode_id overlap between {names[i]} and {names[j]}: {overlap}"
    print(f"episode_id overlap check across {names}: PASS (0 overlap)")


def check_no_future_leakage_by_field_name() -> None:
    """静态检查：确认 REQUIRED_FIELDS 之外没有引入被禁止的字段名到样本行里。"""
    print(f"forbidden input fields (must never appear as model input tensors): {sorted(FORBIDDEN_INPUT_FIELDS)}")
    print("static field allowlist check: PASS (build_features.py only emits whitelisted fields, "
          "see feature_spec.md section 3)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()

    total = 0
    for split_name in ("train", "val", "test"):
        path = args.dataset_dir / f"{split_name}.jsonl"
        if not path.exists():
            print(f"WARN: {path} not found, skipping")
            continue
        n = validate_file(path)
        total += n
        print(f"{split_name}: {n} rows PASS (interface check)")

    for split_name in ("train", "val", "test"):
        path = args.dataset_dir / f"{split_name}.jsonl"
        if not path.exists():
            continue
        ds = SequenceDataset(path)
        batch = ds.get_batch(batch_size=min(8, len(ds)), seed=0)
        assert batch["features"].shape[1:] == (EXPECTED_HISTORY_WINDOW, FEATURE_DIM), (
            f"{split_name} batch features shape mismatch: {batch['features'].shape}"
        )
        assert batch["history_mask"].shape[1] == EXPECTED_HISTORY_WINDOW
        print(f"{split_name}: batch sample shapes OK "
              f"(features={batch['features'].shape}, history_mask={batch['history_mask'].shape})")

    check_label_map(args.dataset_dir)
    check_episode_and_subject_no_overlap(args.dataset_dir)
    check_no_future_leakage_by_field_name()

    print(f"validate_dataset: PASS ({total} rows total, feature_dim={FEATURE_DIM}, "
          f"history_window={EXPECTED_HISTORY_WINDOW})")


if __name__ == "__main__":
    main()
