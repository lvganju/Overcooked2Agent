"""validate_interface.py — D4 训练接口一致性校验（数据组）。

对 smoke_dataset/{train,val,test}.jsonl 逐行校验是否符合 training_interface_v0.2.md
冻结的字段规范。任何一条不合规即判 FAIL 并打印具体原因，不做"大概率通过就算了"的
近似检查。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_features import FEATURE_DIM

EXPECTED_HISTORY_WINDOW = 5
VALID_TARGETS = {-1, 0, 1, 2, 3, 4, 5}
REQUIRED_FIELDS = {
    "episode_id", "timestep", "layout_id", "history_mask", "features",
    "intent_target", "classification_mask", "label_name", "subject_id", "seed",
}


def validate_file(path: Path) -> int:
    n_rows = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_rows += 1

            missing = REQUIRED_FIELDS - row.keys()
            assert not missing, f"{path}:{line_no} missing fields {missing}"

            features = row["features"]
            mask = row["history_mask"]
            assert len(features) == EXPECTED_HISTORY_WINDOW, (
                f"{path}:{line_no} features window length {len(features)} != {EXPECTED_HISTORY_WINDOW}"
            )
            assert len(mask) == EXPECTED_HISTORY_WINDOW, (
                f"{path}:{line_no} history_mask length {len(mask)} != {EXPECTED_HISTORY_WINDOW}"
            )
            for step_vec in features:
                assert len(step_vec) == FEATURE_DIM, (
                    f"{path}:{line_no} feature vector dim {len(step_vec)} != {FEATURE_DIM}"
                )
                arr = np.asarray(step_vec, dtype=np.float32)
                assert np.isfinite(arr).all(), f"{path}:{line_no} NaN/Inf detected in features"

            intent_target = row["intent_target"]
            assert intent_target in VALID_TARGETS, f"{path}:{line_no} invalid intent_target={intent_target}"

            classification_mask = row["classification_mask"]
            if intent_target == -1:
                assert classification_mask is False, (
                    f"{path}:{line_no} intent_target=-1 but classification_mask={classification_mask}"
                )
            else:
                assert classification_mask is True, (
                    f"{path}:{line_no} intent_target={intent_target} but classification_mask={classification_mask}"
                )

            # padding 一致性：history_mask=False 的位置必须是全 0 特征向量。
            for i, valid in enumerate(mask):
                if not valid:
                    assert all(v == 0.0 for v in features[i]), (
                        f"{path}:{line_no} history_mask[{i}]=False but features[{i}] not all-zero"
                    )
    return n_rows


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
        print(f"{split_name}: {n} rows PASS")

    print(f"validate_interface: PASS ({total} rows total, feature_dim={FEATURE_DIM}, "
          f"history_window={EXPECTED_HISTORY_WINDOW})")


if __name__ == "__main__":
    main()
