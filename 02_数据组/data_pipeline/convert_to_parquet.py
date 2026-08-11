"""convert_to_parquet.py — 将 jsonl 数据集 split 转成 parquet（D5 交接物之一）。

training_interface_v0.2.md 允许 jsonl 或 parquet 两种物理格式，parquet 便于
Agent 组用 pandas/pyarrow 大批量读取。转换只做格式搬运，不改变任何字段值。

用法（cole-platform 环境，需已 pip install pandas pyarrow）：
    python convert_to_parquet.py --dataset-dir smoke_dataset
    # 对 smoke_dataset/{train,val,test}.jsonl 就地生成同名 .parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def convert_one(jsonl_path: Path) -> Path:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    out_path = jsonl_path.with_suffix(".parquet")
    df.to_parquet(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        help="要转换的 split 文件名（不含扩展名），默认 train/val/test",
    )
    args = parser.parse_args()

    for split in args.splits:
        jsonl_path = args.dataset_dir / f"{split}.jsonl"
        if not jsonl_path.exists():
            print(f"skip {jsonl_path}: not found")
            continue
        out_path = convert_one(jsonl_path)
        n_rows_jsonl = sum(1 for _ in open(jsonl_path, "r", encoding="utf-8"))
        n_rows_parquet = len(pd.read_parquet(out_path))
        assert n_rows_jsonl == n_rows_parquet, "row count mismatch after parquet conversion"
        print(f"{jsonl_path} -> {out_path} ({n_rows_parquet} rows, verified)")


if __name__ == "__main__":
    main()
