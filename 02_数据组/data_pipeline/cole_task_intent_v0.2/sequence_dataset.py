"""sequence_dataset.py — cole_task_intent_v0.2 数据加载器（交付给 Agent 组）。

按 training_interface_v0.2.md 冻结的字段读取 jsonl/parquet 格式的数据集切片，
产出 numpy 数组供训练框架消费。不依赖 TensorFlow/PyTorch，保持框架无关。

用法：
    from sequence_dataset import SequenceDataset
    ds = SequenceDataset("cole_task_intent_v0.2/smoke/train.parquet")
    batch = ds.get_batch(batch_size=32, seed=0)
    batch["features"].shape       # (32, W, F)
    batch["history_mask"].shape   # (32, W)
    batch["intent_target"].shape  # (32,)
    batch["classification_mask"].shape  # (32,)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

REQUIRED_COLUMNS = (
    "episode_id", "timestep", "layout_id", "history_mask", "features",
    "intent_target", "classification_mask", "label_name", "subject_id", "seed",
)


class SequenceDataset:
    """加载 cole_task_intent_v0.2 的一个 split（jsonl 或 parquet），提供批采样接口。"""

    def __init__(self, path):
        self.path = Path(path)
        if self.path.suffix == ".parquet":
            import pandas as pd  # 延迟导入，避免无 parquet 需求时也强制安装 pyarrow
            df = pd.read_parquet(self.path)
            self._rows = df.to_dict("records")
            # parquet 往返后，嵌套 list 字段（如 features/history_mask）会变成
            # numpy.ndarray（dtype=object）而不是原生 list，np.asarray 对
            # “ndarray 的 ndarray”不会自动递归堆叠成规则的多维数组
            # （会报 "setting an array element with a sequence"）。
            # 这里统一转回原生嵌套 list，行为与 jsonl 加载路径保持一致。
            for r in self._rows:
                for key in ("features", "history_mask"):
                    if key in r and isinstance(r[key], np.ndarray):
                        r[key] = r[key].tolist()
        elif self.path.suffix in (".jsonl", ".json"):
            self._rows = []
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        self._rows.append(json.loads(line))
        else:
            raise ValueError(f"unsupported file extension: {self.path.suffix}")

        missing = set(REQUIRED_COLUMNS) - set(self._rows[0].keys()) if self._rows else set()
        if missing:
            raise ValueError(f"dataset missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self._rows)

    def num_classification_rows(self) -> int:
        """只统计 classification_mask=True 的行数（unknown/invalid 不计入）。"""
        return sum(1 for r in self._rows if r["classification_mask"])

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """把整个 split 一次性转成 numpy 数组（小到中等规模数据集适用）。"""
        features = np.asarray([r["features"] for r in self._rows], dtype=np.float32)
        history_mask = np.asarray([r["history_mask"] for r in self._rows], dtype=bool)
        intent_target = np.asarray([r["intent_target"] for r in self._rows], dtype=np.int64)
        classification_mask = np.asarray([r["classification_mask"] for r in self._rows], dtype=bool)
        episode_id = np.asarray([r["episode_id"] for r in self._rows], dtype=object)
        subject_id = np.asarray([r["subject_id"] for r in self._rows], dtype=object)
        return {
            "features": features,
            "history_mask": history_mask,
            "intent_target": intent_target,
            "classification_mask": classification_mask,
            "episode_id": episode_id,
            "subject_id": subject_id,
        }

    def get_batch(self, batch_size: int, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """随机采样一个 batch（用于快速抽样检查；训练循环建议用 to_arrays() 自行做 DataLoader）。"""
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(self._rows), size=min(batch_size, len(self._rows)), replace=False)
        arrays = self.to_arrays()
        return {k: v[idx] for k, v in arrays.items()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    ds = SequenceDataset(args.path)
    print(f"loaded {len(ds)} rows from {args.path}")
    print(f"classification rows (mask=True): {ds.num_classification_rows()}")
    batch = ds.get_batch(args.batch_size, seed=0)
    for k, v in batch.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")
