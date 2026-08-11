"""build_stats.py — 生成 stats.json（D5 交接物之一）。

在 quality_report.json（build_dataset.py 产出）基础上，补充一些 Agent 组
训练前常用的统计量：类别不平衡比例、每个 subject_id 的标签分布、序列长度
（history_window 固定与否）、classification_mask=True 的行占比等。

不重新计算特征，只是对 quality_report.json 和最终 jsonl 做一次轻量二次汇总，
避免与 quality_report.json 重复维护同一份统计逻辑。

用法：
    python build_stats.py --dataset-dir cole_task_intent_v0.2/smoke \
        --quality-report cole_task_intent_v0.2/smoke/quality_report.json \
        --output cole_task_intent_v0.2/smoke/stats.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_rows(dataset_dir: Path, splits):
    rows = []
    for split in splits:
        p = dataset_dir / f"{split}.jsonl"
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    r["_split"] = split
                    rows.append(r)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    quality = json.loads(args.quality_report.read_text(encoding="utf-8"))
    rows = load_rows(args.dataset_dir, args.splits)

    total_rows = len(rows)
    label_counts_overall = Counter(r["label_name"] for r in rows)
    classification_rows = [r for r in rows if r.get("classification_mask")]
    n_classification = len(classification_rows)

    # 类别不平衡比例：max_count / min_count（仅统计参与分类的类别，即不含 unknown/invalid）
    cls_label_counts = Counter(r["label_name"] for r in classification_rows)
    if cls_label_counts:
        max_c = max(cls_label_counts.values())
        min_c = min(cls_label_counts.values())
        imbalance_ratio = round(max_c / min_c, 3) if min_c > 0 else None
    else:
        imbalance_ratio = None

    # 每个 subject_id 的标签分布
    by_subject: dict = defaultdict(Counter)
    for r in rows:
        by_subject[r["subject_id"]][r["label_name"]] += 1

    # history_mask 覆盖率：平均每行有效历史步数占 history_window 的比例
    history_window = quality.get("history_window")
    if history_window and rows and "history_mask" in rows[0]:
        total_valid = sum(sum(1 for m in r["history_mask"] if m) for r in rows)
        avg_history_coverage = round(total_valid / (total_rows * history_window), 4)
    else:
        avg_history_coverage = None

    stats = {
        "schema_version": quality.get("schema_version"),
        "total_rows": total_rows,
        "num_classification_rows": n_classification,
        "classification_row_ratio": round(n_classification / total_rows, 4) if total_rows else None,
        "label_counts_overall": dict(label_counts_overall),
        "label_counts_classification_only": dict(cls_label_counts),
        "class_imbalance_ratio_max_over_min": imbalance_ratio,
        "label_counts_by_subject_id": {k: dict(v) for k, v in by_subject.items()},
        "avg_history_window_coverage": avg_history_coverage,
        "splits_summary": quality.get("splits"),
    }

    args.output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote stats.json to {args.output} (total_rows={total_rows}, "
          f"classification_rows={n_classification}, imbalance_ratio={imbalance_ratio})")


if __name__ == "__main__":
    main()
