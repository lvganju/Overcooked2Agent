"""labeler.py — D1/D2 弱监督标签器（数据组）。

从 trajectory_schema_v2 轨迹文档中，为 focal player（agent_index == human_index）
的每个 timestep 生成六类标签之一，或 unknown/invalid（intent_target=-1，
classification_mask=false）。

规则详见同目录 label_definitions.md。本脚本只依赖历史/未来窗口内可观察的
`step_records`（events 字段作为监督来源），不产出任何模型输入特征——特征工程属于
D2 阶段的 build_features.py，与本文件解耦。

用法（cole-platform 或后续 cole-data 环境均可，无 TensorFlow 依赖）：
    python labeler.py --input <trajectory.jsonl> --history-window 5 --future-window 15
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


LABEL_MAP = {
    "get_ingredient": 0,
    "put_in_pot": 1,
    "get_plate": 2,
    "plate_food": 3,
    "deliver": 4,
    "no_commitment": 5,
}
EVENT_TO_LABEL = {
    "ingredient_acquired": "get_ingredient",
    "ingredient_put_in_pot": "put_in_pot",
    "plate_acquired": "get_plate",
    "soup_plated": "plate_food",
    "soup_delivered": "deliver",
}
UNKNOWN_TARGET = -1


@dataclass
class LabelResult:
    episode_id: str
    timestep: int
    human_index: int
    intent_target: int
    label_name: str
    classification_mask: bool
    reason: str
    # 历史窗口 [t-W+1, t] 是否完整可用（不影响标签本身，仅供 build_features.py
    # 决定 history_mask 的 padding 位置，见 label_definitions.md v0.2 修正）。
    history_available: bool = True
    source_event_timestep: Optional[int] = None
    source_event_type: Optional[str] = None


def label_episode(
    step_records: List[Dict[str, Any]],
    history_window: int,
    future_window: int,
    episode_status_invalid: bool = False,
) -> List[LabelResult]:
    """对单个 episode 的每个 timestep 生成标签结果。"""
    n = len(step_records)
    if n == 0:
        return []
    human_index = step_records[0]["human_index"]
    episode_id = step_records[0]["episode_id"]

    # 预先收集该 episode 中 focal player 的全部事件，按 timestep 排列。
    events_by_timestep: Dict[int, List[Dict[str, Any]]] = {}
    for record in step_records:
        for event in record.get("events", []):
            if event["agent_index"] != human_index:
                continue
            events_by_timestep.setdefault(event["timestep"], []).append(event)

    results: List[LabelResult] = []
    last_timestep = step_records[-1]["timestep"]

    for record in step_records:
        t = record["timestep"]
        history_ok = t >= history_window - 1
        future_ok = t + future_window <= last_timestep

        if episode_status_invalid:
            results.append(
                LabelResult(
                    episode_id, t, human_index, UNKNOWN_TARGET, "unknown/invalid",
                    False, "episode manifest status is invalid_episode",
                    history_available=history_ok,
                )
            )
            continue
        if not future_ok:
            results.append(
                LabelResult(
                    episode_id, t, human_index, UNKNOWN_TARGET, "unknown/invalid",
                    False, f"future window truncated (t+H={t+future_window} > last_timestep={last_timestep})",
                    history_available=history_ok,
                )
            )
            continue

        # 完整未来窗口内按 timestep 升序扫描第一个命中事件。
        hit = None
        conflict = False
        for future_t in range(t + 1, t + future_window + 1):
            evs = events_by_timestep.get(future_t)
            if not evs:
                continue
            task_evs = [e for e in evs if e["event_type"] in EVENT_TO_LABEL]
            if len(task_evs) > 1:
                conflict = True
                hit = task_evs[0]
            elif task_evs:
                hit = task_evs[0]
            break  # 命中最早的 timestep 即停止扫描

        if conflict:
            results.append(
                LabelResult(
                    episode_id, t, human_index, UNKNOWN_TARGET, "unknown/invalid",
                    False, "multiple task event types at same future timestep (supervision conflict)",
                    history_available=history_ok,
                    source_event_timestep=hit["timestep"], source_event_type=hit["event_type"],
                )
            )
            continue

        if hit is None:
            results.append(
                LabelResult(
                    episode_id, t, human_index, LABEL_MAP["no_commitment"], "no_commitment",
                    True, "no task event found in full future window",
                    history_available=history_ok,
                )
            )
            continue

        label_name = EVENT_TO_LABEL[hit["event_type"]]
        results.append(
            LabelResult(
                episode_id, t, human_index, LABEL_MAP[label_name], label_name,
                True, "first future task event within window",
                history_available=history_ok,
                source_event_timestep=hit["timestep"], source_event_type=hit["event_type"],
            )
        )
    return results


def load_trajectory_documents(path: str) -> List[Dict[str, Any]]:
    docs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def label_file(path: str, history_window: int, future_window: int) -> List[LabelResult]:
    all_results: List[LabelResult] = []
    for doc in load_trajectory_documents(path):
        if doc.get("schema_version") != "trajectory_schema_v2":
            raise ValueError(f"unexpected schema_version: {doc.get('schema_version')}")
        for episode_steps in doc["step_records"]:
            all_results.extend(label_episode(episode_steps, history_window, future_window))
    return all_results


def summarize(results: List[LabelResult]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for r in results:
        summary[r.label_name] = summary.get(r.label_name, 0) + 1
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="D1/D2 弱监督标签器")
    parser.add_argument("--input", required=True, help="trajectory_schema_v2 jsonl 文件路径")
    parser.add_argument("--history-window", type=int, default=5, help="历史窗口 W（步数）")
    parser.add_argument("--future-window", type=int, default=15, help="未来监督窗口 H（步数）")
    parser.add_argument("--output", default=None, help="输出逐样本标签 jsonl（可选）")
    args = parser.parse_args()

    results = label_file(args.input, args.history_window, args.future_window)
    summary = summarize(results)

    print(f"input={args.input}")
    print(f"history_window={args.history_window} future_window={args.future_window}")
    print(f"total_samples={len(results)}")
    for k in sorted(summary):
        print(f"  {k}: {summary[k]}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        print(f"wrote per-sample labels to {args.output}")


if __name__ == "__main__":
    main()
