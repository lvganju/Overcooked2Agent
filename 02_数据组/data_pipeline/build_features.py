"""build_features.py — D2 特征构造（数据组）。

从 trajectory_schema_v2 轨迹中，为 focal player 的每个有效 timestep 构造定长历史窗口
特征，仅读取 [t-W+1, t] 范围内允许的字段（见同目录 feature_spec.md 白名单），历史不足
用 0 padding 并在 history_mask 中标记。不读取任何 t 之后的字段，不读取 events/reward/
done 等结果性或标签来源字段。

与 labeler.py（D1）解耦：本文件只产出 features + history_mask；intent_target /
classification_mask 直接复用 labeler.py 的输出做 join，不在本文件重新判定标签。

用法：
    python build_features.py --input <trajectory.jsonl> --history-window 5 \
        --future-window 15 --output <features.jsonl>
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from labeler import label_episode, LABEL_MAP  # noqa: F401  (LABEL_MAP re-exported for callers)

HELD_OBJECT_VOCAB = ["none", "onion", "tomato", "dish", "soup"]
ACTION_DIM = 6  # overcooked_ai Action.NUM_ACTIONS
FEATURE_DIM = 2 + 2 + 5 + 2 + 2 + 5 + ACTION_DIM + 1 + 1 + 1 + 1  # = 28


def _one_hot(index: int, dim: int) -> List[float]:
    v = [0.0] * dim
    if 0 <= index < dim:
        v[index] = 1.0
    return v


def _held_object_index(held_object: Any) -> int:
    if held_object is None:
        return 0
    name = held_object.get("name", "none")
    return HELD_OBJECT_VOCAB.index(name) if name in HELD_OBJECT_VOCAB else 0


def _pot_summary(objects: List[Dict[str, Any]]) -> Dict[str, float]:
    """仅从 state.objects 中已出现的 soup 对象汇总，空锅不出现在 state 中（见 feature_spec.md 已知限制）。"""
    pot_count = 0
    max_ingredient_count = 0
    any_soup_ready = 0.0
    for obj in objects or []:
        if obj.get("name") != "soup":
            continue
        pot_count += 1
        state = obj.get("state")
        if isinstance(state, (list, tuple)) and len(state) >= 2:
            # 环境组约定：state = [ingredient_type, ingredient_count, cook_time_remaining] 一类结构；
            # 我们只提取通用数值字段，不假设具体语义之外的内容。
            counts = [x for x in state if isinstance(x, (int, float))]
            if counts:
                max_ingredient_count = max(max_ingredient_count, int(max(counts)))
        if isinstance(state, (list, tuple)) and len(state) >= 3 and state[-1] == 0:
            any_soup_ready = 1.0
    return {
        "pot_count": float(pot_count),
        "max_ingredient_count": float(max_ingredient_count),
        "any_soup_ready": any_soup_ready,
    }


def step_feature_vector(record: Dict[str, Any]) -> List[float]:
    """仅从单个 step_record（timestep <= t）计算定长特征向量，不引用任何其他 timestep。"""
    human_index = record["human_index"]
    other_index = 1 - human_index
    players = record["state"]["players"]
    self_p = players[human_index]
    other_p = players[other_index]

    vec: List[float] = []
    vec += [float(self_p["position"][0]), float(self_p["position"][1])]
    vec += [float(self_p["orientation"][0]), float(self_p["orientation"][1])]
    vec += _one_hot(_held_object_index(self_p.get("held_object")), 5)
    vec += [float(other_p["position"][0]), float(other_p["position"][1])]
    vec += [float(other_p["orientation"][0]), float(other_p["orientation"][1])]
    vec += _one_hot(_held_object_index(other_p.get("held_object")), 5)
    vec += _one_hot(record["human_action"], ACTION_DIM)

    pot_summary = _pot_summary(record["state"].get("objects"))
    vec += [pot_summary["pot_count"], pot_summary["max_ingredient_count"], pot_summary["any_soup_ready"]]

    order_list = record["state"].get("order_list")
    order_count = -1.0 if order_list is None else float(len(order_list))
    vec += [order_count]

    assert len(vec) == FEATURE_DIM, f"feature vector dim mismatch: {len(vec)} != {FEATURE_DIM}"
    return vec


def build_window_features(step_records: List[Dict[str, Any]], t: int, history_window: int):
    """为 timestep t 构造 [W, FEATURE_DIM] 特征窗口与 [W] history_mask。

    只允许索引 <= t 的 step_records 参与计算（下方切片已强制这一点，禁止越界读取未来）。
    """
    window_start = t - history_window + 1
    features: List[List[float]] = []
    mask: List[bool] = []
    for i in range(window_start, t + 1):
        if i < 0:
            features.append([0.0] * FEATURE_DIM)
            mask.append(False)
        else:
            features.append(step_feature_vector(step_records[i]))
            mask.append(True)
    return features, mask


def build_features_for_episode(
    step_records: List[Dict[str, Any]], history_window: int, future_window: int,
    episode_status_invalid: bool = False,
) -> List[Dict[str, Any]]:
    labels = label_episode(step_records, history_window, future_window, episode_status_invalid)
    out = []
    for label, record in zip(labels, step_records):
        t = record["timestep"]
        # 防泄露断言：只用 step_records[:t+1]（即到 t 为止的前缀）计算特征，
        # 与用完整 episode 计算的结果必须逐字节一致——见 feature_spec.md 第5节。
        features_full, mask_full = build_window_features(step_records, t, history_window)
        features_prefix, mask_prefix = build_window_features(step_records[: t + 1], t, history_window)
        assert features_full == features_prefix and mask_full == mask_prefix, (
            f"leakage detected at episode={label.episode_id} t={t}: "
            "features differ when computed from truncated vs full trajectory"
        )
        out.append(
            {
                "episode_id": label.episode_id,
                "timestep": t,
                "layout_id": record["layout_id"],
                "history_mask": mask_full,
                "features": features_full,
                "intent_target": label.intent_target,
                "classification_mask": label.classification_mask,
                "label_name": label.label_name,
            }
        )
    return out


def load_trajectory_documents(path: str) -> List[Dict[str, Any]]:
    docs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="D2 特征构造（防泄露自检内置）")
    parser.add_argument("--input", required=True)
    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--future-window", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    all_rows: List[Dict[str, Any]] = []
    for doc in load_trajectory_documents(args.input):
        if doc.get("schema_version") != "trajectory_schema_v2":
            raise ValueError(f"unexpected schema_version: {doc.get('schema_version')}")
        for episode_steps in doc["step_records"]:
            all_rows.extend(
                build_features_for_episode(episode_steps, args.history_window, args.future_window)
            )

    with open(args.output, "w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_masked_in = sum(1 for r in all_rows if r["classification_mask"])
    n_padded = sum(1 for r in all_rows if not all(r["history_mask"]))
    print(f"input={args.input}")
    print(f"total_rows={len(all_rows)} classification_mask_true={n_masked_in} rows_with_padding={n_padded}")
    print(f"feature_dim={FEATURE_DIM} history_window={args.history_window} future_window={args.future_window}")
    print(f"wrote features to {args.output}")
    print("leakage self-check: PASS (assert passed for every row)")


if __name__ == "__main__":
    main()
