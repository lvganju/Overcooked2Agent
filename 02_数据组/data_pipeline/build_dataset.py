"""build_dataset.py — D3 数据集拆分 + smoke 生成 + quality_report（数据组）。

流程：
1. 读取 smoke_raw/{chef,random} 下所有 *.trajectory.jsonl + 对应 *.manifest.json。
2. 对每个 episode 调用 build_features.build_features_for_episode()（内部已含
   labeler.py 的标签判定 + 防泄露自检），得到逐 timestep 记录。
3. 按 episode 做训练/验证/测试拆分（**不按行随机拆分**，同一 episode 的所有行
   必须落在同一个 split，避免同一局内的相邻帧同时出现在 train 和 test 造成信息
   泄露/虚高评估）。拆分在 policy_id 内分层，保证每个 split 都含两种 policy。
4. 写出 smoke_dataset/{train,val,test}.jsonl 与 quality_report.json。

用法（cole-platform 环境）：
    python build_dataset.py --raw-dir smoke_raw --output-dir smoke_dataset \
        --history-window 5 --future-window 15
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from build_features import build_features_for_episode, FEATURE_DIM


def load_episode(traj_path: Path, manifest_path: Path):
    doc = json.loads(traj_path.read_text(encoding="utf-8").splitlines()[0])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert doc["schema_version"] == "trajectory_schema_v2"
    # step_records 是 [[...]]（每个 episode 一个内层列表），我们的生成脚本每个文件只含 1 个 episode。
    assert len(doc["step_records"]) == 1, "expected exactly one episode per trajectory file"
    return doc["step_records"][0], manifest


def discover_episodes(raw_dir: Path) -> List[Dict[str, Any]]:
    episodes = []
    for traj_path in sorted(raw_dir.rglob("*.trajectory.jsonl")):
        manifest_path = traj_path.with_name(traj_path.name.replace(".trajectory.jsonl", ".manifest.json"))
        if not manifest_path.exists():
            continue
        step_records, manifest = load_episode(traj_path, manifest_path)
        if manifest.get("status") != "valid":
            continue  # invalid episode manifests 不参与特征/标签构建，仅在 quality_report 中统计
        episodes.append({"step_records": step_records, "manifest": manifest})
    return episodes


def split_episodes(episodes: List[Dict[str, Any]], train_ratio=0.7, val_ratio=0.15, seed=42):
    """按 policy_id 分层、按 episode 整体分配到 train/val/test，同一 episode 不跨 split。"""
    import random

    by_policy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        by_policy[ep["manifest"]["policy_id"]].append(ep)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for policy_id, eps in by_policy.items():
        eps_sorted = sorted(eps, key=lambda e: e["manifest"]["episode_id"])  # 确定性排序后再洗牌
        rng.shuffle(eps_sorted)
        n = len(eps_sorted)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n - n_train > 1 else 0
        train_eps = eps_sorted[:n_train]
        val_eps = eps_sorted[n_train:n_train + n_val]
        test_eps = eps_sorted[n_train + n_val:]
        splits["train"].extend(train_eps)
        splits["val"].extend(val_eps)
        splits["test"].extend(test_eps)
    return splits


SCRIPTED_LIMITATIONS = [
    "本 smoke 批次仅使用两类生成策略：'scripted_chef_greedy'（手工确定性走位脚本，"
    "保证覆盖全部五类事件）与 'random_scripted'（均匀随机动作），均不是六个训练模型"
    "（BC/SP/PBT/FCP/MEP/COLE）的真实行为分布。正式数据集生成前必须补充真实模型"
    "checkpoint 的 policy_adapter 适配，否则模型学到的将是脚本化行为而非真实人机协作模式。",
    "scripted_chef_greedy 在同一 episode 内高度重复（7 次相同循环），事件间隔规律，"
    "不代表真实玩家的时间不确定性；仅用于验证标签器/特征构造/拆分链路的正确性。",
    "布局仅覆盖 'simple' 单一 layout，未覆盖 random0-3/unident 等其他布局的几何多样性。",
    "空锅不出现在 state.objects 中，特征集无法还原到最近锅的相对距离（见 feature_spec.md）。",
]

REAL_MODEL_LIMITATIONS = [
    "本批次使用 COLE-Platform 官方发布的 5 个真实 checkpoint（SP/PBT/FCP/MEP/COLE），"
    "复用环境组自带的 pantheonrl/tf_utils.py::get_agent_from_saved_model 加载器（未修改），"
    "在 'random1' 布局上做自对弈（human 与队友使用同一 checkpoint）。",
    "BC（behavioural cloning）模型未接入：其加载路径依赖 stable-baselines GAIL 的 "
    ".load()，与本项目 requirements 的 stable-baselines3/torch 生态不完全对齐，需要单独"
    "适配验证，本轮范围裁剪未包含，属于已知缺口（六个模型中缺 1 个）。",
    "本轮为自对弈（同一 checkpoint 对同一 checkpoint），不是 5 个模型互相交叉对战；"
    "交叉对战（如 SP vs COLE）留作后续扩展，当前先验证真实模型行为下标签/特征/拆分链路"
    "的正确性。subject_id 因此按 checkpoint 名称区分（'real_model_selfplay_<MODEL>'），"
    "同一 subject_id 下双方智能体完全同分布，不含跨策略配对的多样性。",
    "布局仅覆盖 'random1' 单一 layout（六个 checkpoint 官方发布的其中一套权重对应此布局），"
    "未覆盖其他 5 个 random 布局或该 checkpoint 未训练过的布局。",
    "空锅不出现在 state.objects 中，特征集无法还原到最近锅的相对距离（见 feature_spec.md），"
    "此限制与生成策略无关，是环境组交接接口的固有限制。",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--future-window", type=int, default=15)
    parser.add_argument(
        "--dataset-label", choices=("smoke_scripted", "formal_real_model"), default="smoke_scripted",
        help="控制 quality_report.json 的 schema_version 与 known_limitations 文案",
    )
    args = parser.parse_args()

    episodes = discover_episodes(args.raw_dir)
    invalid_manifest_count = sum(
        1 for p in args.raw_dir.rglob("*.manifest.json")
        if json.loads(p.read_text(encoding="utf-8")).get("status") != "valid"
    )

    splits = split_episodes(episodes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    limitations = (
        REAL_MODEL_LIMITATIONS if args.dataset_label == "formal_real_model" else SCRIPTED_LIMITATIONS
    )
    quality: Dict[str, Any] = {
        "schema_version": f"cole_task_intent_v0.2_{args.dataset_label}",
        "history_window": args.history_window,
        "future_window": args.future_window,
        "feature_dim": FEATURE_DIM,
        "num_episodes_discovered": len(episodes) ,
        "num_invalid_manifest_skipped": invalid_manifest_count,
        "splits": {},
        "known_limitations": limitations,
    }

    for split_name, eps in splits.items():
        rows: List[Dict[str, Any]] = []
        for ep in eps:
            manifest = ep["manifest"]
            ep_rows = build_features_for_episode(
                ep["step_records"], args.history_window, args.future_window
            )
            for r in ep_rows:
                # training_interface_v0.2.md 第1节：对外字段统一用 subject_id
                # （本版本取值等于生成策略标识，语义降级说明见该文档第3节）。
                r["subject_id"] = manifest["policy_id"]
                r["seed"] = manifest["seed"]
            rows.extend(ep_rows)

        out_path = args.output_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        label_counts = Counter(r["label_name"] for r in rows)
        episode_ids = sorted({r["episode_id"] for r in rows})
        subject_counts = Counter(r["subject_id"] for r in rows)
        quality["splits"][split_name] = {
            "num_episodes": len(episode_ids),
            "num_rows": len(rows),
            "label_counts": dict(label_counts),
            "subject_row_counts": dict(subject_counts),
            "episode_ids": episode_ids,
        }
        print(f"{split_name}: episodes={len(episode_ids)} rows={len(rows)} labels={dict(label_counts)}")

    # 跨 split 的 episode 互斥性自检（硬性要求：同一 episode 不得同时出现在多个 split）
    all_ids_per_split = {name: set(info["episode_ids"]) for name, info in quality["splits"].items()}
    overlap = (
        (all_ids_per_split["train"] & all_ids_per_split["val"])
        | (all_ids_per_split["train"] & all_ids_per_split["test"])
        | (all_ids_per_split["val"] & all_ids_per_split["test"])
    )
    quality["episode_overlap_across_splits"] = sorted(overlap)
    assert not overlap, f"episode leakage across splits detected: {overlap}"

    quality_path = args.output_dir / "quality_report.json"
    quality_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"episode overlap across splits: {len(overlap)} (must be 0)")
    print(f"wrote quality_report.json to {quality_path}")


if __name__ == "__main__":
    main()
