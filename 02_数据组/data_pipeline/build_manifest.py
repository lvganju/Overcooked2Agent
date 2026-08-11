"""build_manifest.py — 生成 manifest.json（D5 交接包元信息，交接03 要求项之一）。

manifest.json 记录交接包的版本、来源、内容清单与文件哈希，供 Agent 组核对
交接包完整性、可复现来源，以及区分 smoke（脚本化验证数据）与
formal（真实模型自对弈数据）两条数据线。

用法：
    python build_manifest.py --package-dir cole_task_intent_v0.2 --output cole_task_intent_v0.2/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_file_hashes(package_dir: Path, exclude_names=("manifest.json",)):
    entries = []
    for p in sorted(package_dir.rglob("*")):
        if p.is_file() and p.name not in exclude_names and "__pycache__" not in p.parts:
            rel = p.relative_to(package_dir).as_posix()
            entries.append({
                "path": rel,
                "size_bytes": p.stat().st_size,
                "sha256": sha256_of(p),
            })
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="cole_task_intent_v0.2")
    args = parser.parse_args()

    files = collect_file_hashes(args.package_dir)

    manifest = {
        "package_name": "cole_task_intent_v0.2",
        "version": args.version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "handoff_from": "数据组 (Data Team)",
        "handoff_to": "Agent 组 (Agent Team)",
        "handoff_reference_doc": "02_数据组/交接03_任务级意图数据集_数据组到Agent组.md",
        "data_lines": {
            "smoke": {
                "description": "脚本化生成的验证数据（scripted_chef_greedy + random_scripted，"
                                "layout=simple），用于验证标签器/特征构造/拆分/训练接口链路的正确性，"
                                "不代表真实模型行为分布，不建议直接用于正式模型训练。",
                "source_generator": "generate_scripted_chef_batch.py / generate_smoke_batch.py",
            },
            "formal": {
                "description": "COLE-Platform 官方 5 个真实 checkpoint（SP/PBT/FCP/MEP/COLE，"
                                "缺 BC）在 layout=random1 上自对弈生成的真实轨迹数据，是本交接包的"
                                "正式训练数据。",
                "source_generator": "generate_real_model_batch.py",
                "models_included": ["SP", "PBT", "FCP", "MEP", "COLE"],
                "models_excluded": ["BC"],
                "exclusion_reason": "BC 依赖 stable-baselines GAIL 的 .load()，"
                                     "与本项目 requirements 的 stable-baselines3/torch 生态不完全对齐，"
                                     "需要单独适配验证，本轮范围裁剪未包含。",
            },
        },
        "label_schema_version": "v0.2",
        "files": files,
        "num_files": len(files),
    }

    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote manifest.json to {args.output} ({len(files)} files hashed)")


if __name__ == "__main__":
    main()
