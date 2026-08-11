"""build_audit.py — 生成 no_commitment_audit.jsonl（D1 人工抽查交付物）。

从两个样例轨迹的标签器输出中抽取正例、边界反例，附上人工核对结论（audit_note），
不重新实现标签逻辑，只做抽样与标注。
"""

import json


def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def main():
    base = r"C:\Users\36724\PycharmProjects\WelcomeScreen\Overcooked2Agent\02_数据组\data_pipeline"
    event_ep = {r["timestep"]: r for r in load(f"{base}\\_check_event_episode_labels.jsonl")}
    no_event_ep = {r["timestep"]: r for r in load(f"{base}\\_check_no_event_episode_labels.jsonl")}

    audits = []

    def add(record, audit_note, boundary_type):
        entry = dict(record)
        entry["audit_note"] = audit_note
        entry["boundary_type"] = boundary_type
        audits.append(entry)

    # --- no_commitment 正例：长时间无任务推进，历史/未来窗口均完整 ---
    add(
        no_event_ep[4],
        "首个历史窗口完整的时间步；未来 15 步内 focal player 无任何成功事件；"
        "判定 no_commitment 正确，不应被误标为具体任务。",
        "no_commitment_positive",
    )
    add(
        no_event_ep[200],
        "episode 中段，长时间无任务推进的典型样本；判定 no_commitment 正确。",
        "no_commitment_positive",
    )

    # --- no_commitment 反例（应为具体任务类，用于确认标签器不会漏判）---
    add(
        event_ep[6],
        "窗口 [7,21] 内命中 ingredient_acquired@7；正确标为 get_ingredient，"
        "不应被误标为 no_commitment——验证标签器在有未来事件时不会退化成 no_commitment。",
        "task_label_positive",
    )
    add(
        event_ep[24],
        "窗口 [25,39] 内第一个命中事件为 plate_acquired@39；标为 get_plate。"
        "人工确认：get_plate 语义是玩家正在推进【取得盘子】这一中期子任务，"
        "而不是当前正在拿食材——此样本发生在 ingredient_put_in_pot@15 之后、"
        "plate_acquired@39 之前，期间玩家可能仍在走位/等锅煮汤，标签指向下一个"
        "即将达成的子任务符合“任务倾向”定义。",
        "task_label_positive",
    )

    # --- v0.2 修正验证：历史不足不再强制 unknown，而是标签正常产出 + history_available=False ---
    add(
        event_ep[0],
        "v0.2 修正验证：episode 首步 t=0，历史窗口不足（history_available=False），"
        "但未来窗口 [1,15] 完整且命中 ingredient_acquired@2，标签仍正常判定为 "
        "get_ingredient（而非 unknown）。历史不足只影响 build_features.py 阶段的 "
        "history_mask padding，不再污染标签可靠性判定——这是本轮对 v0.1 草案的关键修正。",
        "history_padding_not_unknown",
    )
    add(
        no_event_ep[0],
        "同一修正在 no_commitment 场景下的验证：t=0 历史不足，但未来窗口完整且窗口内"
        "无事件，正确判为 no_commitment（而非此前 v0.1 误判的 unknown/invalid）。",
        "history_padding_not_unknown",
    )
    add(
        event_ep[33],
        "t+H=48 > last_timestep=47，未来窗口越界；正确判定 unknown/invalid。"
        "此为实测边界值，替代 label_definitions.md 早期草案中估计的 t=35。",
        "unknown_future_truncated",
    )
    add(
        no_event_ep[385],
        "400 步 episode 中未来窗口越界的边界样本（t+H=400>399）；判定 unknown/invalid，"
        "核对通过。",
        "unknown_future_truncated",
    )

    # --- 已知发现：短 episode + 大 H 导致某些任务类别系统性缺失 ---
    add(
        event_ep[32],
        "关键发现：t=32 是满足未来窗口不越界的最大时间步（t+H=47<=47），其窗口 "
        "[33,47] 同时包含 plate_acquired@39 / soup_plated@43 / soup_delivered@47，"
        "但规则始终选择【时间最早】的事件，故仍标为 get_plate。在本 episode 长度"
        "（48步）与 H=15 的组合下，soup_plated 和 soup_delivered 两个标签永远无法"
        "被选中——这不是标签器缺陷，而是参数与 episode 长度的必然结果，需在正式生成"
        "数据前调整 H 或改用按事件锚点采样，并在 quality_report.json 中报告。",
        "known_limitation_parameter_tuning",
    )

    out_path = f"{base}\\no_commitment_audit.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for a in audits:
            fh.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"wrote {len(audits)} audit entries to {out_path}")


if __name__ == "__main__":
    main()
