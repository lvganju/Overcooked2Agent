# training_interface_v0.2.md — D4 冻结训练接口（数据组 → Agent 组）

本文件冻结 `cole_task_intent_v0.2` 交接给 Agent 组的**训练输入接口**字段规范。字段名、
dtype、shape 一经冻结，Agent 组按此对接；数据组后续（D5 formal 生成）只能扩充行数，
不得改变本文件定义的字段结构，除非双方另行约定 v0.3。

## 1. 每条样本（一个 timestep）的字段

| 字段 | 类型 / 形状 | 说明 |
|---|---|---|
| `features` | `float32[W, F]`（本版本 `W=5, F=28`） | 历史窗口特征，见 `feature_spec.md` 白名单与 28 维顺序定义。索引 0 是窗口最早的一步，索引 `W-1` 是当前 timestep `t`。 |
| `history_mask` | `bool[W]` | `True`=该位置是真实历史步，`False`=episode 开头不足 `W` 步时的 padding（对应 `features` 该行全 0）。 |
| `intent_target` | `int64`，取值 `{-1, 0, 1, 2, 3, 4, 5}` | 六类标签见 `label_map.json`；`-1` 表示 `unknown/invalid`，不是训练类别。 |
| `classification_mask` | `bool` | `intent_target=-1` 时必为 `False`；否则为 `True`。分类损失只在 `True` 的样本上计算。 |
| `episode_id` | `str` | 拆分与审计用，禁止参与模型前向计算。 |
| `subject_id` | `str` | 见第 3 节：本版本用 `policy_id` 代替（无真实人类受试者标识）。 |
| `seed` | `int` | 环境组录制时的随机种子，审计用，不进入模型输入。 |
| `layout_id` | `str` | 布局标识，可选进模型（作为分类特征），不属于历史窗口浮点数组的一部分。 |
| `label_name` | `str` | `intent_target` 的可读名，仅供人工审计，不进入模型输入。 |

字段来源对照现有实现：`build_features.py::build_features_for_episode()` 产出
`episode_id/timestep/layout_id/history_mask/features/intent_target/classification_mask/label_name`；
`build_dataset.py` 在写出 split 文件时补充 `policy_id/seed`。**本文件冻结后**，
`build_dataset.py` 输出字段需重命名 `policy_id` 为 `subject_id`（对外接口统一用
`subject_id`，避免 Agent 组误以为这是"策略选择"特征而非"数据溯源标识"），已在本轮同步修改。

## 2. 明确排除在模型输入之外的字段（与 feature_spec.md 第3节一致）

`intent_target`、`classification_mask`、`label_name`、`episode_id`、`subject_id`、`seed`、
以及 `features`/`history_mask` 之外的任何原始 `step_records` 字段（`events`、
`team_reward`、`done`、未来 timestep 的任何内容）——这些只能用于监督、审计、拆分，
绝不能被送入模型前向计算。Agent 组接收后必须自行核对其训练代码的输入张量只取
`features`（+ 可选 `history_mask` 用于加权/池化），并在验收时"证明未来事件、标签和
steps_to_event 未进入输入白名单"（HANDOFF03 第100节验收要求）。

## 3. 已知限制：`subject_id` 语义降级说明

原始 HANDOFF03 设计中 `subject_id` 暗示真实人类受试者标识，用于拆分时避免同一受试者
跨 split。**本版本环境组交接的是程序化对局（无真实人类玩家）**，因此：

- `subject_id` 取值实际是生成策略标识（`scripted_chef_greedy` / `random_scripted`），
  不是人类身份。
- 拆分时按 `subject_id`（即 policy_id）分层 + 按 `episode_id` 整体分配（见
  `split_protocol.md`），语义上等价于"不同行为风格不跨 split"，但**不能**声称
  已验证"对未见真实人类玩家的泛化"——这一点必须写进 `quality_report.json` 和最终
  `HANDOFF_REPORT.md` 的限制章节，防止 Agent 组过度解读评估结果。

## 4. 冻结版本与变更记录

- v0.2（本次）：初始冻结，字段集合如上表。字段 `policy_id` 重命名为 `subject_id`
  （仅重命名，语义与取值不变）。
- 未来若需要新增字段（如真实人类 subject 标识、多布局几何特征），必须递增到 v0.3
  并在变更记录中注明向后兼容性（是否要求 Agent 组修改加载代码）。
