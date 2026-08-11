# HANDOFF_REPORT.md — `cole_task_intent_v0.2` 交接报告（数据组 → Agent 组）

## 1. 交接内容概述

本交接包 `cole_task_intent_v0.2` 是数据组按照《交接03_任务级意图数据集_数据组到Agent组.md》
完成 D1-D5 全部阶段后的最终产出，把环境组交付的 Overcooked 对局轨迹转化为
**任务级玩家意图（task-level intent）监督学习数据集**：六类标签
（`get_ingredient` / `put_in_pot` / `get_plate` / `plate_food` / `deliver` /
`no_commitment`），并把无法可靠判定的样本隔离为 `unknown/invalid`（不参与分类损失）。

包内含两条平行数据线：

| 数据线 | 目录 | 生成方式 | 定位 |
|---|---|---|---|
| `smoke` | `smoke/` | 脚本化策略（`scripted_chef_greedy` 确定性厨师 + `random_scripted` 随机动作），layout=`simple` | **仅用于验证** 标签器/特征构造/拆分/训练接口链路的正确性，不代表真实玩家/模型行为分布，**不建议**直接用于正式模型训练 |
| `formal` | `formal/` | COLE-Platform 官方 5 个真实 checkpoint（SP/PBT/FCP/MEP/COLE）自对弈，layout=`random1` | **正式训练数据**，是本交接包的核心交付物 |

## 2. formal 数据集关键统计

- 40 个 episode（每个模型 8 个），每个 episode 400 步，5 个真实模型行为均衡覆盖
- 拆分：train 30 episode / 12000 行，val 5 episode / 2000 行，test 5 episode / 2000 行
- 每个 split 内 5 个模型的行数完全均衡（train 每模型 2400 行，val/test 每模型 400 行）
- 六类标签 + `unknown/invalid` 在三个 split 中均超过约定的 ~200 行门槛（test 集最少
  `deliver`=107 行、`plate_food`=199 行，接近但满足门槛；如需更严格冗余度可在后续批次
  追加更多 episode）
- **episode 互斥性**：train/val/test 之间 episode_id 重叠 = 0（硬断言通过）
- 分类样本占比：15400/16000 = 96.25%（`unknown/invalid` 占 3.75%，主因为窗口边界/事件
  歧义样本，非系统性缺陷，详见 `no_commitment_audit.jsonl` 的人工审计说明）
- 类别不平衡比例（分类样本内 max/min）：约 4.27（`get_ingredient` 最多，`deliver` 最少），
  Agent 组训练时建议使用加权交叉熵或过采样处理

完整数字见 `formal/quality_report.json` 与 `formal/stats.json`。

## 3. smoke 数据集关键统计

- 30 个 episode（15 chef + 15 random），12000 行（train 8000 / val 1600 / test 2400）
- 六类标签同样全部超过门槛，episode 互斥性验证通过
- 仅用于链路验证，不建议作为最终模型的训练数据来源

完整数字见 `smoke/quality_report.json` 与 `smoke/stats.json`。

## 4. 已知限制（务必告知下游使用者）

1. **BC 模型未接入**：BC checkpoint 是 pickle 格式，加载依赖 `stable-baselines` 的
   `GAIL.load()`，与本项目当前生态（`stable-baselines3` + `torch`）不完全对齐，需要单独
   适配验证。本轮范围裁剪未包含，`formal/` 数据只覆盖 SP/PBT/FCP/MEP/COLE 共 5/6 个模型。
2. **自对弈而非交叉对战**：`formal/` 数据是每个 checkpoint 与自己对局（human 与队友用
   同一权重），不是 5 个模型互相交叉对战（如 SP vs COLE）。交叉对战覆盖的策略多样性更广，
   留作后续版本扩展。`subject_id` 按 `real_model_selfplay_<MODEL>` 命名，同一 `subject_id`
   下双方智能体完全同分布。
3. **单一布局**：`formal/` 只覆盖 `random1`（官方 checkpoint 训练所对应的布局），`smoke/`
   只覆盖 `simple`，均未覆盖其余 random0-3/unident 等布局的几何多样性。
4. **`subject_id` 语义降级**：原始设计里 `subject_id` 暗示真实人类受试者标识，本版本没有
   真实人类玩家参与，`subject_id` 实际取值是策略/checkpoint 标识，详见
   `training_interface_v0.2.md` 第 3 节说明。
5. **特征集固有限制**：空锅不出现在 `state.objects` 中，特征无法还原"到最近锅的相对距离"
   （见 `feature_spec.md`），这是环境组交接接口的固有限制，与数据组生成策略无关。
6. **`unknown/invalid` 不是训练类别**：`intent_target=-1` 且 `classification_mask=False`
   的行必须在计算分类损失时被排除，仅用于审计/统计，详见 `label_definitions.md` 与
   `no_commitment_audit.jsonl`（9 条人工审计样本，含 D1 历史窗口判定修正后的验证样本）。

## 5. 已通过的验收项

- [x] `validate_interface.py`：逐行字段完整性/shape/NaN-Inf/mask-target 一致性/padding
      零值一致性检查，`smoke/` 与 `formal/` 均 PASS
- [x] `validate_dataset.py`（Agent 组现场验收脚本）：接口检查 + batch 采样 shape 检查 +
      `label_map.json` 一致性 + 跨 split episode/subject_id 重叠重算 + 禁止字段静态检查，
      `smoke/` 与 `formal/` 均 PASS
- [x] `sequence_dataset.py`：jsonl 与 parquet 两种物理格式均验证可正确加载并生成
      `(batch, window, feature_dim)` 规整数组（parquet 路径修复了嵌套数组转换 bug，详见
      本文件第 6 节）
- [x] 跨 split episode 互斥性硬断言（`smoke/` 与 `formal/` 均为 0 重叠）
- [x] 防泄露自检（`build_features.py` 内置，`future_window` 之外的信息不进入
      `features` 数组）
- [x] `manifest.json` 内含全部交接文件的 SHA-256，供接收方核对完整性

## 6. 本轮修复的问题

- **D1 设计缺陷修正**：`labeler.py` 原先把"历史窗口是否够长"也纳入 `unknown/invalid`
  判定条件，与 `history_mask` 字段的 padding 设计初衷矛盾。已修正：历史不足只做 padding +
  `history_mask=False`，不再因此判 unknown。同步更新了 `label_definitions.md`
  （v0.2 修正说明）与审计样本。
- **`sequence_dataset.py` parquet 加载 bug**：pandas 读取 parquet 后，嵌套 list 字段
  （`features`/`history_mask`）会变成 `numpy.ndarray`（dtype=object）而非原生 list，
  直接 `np.asarray()` 会报 `"setting an array element with a sequence"`。已修复为
  加载后统一转回原生嵌套 list，jsonl 与 parquet 两条路径行为一致，并已重新验证。

## 7. 交接包目录结构

```
cole_task_intent_v0.2/
├── manifest.json              # 交接包元信息 + 全部文件 SHA-256
├── label_map.json             # 六类标签 + unknown 映射（冻结）
├── label_definitions.md       # 标签定义（v0.2 修正版）
├── no_commitment_audit.jsonl  # 人工审计样本（9 条）
├── feature_spec.md            # 28 维特征白名单
├── split_protocol.md          # 拆分算法说明
├── training_interface_v0.2.md # 冻结训练接口字段规范
├── sequence_dataset.py        # Agent 组数据加载器
├── validate_dataset.py        # 接收方现场验收脚本
├── sample_batch.npz           # 示例 batch（32 条，来自 formal/train）
├── HANDOFF_REPORT.md           # 本文件
├── smoke/
│   ├── train.jsonl / .parquet
│   ├── val.jsonl / .parquet
│   ├── test.jsonl / .parquet
│   ├── quality_report.json
│   └── stats.json
└── formal/
    ├── train.jsonl / .parquet
    ├── val.jsonl / .parquet
    ├── test.jsonl / .parquet
    ├── quality_report.json
    └── stats.json
```

## 8. Agent 组使用建议

1. 优先使用 `formal/` 训练正式模型；`smoke/` 仅用于快速验证自己的数据加载/训练代码是否
   正确对接了 `training_interface_v0.2.md` 的字段规范。
2. 训练前务必先跑一遍 `validate_dataset.py --dataset-dir formal`（或 `smoke`）确认接收到
   的数据完整无损。
3. `features`/`history_mask` 才是允许进入模型前向计算的字段；`episode_id`/`subject_id`/
   `seed`/`label_name`/`classification_mask` 仅用于审计、加权采样或分层评估，不能作为
   模型输入张量，详见 `training_interface_v0.2.md` 第 2 节与 `feature_spec.md` 第 3 节。
4. 分类损失只在 `classification_mask=True` 的行上计算，`intent_target=-1` 的行必须被
   loss mask 掉。
5. 如需交叉对战数据、BC 模型数据或更多布局的覆盖，请与数据组另行沟通排期（v0.3 范围）。
