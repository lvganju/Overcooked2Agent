# split_protocol.md — D3/D4 拆分协议冻结（数据组）

## 拆分单位

**以 episode 为最小拆分单位**，同一 `episode_id` 的所有 timestep 行必须整体分配到
同一个 split（train / val / test），禁止按行随机拆分。理由：同一 episode 内相邻
timestep 的历史窗口高度重叠（`W=5`），按行拆分会让几乎相同的样本同时出现在 train
和 test 中，人为拉高验证/测试指标，属于数据泄露。

## 分层依据

按 `subject_id`（本版本取值为生成策略：`scripted_chef_greedy` / `random_scripted`，
语义降级说明见 `training_interface_v0.2.md` 第3节）分层，保证每个 split 内都同时
包含两种行为风格，避免某个 split 只含单一风格导致评估偏差。

## 具体算法（`build_dataset.py::split_episodes`）

1. 按 `subject_id` 分组。
2. 组内按 `episode_id` 字典序排序后，用固定种子（`seed=42`）洗牌，保证可复现。
3. 按 70% / 15% / 15% 近似比例切成 train / val / test（小样本组至少给 val 分配
   1 个 episode，避免空 split）。
4. 汇总各 `subject_id` 组的分配结果得到最终 split。

## 强制自检

`build_dataset.py` 在写出 `quality_report.json` 前执行硬断言：
`(train_episode_ids ∩ val_episode_ids) ∪ (train ∩ test) ∪ (val ∩ test) == ∅`。
若不为空集，构建流程直接抛异常终止（非警告打印），防止带泄露的数据集被静默交付。
当前 smoke 数据集（30 episode）已验证 `episode_overlap_across_splits: []`。

## 已知限制（需在 quality_report.json / HANDOFF_REPORT.md 中同步披露）

- 本版本只有两种 `subject_id`（两种脚本化策略），不是 HANDOFF03 设想的"程序化玩家
  生成器的未见参数组合"或"真实人类受试者"留出。是否满足"至少留出未见参数组合或
  行为风格"这一要求，取决于后续是否接入六个真实 checkpoint；当前用两种明显不同的
  脚本风格（贪心厨师 vs 纯随机）近似满足"风格留出"的精神，但严格来说两者都出现在
  train 中（只是不同 episode 实例），**不构成"完全未见的策略类型"留出**。
- 若后续接入真实模型（BC/SP/PBT/FCP/MEP/COLE），建议采用"留一 policy"策略：至少
  一种模型的全部 episode 只出现在 test，不出现在 train，以获得更有说服力的
  跨策略泛化证据。本版本 smoke 阶段暂不强制。
