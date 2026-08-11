# feature_spec.md — D2 特征白名单与窗口规范（v0.1）

冻结范围：`build_features.py` 从 `step_records` 构造模型输入特征的字段白名单、窗口定义、
防泄露规则。本文档与 `label_definitions.md`（标签规则）解耦：标签只依赖未来窗口，特征只
依赖历史窗口，两者互不影响。

## 1. 窗口定义

- 历史窗口 `W`（当前 smoke 阶段取 5）：对每个有效 timestep `t`，读取
  `step_records[max(0, t-W+1) : t+1]`（含 `t` 本身，最多 `W` 步）。
- `t` 自身的 `state`/`human_action`/`cole_action`/`final_ai_action` 属于"当前观测时刻已发生的
  信息"，允许进入历史窗口最后一位——它不是未来数据（对应 `t` 时刻采取的动作与 `t` 时刻的状态
  是同一次记录写入的，环境组录制时序如此，详见 `trajectory_schema_v2.json`）。
- 若 `t - W + 1 < 0`（episode 开头不足 `W` 步历史），**不跳过该样本**：用零向量在窗口前部
  padding，并将对应位置的 `history_mask` 置 `false`。真实历史步对应 `history_mask=true`。
  见 `label_definitions.md` v0.2 节的修正说明。

## 2. 特征字段白名单（每个历史步内，允许读取的 `step_records[i]` 字段）

| 来源字段 | 说明 | 编码 |
|---|---|---|
| `state.players[human_index].position` | focal 玩家坐标 (x,y) | float32 x2 |
| `state.players[human_index].orientation` | focal 玩家朝向 (dx,dy) | float32 x2 |
| `state.players[human_index].held_object.name` | focal 玩家手持物（`none/onion/tomato/dish/soup`） | one-hot x5 |
| `state.players[1-human_index].position` | 队友坐标 | float32 x2 |
| `state.players[1-human_index].orientation` | 队友朝向 | float32 x2 |
| `state.players[1-human_index].held_object.name` | 队友手持物 | one-hot x5 |
| `human_action` | focal 玩家在该 timestep 采取的动作（当前时刻已发生，非未来） | one-hot x6 |
| `state.objects[*]`（`name=="soup"`） | 世界中锅/组装中的汤对象：数量、`state`（配料计数、是否煮好） | 汇总为：pot_count(int), max_ingredient_count(int), any_soup_ready(bool) |
| `state.order_list` | 剩余订单数量（`null` 记为 -1） | int |
| `layout_id` | 布局标识（**静态**，每 episode 不变，单独作为分类特征，不进入历史窗口浮点数组） | categorical string |

**已知限制**（写入 `quality_report.json` 供 Agent 组知悉）：空锅（未放入任何配料）不会出现在
`state.objects` 中，因此无法从 state 直接还原"锅的固定位置"或"到最近空锅的距离"这类几何特征。
若 Agent 组需要该类特征，需要环境组额外提供 `layout_id -> 锅/操作台/配送点坐标` 的静态几何表
（当前交接包未包含），本版本特征集不包含任何依赖布局几何的相对距离特征。

## 3. 明确禁止进入特征的字段（防止数据穿越）

- 未来 timestep（`> t`）的任何字段（`state`、`events`、`human_action` 等）。
- `events` 字段本身（无论历史还是未来）——`events` 是监督标签来源，不是可观察状态，
  混入特征等于把标签信息泄露进输入。
- `team_reward`、`done`（结果性字段，可能与未来终止/得分强相关，不作为输入特征）。
- 任何 `intent_target`/`label_name`/`classification_mask`/`source_event_*`（标签器输出）。

## 4. 输出 schema（每个有效 timestep 一条记录）

```
{
  "episode_id": str,
  "timestep": int,
  "layout_id": str,
  "history_mask": [bool] * W,          # True=真实历史步，False=padding
  "features": [[float]] * W,           # 每步一个定长特征向量，padding 步全 0
  "intent_target": int,                # -1 或 0..5，来自 labeler.py，不作为输入
  "classification_mask": bool,
  "label_name": str
}
```

`features` 的每步向量维度与顺序固定为：
`[self_x, self_y, self_ori_x, self_ori_y, self_held(5 one-hot),
  other_x, other_y, other_ori_x, other_ori_y, other_held(5 one-hot),
  human_action(6 one-hot), pot_count, max_ingredient_count, any_soup_ready, order_count]`
共 `2+2+5+2+2+5+6+1+1+1+1 = 28` 维。

## 5. 防泄露自检机制

`build_features.py` 内置 `assert_no_leakage()`：对同一 episode，分别用完整 `step_records`
和截断到 `[:t+1]` 的前缀计算 timestep `t` 的特征，断言两者字节级相等。若不相等说明实现引用了
未来数据，视为严重 bug，构建流程直接失败退出（非警告）。已在两个样例 episode 上跑通该自检。
