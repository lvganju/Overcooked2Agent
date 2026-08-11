# label_definitions.md — 六类任务倾向标签定义（v0.1，D1 冻结草案）

## 1. 标签体系

模型预测的是玩家正在推进的中期子任务倾向，不是单步动作。标签依赖“未来窗口内第一个
成功任务事件”作为弱监督信号；未来事件本身绝不进入模型输入，只用于生成标签。

| id | 标签 | 对应环境组事件 | 含义 |
|---|---|---|---|
| 0 | `get_ingredient`  | `ingredient_acquired`   | 玩家正在推进取得食材 |
| 1 | `put_in_pot`      | `ingredient_put_in_pot` | 玩家正在推进向锅内放入食材 |
| 2 | `get_plate`       | `plate_acquired`        | 玩家正在推进取得盘子 |
| 3 | `plate_food`      | `soup_plated`           | 玩家正在推进把熟汤装盘 |
| 4 | `deliver`         | `soup_delivered`        | 玩家正在推进配送成品 |
| 5 | `no_commitment`   | （无）                   | 当前没有足够证据认定玩家已承诺某一任务 |
| -1 | `unknown/invalid` | （无）                  | 窗口损坏、越界或监督冲突，不参与六类分类训练 |

`intent_target` 取值 0..5；`unknown/invalid` 用 `intent_target=-1`，并设
`classification_mask=false`。

## 2. 主体（focal player）约定

- 每个 `step_record` 含两名玩家。**标签只针对 `agent_index == human_index` 的玩家**
  （即被 COLE 观察、可能被协助的对象），不对 COLE/程序化伙伴一侧打标签。
- `human_action` 字段名称具有历史沿革，在程序化实验中代表 focal player policy 的动作，
  不代表真人数据；这不影响其作为标签主体的选择，只影响“未来是否可泛化到真人”的结论
  （在 `quality_report.json` 与 `split_protocol.md` 中需要显式限定）。

## 3. 未来窗口弱监督规则

对每个时间步 `t`（0-indexed，属于某个 episode 的 `step_records`）：

```
label_source[t] = first_successful_task_event(agent_index=human_index)
                  in step_records[t+1 : t+H]
```

- `H` 为未来监督窗口长度（步数），smoke 阶段暂定 `H=15`，随特征窗口 `W`（历史窗口，暂定
  `W=5`）一起在 D2 阶段与 Agent 组联合确认，若确认调整需在本文件记录版本变更。
- 在 `step_records[t+1 : t+H]` 范围内，按 `timestep` 升序扫描该 episode 中
  `agent_index == human_index` 的事件；若命中五类事件之一，取**第一个**命中事件的
  `event_type` 映射为对应标签（0-4）。
- 若同一 timestep 内该 agent 命中多个事件类型（例如极端边界下同一步同时满足两条规则），
  记为**冲突**，标签置为 `unknown/invalid`（不作猜测/加权处理），并写入
  `no_commitment_audit.jsonl` 供人工复核。目前在 `event_detector.py` 的规则下，单步同一
  agent 命中多个事件类型的概率极低（不同事件要求互斥的 held_object 前置状态），复核时
  需确认样本是否为规则漏洞而非正常现象。

## 4. `no_commitment` 判定规则（可学习语义，非 Assist 许可）

`no_commitment` 只能来自**完整未截断的未来窗口**：

1. 未来窗口完整：`t + H <= last_timestep`（即窗口 `[t+1, t+H]` 不越过 episode 末尾）。
2. episode 的 `manifest.status != "invalid_episode"`。
3. 在完整的 `[t+1, t+H]` 窗口内，`agent_index == human_index` 没有命中任何五类事件。

当以上 3 项全部满足且第 3 项为“无事件”时，标签为 `no_commitment`（`intent_target=5`，
`classification_mask=true`）。

**禁止的近似**：不得仅凭“未来没有事件”就判 `no_commitment`，必须同时满足未来窗口完整（1）
与 episode 有效（2）。短暂走位、被墙/队友挡路、临时改向但窗口内确实没有成功事件的情况，
按规则仍然是合法的 `no_commitment` 正例，而不是需要排除的噪声——它反映的是“当前没有足够
证据认定玩家已承诺某任务”，而不是“玩家可能已承诺但表现异常”。是否将其细分为更多子类型
留待人工抽查后由 D2 决定是否需要置信度加权，本版本不引入。

**v0.2 修正（D2 阶段发现并订正）**：v0.1 草案曾把“历史窗口是否够长（`t >= W-1`）”也作为
`no_commitment`/`unknown` 判定的前置条件之一。重新核对交接接口的 `history_mask: bool[W]`
字段后确认：**历史是否够长是特征构造（feature padding + mask）的问题，不是标签可靠性的
问题**——标签只描述“未来会不会发生成功事件”，与“回看了多少步历史”无关。因此 v0.2 起，
`no_commitment`/任务标签的判定只看未来窗口和 episode 有效性；历史不足的样本（episode 开头
前 `W-1` 步）仍然产出该 timestep 的记录，但对应的 `history_mask` 前段为 `false`（padding），
不再因为历史不足而整体判为 `unknown/invalid`。

## 5. `unknown/invalid` 判定规则（不参与训练）

以下任一情况标记为 `unknown/invalid`（`intent_target=-1`，`classification_mask=false`）：

- 未来窗口越界（`t + H > last_timestep`，即 episode 结尾前 `H` 步）。
- episode `manifest.status == "invalid_episode"`（环境组已标记的异常对局，例如程序化脚本
  提前耗尽、运行失败）。
- 同一 timestep 内该 agent 命中多个事件类型（监督冲突，见第 3 节）。
- `state` 无法反序列化 / 缺少 schema 必需字段（结构损坏）。

`unknown/invalid` 不是第七个分类目标，训练损失掩码 `classification_mask=false` 时不计入
分类损失，只做统计与运行时安全回退依据。

**注意**：历史窗口越界（`t < W - 1`）**不在此列**——见第 4 节 v0.2 修正说明，历史不足通过
`history_mask` padding 在特征层处理，不影响标签判定。

## 6. 与环境组事件规范的对齐核对（D1 人工核对结论）

已核对 `event_audit.jsonl`（50 条：五类事件各 5 正例 + 5 负例）与
`task_event_spec_v2.md` 规则一致：

- `ingredient_acquired` 负例覆盖：手部无变化 / 拿到的是 dish 而非食材 / 拿之前手已持有 /
  单纯移动无交互 / 拿之前手非空——均符合“不得用 action 猜测意图”的边界。
- `ingredient_put_in_pot` 负例覆盖：食材消失但锅无变化 / 锅计数未增加 / 锅内类型不匹配 /
  食材仍在手上 / 未面向发生变化的锅。
- `plate_acquired` 负例覆盖：手无变化 / 拿到食材而非盘子 / 盘子被放下 / 拿之前手非空 /
  dish 出现在世界而非手中。
- `soup_plated` 负例覆盖：满状态锅未清空 / 汤未就绪 / 拿之前手中不是 dish / 拿之后手中
  不是 soup / 未面向清空的锅。
- `soup_delivered` 负例覆盖：无正奖励证据 / 未面向配送点 / 汤未完全煮熟 / 有限订单列表未
  缩短 / 配送的汤与当前订单不匹配。

结论：环境组事件检测器的证据链完整，`event_detector.py` 中没有基于 `action` 的意图猜测；
数据组可以直接采用五类事件时间戳作为弱监督来源，无需重新实现事件检测逻辑。

## 7. 样例校验（D1 冒烟核对，已用 `labeler.py` 实测验证）

使用 `sample_event_episode.jsonl`（1 个 episode，48 步，`last_timestep=47`，事件
时间线：`ingredient_acquired@2,7,12` → `ingredient_put_in_pot@5,10,15` →
`plate_acquired@39` → `soup_plated@43` → `soup_delivered@47`，均为
`agent_index=0=human_index`），`W=5, H=15` 下人工核对（实测值，见
`no_commitment_audit.jsonl`）：

- `t=0` → `history window truncated (t=0 < W-1=4)` → `unknown/invalid`，核对通过
  （episode 开头历史不足 4 步）。
- `t=6`（窗口 `[7,21]` 命中 `ingredient_acquired@7`）→ `get_ingredient`，核对通过。
- `t=24/28/32`（窗口分别为 `[25,39]/[29,43]/[33,47]`）均命中 `plate_acquired@39`
  作为窗口内**第一个**未来事件 → 全部标为 `get_plate`，核对通过；同时验证了规则
  “按 timestep 升序取第一个命中事件”的实现符合第 3 节定义。
- `t=33` 起（`t+H=48 > last_timestep=47`）→ `future window truncated` →
  `unknown/invalid`，为实测边界值（原草案误写为 `t=35`，现已更正）。

**重要发现（记录在案，供 D2 参数确定参考）**：在 `H=15` 且 episode 仅 48 步的条件
下，`plate_food`（`soup_plated@43`）与 `deliver`（`soup_delivered@47`）两类标签
在本样例中**从未出现**。原因：任何满足“未来窗口不越界”（`t+H<=47`）的 `t` 都
`<=32`，而窗口 `[t+1, t+15]` 只要包含 `plate_acquired@39` 就会优先命中它（因为
按时间升序取第一个事件），所以窗口不可能以 `soup_plated`/`soup_delivered` 作为
“第一个未来事件”被选中。这不是标签器缺陷，而是短 episode 与较大 `H` 组合下的
必然结果。**结论**：正式生成 smoke/formal 数据集时，`H` 需要相对 episode 长度
和相邻事件间隔审慎选取，且必须在 `quality_report.json` 中报告六类分布是否有类别
因窗口参数系统性缺失，若某类样本数接近 0，需要调小 `H` 或改用按事件重新采样锚点
（而非等距时间步）等策略，并记录在 `split_protocol.md`。

`sample_episode.jsonl`（400 步，全程无事件）验证：
- `t=0` → `history window truncated` → `unknown/invalid`，核对通过。
- `t=4`（首个历史窗口完整的时间步）起至 `t=384` → 全部 `no_commitment`，核对通过，
  用于验证长时间无任务推进场景不被误标为某个具体任务。
- `t=385` 起（`t+H=400 > last_timestep=399`）→ `unknown/invalid`，边界核对通过。

以上人工核对样例保存在同目录 `no_commitment_audit.jsonl`，均为 `labeler.py`
实测输出而非手工构造。
