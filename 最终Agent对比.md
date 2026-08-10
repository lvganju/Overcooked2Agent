# 最终 Agent 对比：六项核心指标计算规范

## 1. 文档目的

本文件规定最终 Agent 与 baseline 的六项正文指标，包括统一定义、事件判定、计算公式、日志字段、汇总方法和边界情况。

正式比较三种方法：

1. 原始冻结 COLE；
2. COLE + 固定辅助规则；
3. COLE + 任务倾向置信度控制器（我们的方法）。

六项核心指标为：

1. 团队得分；
2. 不必要介入率；
3. 协作冲突率；
4. 玩家目标改变后的恢复时间；
5. 紧急处理成功率；
6. 玩家流畅度评分。

> 核心原则：团队得分用于证明基本能力没有明显下降，其余指标用于证明 AI 更少添乱、更会配合、更快适应玩家，同时在紧急情况下不会袖手旁观。

---

## 2. 公平比较的统一实验条件

三种方法必须尽可能使用相同条件：

- 相同地图；
- 相同初始状态；
- 相同订单序列；
- 相同玩家脚本、玩家行为风格或同一位真人玩家；
- 相同随机种子；
- 相同对局时长；
- 相同 COLE checkpoint；
- 相同动作频率和超时设置；
- 相同事件检测器和指标计算脚本。

建议使用“配对实验”：对每一个 `scenario_id + player_id/style + seed`，分别运行三种方法。比较时优先计算同一配对条件下的方法差值，以减少地图、玩家和随机性的影响。

### 2.1 推荐的原始日志字段

每个时间步至少记录：

```json
{
  "method": "cole|fixed_support|ours",
  "episode_id": "...",
  "scenario_id": "...",
  "seed": 0,
  "timestep": 0,
  "state": {},
  "human_action": 0,
  "ai_action": 0,
  "cole_action": 0,
  "final_ai_action": 0,
  "team_reward": 0.0,
  "human_task": "get_plate|null",
  "ai_task": "manage_pot|null",
  "mode": "assist|support|observe|emergency|null",
  "current_option": "...|null",
  "option_start": false,
  "option_end": false,
  "option_end_reason": "...|null",
  "p_task_max": 0.0,
  "p_no_commitment": 0.0,
  "override_cole": false,
  "override_reason": "...|null",
  "emergency": false,
  "emergency_type": "...|null",
  "events": []
}
```

原始 COLE 和固定规则没有 `mode`、`current_option` 等字段时，可以填 `null`，但状态、双方动作、奖励和事件必须保留。对三种方法都应运行同一套离线任务与冲突检测器，避免只利用我们的方法内部日志而造成不公平。

---

## 3. 指标一：团队得分

### 3.1 指标含义

团队得分衡量 AI 与玩家共同完成游戏全局目标的基本能力。它不是本项目唯一目标，但可以防止系统通过“什么都不做”获得很低的干扰率。

### 3.2 单局计算

只使用环境定义的团队任务奖励，不把为了训练方便添加的内部 shaping reward 混入正文指标。

```text
Score_e = 一局中所有 team_reward 的总和
        = Σ_t team_reward(e, t)
```

其中：

- `e` 表示一局；
- `t` 表示时间步；
- 如果平台只在成功配送时给奖励，则直接累加配送奖励；
- 如果不同地图奖励尺度不同，应分地图报告，或在同一地图内比较。

### 3.3 多局汇总

```text
MeanScore_m = mean(Score_e)，e 属于方法 m
```

正文建议报告：

- 平均团队得分；
- 标准差或 95% 置信区间；
- 与原始 COLE 的配对平均差值。

### 3.4 边界情况

- 对局异常退出：标记为 `invalid_episode`，不得静默删除，应报告数量和原因；
- 对局长度不同：优先统一时长；无法统一时补充“每 100 步得分”，但主表仍应使用同一时长；
- 团队得分相同但行为体验不同：由后五项指标区分；
- 完成订单数可作为附录指标，不必与团队得分同时占用正文主表。

### 3.5 所需字段

`method`、`episode_id`、`scenario_id`、`seed`、`timestep`、`team_reward`、`done`。

---

## 4. 指标二：不必要介入率

### 4.1 指标含义

不必要介入率衡量 AI 是否在没有合理协作理由时抢活、提前占用玩家所需资源，或启动与玩家任务无关的高层任务。它直接对应“让玩家保持主导性”。

### 4.2 为什么不能只统计 `override_cole`

原始 COLE 没有“覆盖自己动作”的概念，因此如果只统计 `override_cole`，就无法与 baseline 公平比较。

正式比较时，应对三种方法统一检测“AI 介入事件”：AI 开始一个新的高层任务、占用关键资源或明显改变玩家可执行路径。我们的方法可以用 `option_start` 辅助定位，但最终仍由同一离线检测器复核。

### 4.3 一次介入事件的定义

满足以下任一条件，开始一个新的 `intervention_event_id`：

- AI 开始新的高层任务；
- AI 获取盘子、食材、汤等关键物品；
- AI 占据锅边、配送口、唯一通道等关键位置；
- AI 的动作使玩家原计划必须绕行、等待或更换任务；
- 我们的方法启动新的 Assist 或 Support Option。

连续推进同一任务的多个时间步只算一次介入，不能每一步重复计数。任务结束、放弃或切换时结束该事件。

### 4.4 不必要介入判定

一次介入满足以下任一条件，记为 `unnecessary = true`：

1. AI 与玩家正在推进同一任务，形成重复劳动；
2. AI 抢占玩家即将使用的关键物品或关键位置；
3. 开始介入时不存在可执行的互补任务；
4. 介入导致玩家等待、绕行、放弃原任务或发生资源争抢；
5. AI 启动任务后因“与玩家冲突”“任务本来不需要”而终止；
6. 我们的方法在 `p_no_commitment` 高于配置门槛时进入 Assist；
7. 明确任务概率高，但环境状态不支持该任务，AI 仍强力介入。

以下情况不自动判为不必要：

- Option 没有完成，但玩家突然改变目标；
- 合理介入后被紧急事件打断；
- AI 做了必要让路；
- AI 执行最低限度紧急处理。

紧急事件单独由第五项指标评价，默认不进入不必要介入率分母。

### 4.5 单局计算

```text
UIR_e = N_unnecessary_interventions(e)
        ÷ N_non_emergency_interventions(e)
```

其中 `UIR` 为 Unnecessary Intervention Rate。

若某局没有任何非紧急介入：

- 将该局 `UIR_e` 记为 `NA`，不能强行记为 0；
- 同时报告该方法的总介入次数，避免“完全不帮助”看起来最好。

### 4.6 多局汇总

正文主值建议使用所有有效介入事件的总体比例：

```text
UIR_m = Σ_e N_unnecessary_interventions(e)
        ÷ Σ_e N_non_emergency_interventions(e)
```

同时在附录报告每局 `UIR_e` 分布和总介入次数。

### 4.7 质量控制

- 在正式计算前，由两名成员共同检查至少 30 个介入事件；
- 对“必要/不必要”意见不一致的事件先讨论并完善规则；
- 保存 `intervention_audit.jsonl`，记录判定原因和证据时间步；
- 不能把“所有未成功 Option”都直接判为不必要。

### 4.8 所需字段

双方任务、双方动作、持有物、关键位置、`option_start/end`、`option_end_reason`、`p_no_commitment`、环境一致性、互补机会、紧急标记和事件证据。

---

## 5. 指标三：协作冲突率

### 5.1 指标含义

协作冲突率把三类最直观的“玩起来不丝滑”现象合并统计：

1. 重复劳动；
2. 关键资源争抢；
3. 路径或交互位阻挡。

### 5.2 三类冲突的判定

#### A. 重复劳动

在同一短时间窗口内，玩家和 AI 推进相同高层任务，而且该任务通常只需一人执行。例如双方同时取盘子、同时取同一个食材，或同时前往完成同一个唯一交互。

以下情况不算重复劳动：锅还需要多个食材，双方分别拿取不同食材且都有效。

#### B. 关键资源争抢

满足以下任一条件：

- AI 抢先拿走玩家下一步要用的唯一关键物品；
- AI 占用玩家正在接近的锅边、配送口或工作台交互位；
- AI 的资源操作使玩家必须等待、改向或放弃当前任务。

#### C. 阻挡

AI 占据唯一通道或关键交互位置，并且玩家连续至少 `block_min_steps` 步无法沿合理路径推进。建议初始设置：

```yaml
block_min_steps: 2
conflict_merge_gap: 3
```

### 5.3 冲突事件合并

同一连续事件可能同时具有“争抢”和“阻挡”标签。正文总冲突数按唯一 `conflict_event_id` 计数，不重复相加；子类型可以在附录分别报告。

如果两次冲突之间的间隔小于或等于 `conflict_merge_gap`，而且对象和原因相同，应合并为一次事件。

### 5.4 单局计算

为避免对局长度影响，正文使用每 100 时间步的事件率：

```text
CCR_e = 100 × N_unique_conflict_events(e)
        ÷ N_valid_timesteps(e)
```

其中 `CCR` 为 Coordination Conflict Rate。

### 5.5 多局汇总

```text
CCR_m = mean(CCR_e)，e 属于方法 m
```

建议报告平均值和 95% 置信区间。附录可以拆分：

- 重复劳动事件/100 步；
- 资源争抢事件/100 步；
- 阻挡事件/100 步；
- 总阻挡持续步数。

### 5.6 边界情况

- 双方短暂相邻不算阻挡，必须影响玩家推进；
- 玩家主动撞向静止 AI 不一定算 AI 阻挡，应结合可替代路径和关键位置判断；
- 同一冲突不能在三个子类别中被总计三次；
- 三种方法必须使用同一冲突检测器和同一参数。

### 5.7 所需字段

每步双方位置、朝向、动作、任务、持有物、可行路径、关键格、交互对象和事件 ID。

---

## 6. 指标四：玩家目标改变后的恢复时间

### 6.1 指标含义

该指标衡量玩家改变计划后，AI 多久能够停止旧协作任务，并转为新的合理行为。它对应我们的方法是否真正根据任务倾向变化调整介入。

### 6.2 玩家目标改变时刻

首先定义 `t_change`。满足以下条件时认为玩家改变任务：

1. 玩家参考任务从任务 A 变为任务 B 或 `no_commitment`；
2. 新任务连续保持至少 `human_task_stable_steps` 步；
3. 有行为和环境证据支持该变化，而不是单步转向或避障。

建议初始配置：

```yaml
human_task_stable_steps: 3
ai_recovery_stable_steps: 3
```

程序化玩家可以使用其脚本中的真实任务状态作为参考；真人玩家应使用轨迹离线标注或统一任务检测器。三种方法必须共享相同的 `t_change` 列表。

### 6.3 AI 恢复时刻

定义 `t_recover` 为以下条件首次同时满足的时间步：

- AI 已终止或停止推进与旧任务 A 绑定的行为；
- AI 不再占用旧任务的关键资源或位置；
- AI 连续 `ai_recovery_stable_steps` 步执行以下任一行为：适配新任务、低风险 Support、Observe/让路或合理 COLE 行为。

对我们的方法，可使用 `option_end` 和 `termination_reason` 辅助判断；对 baseline，使用统一离线 AI 任务检测器判断，不能因为 baseline 没有 Option 就跳过。

### 6.4 单次目标改变的计算

```text
RecoveryTime_k = t_recover(k) - t_change(k)
```

单位为时间步。若游戏每秒执行固定步数，也可以换算成秒：

```text
RecoverySeconds_k = RecoveryTime_k ÷ steps_per_second
```

### 6.5 未恢复情况

若在允许窗口 `recovery_timeout` 内没有恢复：

- 标记为 `censored = true`；
- 恢复时间记为 `recovery_timeout`，同时单独报告恢复成功率；
- 不得直接删除这些困难案例。

建议初始配置：

```yaml
recovery_timeout: 20
```

### 6.6 多事件汇总

恢复时间通常偏斜，正文建议报告：

- 中位恢复时间；
- 四分位距或 95% bootstrap 置信区间；
- 超时未恢复数量。

若篇幅只能放一个数字，使用中位恢复时间。

### 6.7 边界情况

- 玩家只是绕过障碍但任务未变：不产生 `t_change`；
- 玩家在两个目标之间快速摇摆且未稳定三步：暂记为不确定，不计正式目标改变；
- 紧急事件迫使 AI 暂停旧任务：可以记为合理恢复，但需标注 `emergency_interrupted=true`；
- 玩家改变目标后又立即改回：如果未达到稳定步数，不纳入统计。

### 6.8 所需字段

玩家参考任务、AI 任务/Option、任务稳定步数、Option 结束原因、双方动作、持有物、关键位置、紧急事件和时间步。

---

## 7. 指标五：紧急处理成功率

### 7.1 指标含义

紧急处理成功率证明 Observe 不是消极发呆：即使 AI 无法可靠判断玩家的长期任务，也能处理眼前明确、即将造成损失的危险。

### 7.2 紧急事件目录

正式实验前冻结 `emergency_spec.yaml`。至少包含：

| 紧急类型 | 触发条件 | 失败结果 | 成功条件 |
|---|---|---|---|
| 汤/成品临界 | 成品已完成且存在明确处理时限 | 烧坏、长期占锅或错失处理窗口 | 在截止前完成最低必要取出/释放 |
| 订单临界 | 当前订单距超时低于门槛 | 订单超时 | 在截止前完成配送或必要最短链 |
| 唯一通道阻塞 | AI 挡住玩家唯一合理路径 | 玩家连续无法推进 | AI 在时限内让到安全格 |
| 唯一关键物品被 AI 占用 | AI 持有当前唯一关键物品且无进展 | 全局任务停滞 | AI 推进或释放到可交接位置 |
| 双方卡死 | 连续若干步没有有效位移或任务进展 | 持续僵局 | 解除卡死并恢复正常任务推进 |

每类紧急事件都必须明确：`start_condition`、`deadline_steps`、`success_condition`、`failure_condition`。

### 7.3 单个事件判定

事件 `j` 满足以下条件时记为成功：

```text
紧急触发后，在 deadline_steps 内达到 success_condition，
且没有先发生 failure_condition。
```

紧急处理应是最低限度动作。为了防止把一切都算作紧急，应同时记录非紧急情况下的错误触发，作为附录中的“紧急过度干预率”。

### 7.4 计算公式

```text
ESR_m = N_successful_emergencies(m)
        ÷ N_eligible_emergencies(m)
```

其中 `ESR` 为 Emergency Success Rate。

`eligible` 表示该事件在触发时确实可由 AI 在规定动作和时间内解决。不可解决事件需要在不知道方法结果的前提下由统一规则排除。

### 7.5 公平实验方法

建议准备一组固定紧急场景：相同初始状态、玩家动作脚本、订单时间和随机种子，分别让三种方法运行。不要让某种方法遇到更多或更容易的紧急事件。

### 7.6 多事件汇总

正文报告：

- 总紧急处理成功率；
- 成功事件数/有效事件数，例如 `18/20`。

如果各类样本充足，可在附录按紧急类型拆分。样本很少时必须同时报告原始数量，不能只报百分比。

### 7.7 边界情况

- 玩家自己已经在截止前明确处理，AI 无需介入：根据场景设计标记为“无需 AI”，不作为 AI 成功；
- AI 成功救场但造成新的严重冲突：紧急事件仍可记成功，但冲突由第三项指标惩罚；
- AI 提前很久接管任务：不能事后称为紧急处理；
- 服务超时导致安全让路：只有满足该紧急事件的成功条件才算成功。

### 7.8 所需字段

紧急类型、触发时刻、截止时间、成功/失败条件、AI 动作、玩家动作、结束时刻、结果和最低必要动作说明。

---

## 8. 指标六：玩家流畅度评分

### 8.1 指标含义

这是最终体验指标，衡量玩家是否感到 AI 帮助恰当、尊重其主导权，并且整体合作过程流畅、可预测。

### 8.2 推荐问卷

每局结束后，请玩家按 1—7 分评分：

1. **帮助恰当性**：“AI 在合适的时候提供了帮助。”
2. **玩家主导性**：“与 AI 合作时，我仍然能够主导自己的任务选择。”
3. **合作流畅性**：“我与 AI 的配合过程流畅，AI 的行为容易理解和预测。”

分值含义：

```text
1 = 非常不同意
2 = 不同意
3 = 比较不同意
4 = 一般
5 = 比较同意
6 = 同意
7 = 非常同意
```

如果加入负向题，例如“AI 经常妨碍我”，计算前必须反向计分：

```text
reverse_score = 8 - original_score
```

### 8.3 单局计算

```text
FluencyTrial = (Q_help + Q_agency + Q_smooth) ÷ 3
```

若缺少任一核心题，该局综合评分记为缺失，不用其余两题强行替代。

### 8.4 单个玩家的计算

先对同一玩家体验同一方法的多局评分取平均：

```text
FluencyParticipant(p, m)
= mean(FluencyTrial)，属于玩家 p、方法 m
```

然后再跨玩家求平均。不能把所有局直接混在一起，当作彼此独立的玩家样本。

### 8.5 实验安排

- 尽量采用被试内设计：每位玩家都体验三种方法；
- 隐藏方法名称，界面只显示 A/B/C；
- 随机或平衡三种方法的体验顺序，降低练习和疲劳影响；
- 每种方法使用相近难度的地图和订单条件；
- 正式评分前安排一局不计分的练习局；
- 记录玩家 ID，但公开报告中匿名化。

### 8.6 多玩家汇总

正文报告：

- 玩家级综合流畅度平均值；
- 标准差或 95% 置信区间；
- 有效玩家人数 `N`。

如果篇幅允许，可在附录分别展示帮助恰当性、玩家主导性和合作流畅性。

### 8.7 边界情况

- 不同玩家完成局数不同：必须先按玩家求平均；
- 玩家知道方法名称：可能产生期望偏差，应在报告中说明；
- 样本人数很少：不要过度宣称统计显著，应展示每位玩家的配对结果；
- 主观评分不能替代行为指标，应与前五项共同解释。

### 8.8 所需字段

`participant_id`、匿名方法编号、真实方法映射、对局顺序、三个题目分数、综合分数、缺失原因和时间戳。

---

## 9. 推荐输出文件

```text
evaluation_release_v1/
├── metric_config.yaml
├── emergency_spec.yaml
├── decision_logs/
│   └── *.jsonl
├── detected_interventions.jsonl
├── detected_conflicts.jsonl
├── detected_goal_changes.jsonl
├── detected_emergencies.jsonl
├── intervention_audit.jsonl
├── episode_metrics.csv
├── human_ratings.csv
├── summary_metrics.csv
└── METRIC_REPORT.md
```

### 9.1 `episode_metrics.csv` 推荐字段

```text
method,episode_id,scenario_id,player_id,seed,
team_score,intervention_count,unnecessary_intervention_count,
unnecessary_intervention_rate,conflict_event_count,
coordination_conflict_rate,recovery_event_count,
median_recovery_steps,recovery_timeout_count,
emergency_count,emergency_success_count,emergency_success_rate
```

### 9.2 `summary_metrics.csv` 推荐字段

```text
method,n_episodes,n_players,
team_score_mean,team_score_ci_low,team_score_ci_high,
unnecessary_intervention_rate,
coordination_conflict_rate_mean,
recovery_steps_median,
emergency_success_rate,
fluency_score_mean,fluency_score_ci_low,fluency_score_ci_high
```

---

## 10. 正文最终结果表

| 方法 | 团队得分 ↑ | 不必要介入率 ↓ | 协作冲突率/100步 ↓ | 恢复时间/步 ↓ | 紧急成功率 ↑ | 流畅度评分/7 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| 原始 COLE |  |  |  |  |  |  |
| COLE + 固定辅助规则 |  |  |  |  |  |  |
| 我们的方法 |  |  |  |  |  |  |

表格下必须注明：

- 对局数、玩家数和随机种子数量；
- 是否为相同场景的配对实验；
- 指标中的 `↑` 表示越高越好，`↓` 表示越低越好；
- 恢复时间建议使用中位数，其余指标说明使用平均值、总体比例还是玩家级平均值；
- 所有比例同时保留分子/分母原始数量。

---

## 11. 不放入正文主表的诊断指标

以下指标仍应记录，但放在附录或调试报告：

- 意图模型 Accuracy、Macro-F1、各类召回、ECE；
- Assist / Support / Observe 模式占比；
- 模式切换率和平均模式持续时间；
- Option 完成率与终止原因；
- `p_no_commitment` 高时误入 Assist 的次数；
- 紧急过度干预率；
- 完成订单数、超时数和空闲时间。

其中，`p_no_commitment` 高于门槛时进入 Assist 的次数应当为 **0**。这属于系统验收硬条件，而不是用来追求平均值的普通性能指标。

---

## 12. 推荐的报告结论句式

> 与原始 COLE 和固定辅助规则相比，我们的方法在团队得分基本保持的同时，降低了不必要介入率和协作冲突率，在玩家改变任务后恢复更快，并能在意图不确定时保留必要的紧急处理能力；玩家主观评分也表明其合作过程更加流畅，同时保留了玩家的主导性。

只有实际结果支持时才能使用上述完整结论。如果团队得分、紧急成功率或主观评分没有改善，应如实写成“基本持平”“未观察到明确差异”或报告对应限制。
