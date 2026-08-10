# 玩家意图置信度驱动的支持型 AI：最终设计规格

版本：1.1（冻结版）
日期：2026-08-10
适用仓库：`cole-platform`
唯一地图：`simple`（Cramped Room）

## 项目整体背景

本项目研究《Overcooked》中的人机合作：一名玩家和一名 AI 厨师共同取食材、煮汤、装盘和送餐。

现有游戏 AI 通常最关心团队得分，但高分不一定代表合作体验好。AI 可能抢走玩家正在做的任务、重复拿同一种物品、堵住狭窄通道，或者在没有理解玩家时擅自接管。因此，我们希望 AI 不只是“做得快”，还要知道什么时候应该帮助，什么时候应该让玩家主导。

项目时间很紧，10 人小组需要在次日下午完成复现、正式实验和汇报。因此，本项目直接使用 COLE 开源平台、官方地图和官方预训练模型，不从零搭建游戏平台，也不从头训练大型模型。最终成果包括自动实验指标，以及 AI 在游戏中实际表现的录像。

## 方案调整摘要

经过讨论，项目从较宽泛的研究计划收缩为一个可以按时完成的方案：

- 平台固定为 COLE-Platform，不再寻找其他游戏平台。
- 地图固定为 `simple / Cramped Room`，不测试陌生地图。该地图空间狭窄，更容易观察抢任务、重复劳动和挡路。
- 对比模型固定为 COLE、FCP、MEP，直接运行官方预训练权重，不重新训练 baseline。
- 我们的方法以 COLE 为基础并冻结 COLE，只训练两个小模块：玩家意图判断和动作修正。
- 正式数据由程序化玩家自动生成，不依赖临时招募大量真人。
- 正式证据以本地模拟实验为主；少量真人试玩和游戏录像只用于展示。
- 评价重点从“只看团队得分”调整为“得分基本保持，同时减少干扰、挡路和重复任务”。
- 本次不做消融实验、角色分配、多 AI 集群或多地图扩展。

## 1. 项目目标与边界

本项目在官方 COLE 模型外增加两个小模块，使 AI 根据玩家意图和判断把握程度调整介入强度。目标是在团队任务正常推进的同时，减少抢任务、重复劳动、挡路和擅自接管。

本次范围固定如下：

- 只使用 `simple / Cramped Room`。
- 基础 COLE 权重完全冻结。
- 只训练意图判断模块和动作修正模块。
- 正式比较 COLE、FCP、MEP、Ours 四种方法。
- 每种方法固定运行 400 局。
- 不测试陌生地图，不做消融实验，不实现角色分配或多 AI 集群。
- 少量真人试玩只用于录像展示，不作为正式统计证据。

一句话方法：COLE 先提出动作，我们的方法根据最近 5 步玩家行为判断其意图；判断明确时主动补位，判断一般时只做可撤销辅助，判断不清时让路观察。

## 2. 运行环境与兼容性

执行人员必须使用仓库要求的兼容环境，不直接使用当前机器的 Python 3.11：

- Python 3.7；
- TensorFlow 1.15.0，用于加载官方 COLE、FCP、MEP SavedModel；
- PyTorch 1.12.1，用于新增的两个模型；
- Gym 0.21.0；
- NumPy 1.21.4；
- 地图配置使用仓库 `OvercookedMultiEnv` 的默认 400 步局长。

官方模型沿用 TensorFlow SavedModel 文件夹格式。新增模型保存为 PyTorch `.pt` 文件。两个框架只在推理包装器中顺序调用，不互相转换权重。

动作编号必须通过仓库的 `Action.ACTION_TO_INDEX` 和 `Action.INDEX_TO_ACTION` 转换，禁止在多个文件中手写编号。仓库当前顺序是北、南、东、西、停留、交互。

## 3. 统一目录与配置源

新增代码统一放在 `cole-platform/player_centric/`：

```text
player_centric/
  config.yaml              # 所有固定参数的唯一来源
  features.py              # 状态特征与5步历史
  partner_policy.py        # 四类模拟玩家
  teacher.py               # 规则标注器
  generate_data.py         # 原始轨迹和数据集生成
  dataset.py               # 数据读取、标准化和校验
  models.py                # IntentGRU 与 ActionAdapter
  train_intent.py          # 意图模型训练和概率校准
  train_adapter.py         # 动作修正模型训练
  agent.py                 # PlayerCentricAgent
  metrics.py               # 六项统一指标
  evaluate.py              # 四种方法共用的评测入口
  record_video.py          # 固定场景录像
  tests/                   # 单元测试和冒烟测试
```

输出统一放在：

```text
artifacts/
  raw_trajectories/{train,val,test}/
  datasets/
  checkpoints/
  results/
  logs/
  videos/
```

地图、阈值、种子、数据数量、训练参数、权重路径和评测局数全部从 `config.yaml` 读取，其他代码不得复制这些常量。

`config.yaml` 必须包含以下键；键名作为脚本之间的固定协议：

```yaml
layout: simple
horizon: 400
history_length: 5
low_threshold: 0.45
high_threshold: 0.70
seed: 42
train_seed_start: 100000
val_seed_start: 200000
test_seed_start: 300000
eval_seeds: [0, 1, ..., 49]
intent_train_per_class: 10000
intent_val_per_class: 2000
intent_test_per_class: 2000
adapter_train_size: 60000
adapter_val_size: 12000
adapter_test_size: 12000
episode_count_per_method: 400
cole_path: models/simple/COLE
fcp_path: models/simple/FCP
mep_path: models/simple/MEP
```

`eval_seeds` 在真实 YAML 中必须展开为 0 到 49 的完整整数列表，不得把省略号写入配置文件。

## 4. 系统数据流

每个游戏步骤严格按以下顺序执行：

1. 从原始 `OvercookedState` 提取玩家视角和 AI 视角特征。
2. 更新玩家最近 5 步的状态与动作历史。
3. 意图模型输出六类概率。
4. 使用校准参数修正概率，并取最大概率作为置信度。
5. 根据固定阈值确定高、中、低三档介入模式。
6. 冻结的 COLE 输出一个原始动作。
7. 动作修正模型输出 `KEEP` 或一个替代动作。
8. 安全检查排除非法移动和明确碰撞。
9. 向环境返回六种标准动作之一。

正式模拟接口固定为：

```text
PlayerCentricAgent.reset()
PlayerCentricAgent.set_agent_index(index)
PlayerCentricAgent.set_mdp(mdp)
PlayerCentricAgent.action(state) -> Action.ALL_ACTIONS 中的一个动作
```

一局结束后必须调用 `reset()`，清空 5 步历史和堵塞计数。

## 5. 状态特征

不使用截图或像素输入。每一步调用仓库：

```text
mdp.featurize_state(state, medium_level_planner)
```

该函数返回双方各自视角的一维特征。代码在运行时读取其长度 `F`，不得假定固定维数。

`MediumLevelPlanner` 每个进程只为 `simple` 地图创建一次并缓存，禁止每一步重新计算。官方 COLE、FCP、MEP 仍使用它们原本要求的 `mdp.lossless_state_encoding`；`featurize_state` 只供新增模块使用，不能替换官方模型输入。

意图模型每一步的输入为：

```text
玩家视角状态特征 F
+ 玩家上一步动作 one-hot 6
= F + 6
```

连续 5 步组成 `[5, F+6]`。一局开始不足 5 步时，重复最早的有效状态补齐；玩家上一动作未知时使用“停留”。

动作修正模型使用：

```text
AI当前视角状态 F
+ 意图概率 6
+ 置信度 1
+ 介入档位 one-hot 3
+ COLE原始动作 one-hot 6
= F + 16
```

所有连续特征只使用训练集计算均值和标准差。标准差小于 `1e-6` 的维度按 `1` 处理。统计量保存为 `artifacts/checkpoints/normalization.npz`，验证、测试、正式实验和浏览器展示只能读取该文件。

## 6. 六类意图与固定编号

```text
0 fetch_onion       取洋葱
1 place_onion       把手中洋葱放入锅
2 fetch_dish        取空盘
3 pickup_soup       用盘子装汤
4 deliver_soup      把成品送到交付区
5 reposition        等待、让路、绕行或暂时无明确任务
```

标签表示玩家正在追求的高层目标，而不是当步移动方向。玩家停顿或短暂绕路时仍保留原意图；只有生成器正式更换目标时才改变标签。

“无法判断”不是第七类。模型始终输出六类概率，不确定性由置信度表示。

## 7. 意图判断模型

模型名：`IntentGRU`。

固定结构：

```text
输入：[batch, 5, F+6]
单层 GRU：hidden_size=64，batch_first=True
取最后一个时间步的隐藏状态
Linear(64, 6)
Softmax 后得到六类概率
```

不使用 dropout，不增加卷积层或第二层 GRU。

训练参数固定为：

- 损失：交叉熵；
- 优化器：Adam；
- 学习率：`1e-3`；
- weight decay：`1e-5`；
- batch size：256；
- 最多 30 个 epoch；
- 验证集 Macro-F1 连续 5 个 epoch 没有提高 `1e-4` 时停止；
- 保存验证集 Macro-F1 最高的模型；
- 随机种子：42。

训练后在验证集上只优化一个温度参数 `T`，目标为最小化六分类交叉熵，并把 `T` 限制在 `[0.5, 5.0]`。正式概率为 `softmax(logits / T)`。保存：

```text
artifacts/checkpoints/intent_model.pt
artifacts/checkpoints/confidence_calibration.json
```

置信度为校准后六类概率的最大值，档位固定为：

- `< 0.45`：低；
- `0.45 <= confidence < 0.70`：中；
- `>= 0.70`：高。

阈值在本版本中冻结，不根据正式测试结果修改。

## 8. 动作修正模型

模型名：`ActionAdapter`。

固定结构：

```text
输入：[batch, F+16]
Linear(F+16, 128) + ReLU
Linear(128, 64) + ReLU
Linear(64, 7)
```

七类输出固定为：

```text
0     KEEP，保留COLE动作
1..6  采用 Action.INDEX_TO_ACTION[label - 1]
```

训练参数固定为：

- 损失：带类别权重的交叉熵；
- 每类权重为 `总样本数 / (7 × 该类样本数)`，再截断到 `[0.5, 3.0]`；
- 优化器：Adam；
- 学习率：`1e-3`；
- weight decay：`1e-5`；
- batch size：256；
- 最多 30 个 epoch；
- 验证集 Macro-F1 连续 5 个 epoch 没有提高 `1e-4` 时停止；
- 保存验证集 Macro-F1 最高的模型；
- 随机种子：42。

保存为 `artifacts/checkpoints/action_adapter.pt`。

## 9. 三档介入行为

- 高置信度：允许 `KEEP`，也允许执行取物、放锅、装汤、送餐等互补任务。
- 中置信度：允许 `KEEP`、移动、停留和不争抢的取物准备；禁止改变锅状态、装汤和送餐。
- 低置信度：只允许合规的 `KEEP`、停留或让开玩家预计路线；禁止交互动作。

动作修正模型给出结果后再施加上述动作许可。`KEEP` 只有在 COLE 原动作本身符合当前档位时才算可用。若最高概率输出不符合当前档位，则按七类输出概率从高到低检查其余类别，采用第一个允许的类别；七类都不可用时选择停留。

“玩家预计路线”统一按如下方式计算：取意图模型概率最高的类别，使用 `MotionPlanner` 求玩家当前位置到该意图最近可行目标的最短路径，并把路径第一步的目标格作为玩家预计下一格。意图为 `reposition`、不存在可行目标或路径求解失败时，把玩家当前格作为预计下一格。该预测只使用当前及历史状态，不读取程序化玩家尚未执行的下一动作。

连续 3 步双方位置都不变，或玩家连续 3 步尝试进入 AI 所在格时触发解除堵塞。此时优先选择不会进入玩家下一步预计位置的合法移动；没有这种移动时停留。解除堵塞最多持续 3 步，随后重新按置信度决策。

## 10. 程序化玩家

四类玩家都使用仓库路径规划器走向当前高层目标。每一步按“无效交互、停顿、非推进移动、规划动作”的顺序抽样，前一项命中后不再抽后续项。

| 玩家类型 | 无效交互 | 停顿 | 合法但不推进的移动 | 目标切换 |
|---|---:|---:|---:|---:|
| 果断型 | 0% | 2% | 3% | 0% |
| 犹豫型 | 3% | 20% | 10% | 0% |
| 低效率型 | 10% | 5% | 25% | 0% |
| 中途换目标型 | 3% | 5% | 5% | 每段任务 30% |

中途换目标只在当前目标已经持续 2 至 6 步后抽取一次；命中后从当前可行且不同的意图中均匀选择新目标。不存在其他可行目标时保持原目标。

训练数据中的四类玩家比例固定为 40%、25%、20%、15%。正式实验中四类玩家等量，各占 25%。

## 11. 原始轨迹与数据格式

数据生成必须推进完整合法游戏，禁止独立随机拼接状态。每局固定保存为一个 gzip 压缩的 `<episode_id>.jsonl.gz` 文件，每行对应一个时间步，必填字段为：

```text
episode_id
split
seed
style
timestep
player_index
state_dict
player_action
player_planned_action
true_intent
```

原始轨迹生成完后再派生两个 `.npz` 数据集。

意图数据字段：

```text
history_obs              float32 [N, 5, F]
history_player_actions   int64   [N, 5]
intent_label             int64   [N]
episode_id               int64   [N]
style                    int64   [N]
seed                     int64   [N]
```

动作修正数据字段：

```text
ai_obs                   float32 [N, F]
intent_probs             float32 [N, 6]
confidence               float32 [N]
confidence_band          int64   [N]，低/中/高为0/1/2
base_action              int64   [N]
adapter_label            int64   [N]，KEEP为0，动作标签为1..6
intervention_type        int64   [N]，KEEP/补位修改/让路为0/1/2
true_intent              int64   [N]
episode_id               int64   [N]
style                    int64   [N]
seed                     int64   [N]
```

## 12. 数据数量、平衡与种子隔离

意图数据集固定为 84,000 条：

- 训练集：每类意图 10,000 条，共 60,000 条；
- 验证集：每类意图 2,000 条，共 12,000 条；
- 测试集：每类意图 2,000 条，共 12,000 条。

每个意图内部都保持 40%、25%、20%、15% 的玩家类型比例。相邻窗口每隔 2 步抽取一次，同一意图段最多抽取 20 个窗口。

动作修正数据集也固定为训练 60,000、验证 12,000、测试 12,000 条。每个集合内部按介入类型保持：

- 40% `KEEP`；
- 30% 主动补位或修改；
- 30% 让路或解除堵塞。

每种介入类型内部同样保持 40%、25%、20%、15% 的玩家类型比例。

先按完整游戏局划分集合，再抽取窗口。同一局和同一随机种子不能跨集合。种子空间固定隔离：

```text
正式实验：0..49
训练数据：从100000开始递增
验证数据：从200000开始递增
测试数据：从300000开始递增
```

数据生成器持续生成完整局，直到每个集合的全部配额同时满足；超过配额的类别停止抽样，但游戏局仍正常推进到结束。

## 13. 规则标注器

规则标注器只是训练数据生成程序，不在正式推理时运行。它复用 `MotionPlanner`、`JointMotionPlanner` 和 `mdp.get_state_transition`。

每一步执行：

1. 读取玩家真实意图、玩家下一步规划动作、当前厨房状态和 COLE 原始动作。
2. 若 COLE 动作不会碰撞、挡路、争抢或重复任务，并符合当前置信度档位，标签为 `KEEP`。
3. 否则枚举六种原始动作，与玩家下一步规划动作组成联合动作。
4. 用 `mdp.get_state_transition` 模拟一步，排除撞墙、双方争夺同一格、进入玩家目标格和不符合档位的动作。
5. 从剩余动作中选择到互补任务目标最短路径代价最低者；代价并列时按仓库动作编号选择较小者，保证可复现。
6. 没有可用动作时，优先 `KEEP`；若 COLE 动作明确碰撞，则标为停留。

AI 已持有物品时优先完成手中任务：汤先送餐，盘子在汤就绪时先装汤，洋葱在锅未满时先放锅。AI 空手时，从下列任务中删除玩家当前任务，再选择第一个可行任务：

1. 交付已经装好的汤；
2. 给已经完成的汤装盘；
3. 把洋葱放入未满的锅；
4. 当锅正在煮或已经完成时准备盘子；
5. 为未满的锅取洋葱；
6. 移到不阻塞通道的位置。

若意图模型预测错误，或置信度低于 0.45，规则标注器不允许不可撤销交互，只生成 `KEEP`、停留或让路标签。

## 14. 固定训练顺序

执行顺序不能交换：

1. 生成并划分原始轨迹。
2. 派生 84,000 条意图数据并执行完整校验。
3. 只用训练集计算标准化参数。
4. 训练 `IntentGRU`。
5. 在验证集上拟合温度参数。
6. 冻结意图模型和温度参数。
7. 回放原始轨迹，运行冻结 COLE 和意图模型，由规则标注器生成动作修正数据。
8. 训练 `ActionAdapter`。
9. 在测试集上一次性评估两个模块。
10. 锁定全部权重后运行正式 1,600 局，不再改模型、阈值或规则。

## 15. 基础模型和正式实验

三个 baseline 使用官方预训练权重，不重新训练：

```text
models/simple/COLE/
models/simple/FCP/
models/simple/MEP/
```

若下载包内文件夹名称大小写不同，只在 `config.yaml` 中填写真实路径，结果中的方法名仍固定为 `COLE`、`FCP`、`MEP`。加载失败时记录错误并修复路径，禁止用其他权重冒充。

所有官方策略在正式实验中使用确定性动作，即选择动作概率最大的动作，不进行随机采样。

每种方法固定运行：

```text
4种玩家 × 50个种子 × 2个出生位置 × 400步 = 400局
```

四种方法共 1,600 局。每个“玩家类型 + 种子”运行两次并交换 AI 与玩家位置。四种方法使用完全相同的玩家生成器、种子和位置安排。

三位 baseline 负责人和 Ours 负责人必须调用同一个 `evaluate.py`。不允许复制后各自修改评测脚本。

## 16. 指标的唯一口径

每局 CSV 的必填列固定为：

```text
method, style, seed, ai_index, episode_length,
team_score, deliveries, collision_count,
duplicate_task_count, blocked_steps, interference_count
```

定义如下：

- 团队得分：环境原始稀疏奖励总和，不加入训练用 shaping reward。
- 完成订单数：成功交付事件次数。
- 碰撞或卡住：一方选择移动，但因为另一方占据目标格或双方争夺同一目标格而没有移动；撞墙不计入。
- 重复任务：使用同一个几何任务解码器，根据 AI 最近动作、手持物品和目标设施判断其高层任务；若与程序化玩家真实意图相同，该步记一次。同一连续重复段只计一次，直到任一方目标改变。
- 挡路步数：玩家规划动作指向 AI 当前格，且该步确实尝试进入该格。
- 干扰次数：AI 执行交互并拿取、放置或交付了生成器已分配给玩家的任务对象。同一交互只计一次。

所有次数同时报告每局总数和每 100 步次数。

几何任务解码器固定采用以下顺序：AI 持有洋葱时判为 `place_onion`，持有盘子时判为 `pickup_soup`，持有成品汤时判为 `deliver_soup`；AI 空手且最近两个有效移动连续缩短到洋葱供应点的最短距离时判为 `fetch_onion`，连续缩短到盘子供应点的距离时判为 `fetch_dish`，其余情况判为 `reposition`。同时满足两个空手条件时，选择当前最短路径距离较小者；仍相同时按意图编号较小者。

## 17. 统计与成功标准

先把同一“玩家类型 + 种子”的两个出生位置结果取平均，得到每种方法 200 个配对实验单位。然后用同一单位比较 Ours 与各 baseline。

每个指标报告：

- 方法均值；
- 与 COLE 的平均配对差值；
- 对 200 个配对单位进行 10,000 次 bootstrap 得到的 95% 置信区间。

核心成功标准固定为：

1. Ours 平均团队得分不低于 COLE 平均得分的 95%；
2. Ours 每 100 步干扰次数低于 COLE；
3. Ours 每 100 步挡路步数低于 COLE；
4. 同时报告重复任务和碰撞结果，不因结果不利而删除指标。

四类玩家合并结果是主要结论。每类玩家单独结果只用于解释，不声称每一类都获得统计保证。

意图模块在封存测试集上报告 Accuracy、Macro-F1、六类混淆矩阵，以及使用 10 个等宽区间计算的置信度校准误差。动作修正模块报告七分类 Macro-F1 和 `KEEP / 修改 / 让路` 三类合并后的 Macro-F1。

## 18. 本地运行与浏览器展示

正式实验全部在本地 Python 模拟器中无渲染运行，不依赖官方云服务器。

浏览器展示沿用仓库 Flask 平台。`/predict` 收到 `Ours` 时必须把重建后的原始 `OvercookedState` 交给 `PlayerCentricAgent.action(state)`，不能只传当前一帧编码，否则无法维护 5 步历史。

演示只支持单机单局。历史缓存键使用 `请求端IP + layout + npc_index`，当 `timestep == 0` 或时间步倒退时调用 `reset()`。每次返回仍是仓库原有的整数动作编号，前端动作协议不变。

正式录像场景固定为：

```text
地图 simple
玩家类型 中途换目标型
随机种子 42
AI位置 0 和 1 各录一局
模型 COLE 和 Ours
```

最终视频并排展示相同种子和位置下的 COLE 与 Ours，并叠加当前意图、置信度档位、COLE 原动作和最终动作。

## 19. 安全回退

- 历史不足 5 步：重复最早有效帧。
- 意图输出包含 NaN、Inf 或概率和异常：使用均匀六类概率并按低置信度处理。
- 校准文件缺失：拒绝启动 Ours，不使用未校准概率悄悄继续。
- 动作修正模型文件缺失：拒绝启动 Ours，不把 COLE 结果标成 Ours。
- 修正动作非法或不符合当前档位：按模型概率降序选择下一个允许动作。
- 所有替代动作都不可用：使用 `KEEP`；COLE 动作明确碰撞时停留。
- 一局结束、时间步归零或更换玩家位置：必须清空历史。
- CSV 写入失败：停止该批实验并保留日志，不继续产生无记录结果。

## 20. 下发前检查与验收

数据生成验收：

- 三个集合种子无交集；
- 同一 episode 只属于一个集合；
- 数量、意图比例和玩家类型比例完全符合规格；
- 无 NaN、Inf、非法动作或超出范围的标签；
- 随机种子 42 重复生成两次时输出一致。

模型验收：

- IntentGRU 输入输出形状正确，概率和为 1；
- ActionAdapter 输出 7 类；
- 三个模型文件和标准化文件均能重新加载；
- 一局 reset 后历史长度归零；
- Ours 每一步只返回 `Action.ALL_ACTIONS` 中的动作。

指标验收：

- 用手工构造的撞墙、双方争格、AI 挡路、重复取物和正常通过场景测试计数器；
- 撞墙不能误算为双方碰撞；
- 连续重复任务段只能计一次；
- 两个出生位置必须在统计前配对平均。

正式运行前验收：

- COLE、FCP、MEP、Ours 各运行一局 20 步冒烟测试；
- 四个方法都产生相同字段的 CSV；
- 日志记录配置文件哈希、权重路径、随机种子和开始结束时间；
- 冒烟测试数据放入 `artifacts/results/smoke/`，不得并入正式结果。

## 21. 固定执行命令

实现完成后，执行人员只使用下列入口；脚本内部统一读取 `player_centric/config.yaml`：

```powershell
python -m unittest discover player_centric/tests
python -m player_centric.generate_data --stage raw
python -m player_centric.generate_data --stage intent
python -m player_centric.train_intent
python -m player_centric.generate_data --stage adapter
python -m player_centric.train_adapter
python -m player_centric.evaluate --method COLE
python -m player_centric.evaluate --method FCP
python -m player_centric.evaluate --method MEP
python -m player_centric.evaluate --method Ours
python -m player_centric.metrics --merge
python -m player_centric.record_video
```

每次正式评测启动时，把配置文件复制到 `artifacts/logs/<method>_config.yaml`，并在日志中记录该副本的 SHA-256、官方权重路径、新增权重路径和 Git commit。四个 `evaluate` 命令可以由不同成员并行运行，但必须使用同一份只读配置文件。

## 22. 人员交付物

- COLE 负责人：权重加载记录、400 局 CSV、日志、录像轨迹。
- FCP 负责人：权重加载记录、400 局 CSV、日志、录像轨迹。
- MEP 负责人：权重加载记录、400 局 CSV、日志、录像轨迹。
- Ours 执行负责人：两个模型权重、标准化与校准文件、400 局 CSV、日志、录像轨迹。
- 数据负责人：原始轨迹、两个 84,000 条数据集、split 清单和校验报告。
- 评测负责人：唯一公共评测脚本、合并结果表、bootstrap 统计文件和图表。
- 展示负责人：固定种子并排视频，不更改模型和实验数据。

最终汇报可以声称：在固定地图和程序化玩家条件下，支持型 AI 减少了被明确定义并自动统计的干扰行为。没有足量真人实验时，不得声称已经证明真实玩家体验显著提高。
