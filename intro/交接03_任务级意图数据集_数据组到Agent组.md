HANDOFF 03
任务级意图数据集交接指南
把历史行为转换成五种明确任务倾向与 no_commitment 监督数据，并严格区分 unknown。
负责单元
数据集负责人
输入门槛
platform_data_handoff_v2 验收通过
核心输出
cole_task_intent_v0.2
交接对象
Agent 组

交接目标  Agent 组收到的数据应回答“玩家正在推进哪个中期子任务”，而不是“玩家刚才做了哪个动作”。标签、输入、拆分和质量报告都必须可复查。

所需资料与权威来源
COLE Platform 官方仓库 — 经典 Overcooked 人机实验平台、轨迹保存与冻结策略入口
COLE 论文介绍页 — 核对 COLE 方法边界、实验设定与论文信息
COLE 论文 PDF — 方法和实验细节
Human-Aware RL — 经典 Overcooked 策略及旧 TensorFlow 接口
Overcooked-AI — 状态、动作、布局、规划和事件定义
从环境开始：本组开工操作
先做这一件事  先把自己的运行环境和上一交接包的最小测试跑通，再开始改代码。不要在无法复现输入的情况下直接开发。

1. 建立本组独立环境
新建并激活 `cole-data（建议 Python 3.10）`；把 Python、Conda、操作系统和关键包版本写入 setup_log.md。
复制上一组交接包到只读输入目录；记录交接包版本、文件哈希和来源提交，不直接改写上一组文件。
阅读 README、HANDOFF_REPORT、schema/config 和 known_issues，列出缺失文件与不一致项。
执行第一个最小测试：运行上一组 validate_raw_trajectory.py 和 replay_episode.py，验证一条交接轨迹
测试通过后保存完整命令、退出码和日志；失败时保存完整首个 traceback，不用截图代替文本日志。
git rev-parse HEAD python --version conda env export > environment_lock.yml
统一语义：模型预测的是任务倾向，不是单步行为
本版关键修正  移动、停留、转向和交互是观察到的行为；模型标签应表达玩家正在推进的中期子任务。控制器不能再对六类概率直接取最大值后决定介入强度。

输出
含义
是否允许直接触发 Assist
get_ingredient
玩家正在推进取得食材这一子任务
还需稳定性、环境证据和互补机会
put_in_pot
玩家正在推进向锅内放入食材
还需稳定性、环境证据和互补机会
get_plate
玩家正在推进取得盘子
还需稳定性、环境证据和互补机会
plate_food
玩家正在推进把熟汤装盘
还需稳定性、环境证据和互补机会
deliver
玩家正在推进配送成品
还需稳定性、环境证据和互补机会
no_commitment
目前没有足够证据表明玩家已承诺某一任务
不允许；进入 Observe 或低风险 Support
unknown / invalid
数据缺失、越界或无法可靠监督
不进入分类训练；运行时安全回退

task_probs = probabilities[五个明确任务] p_task_max = max(task_probs) p_no_commitment = probabilities['no_commitment']  允许 Assist = (p_task_max >= assist_enter)            and (p_no_commitment <= no_commitment_max)            and task_stable            and environment_consistent            and complementary_option_exists            and not player_conflict
分步任务与阶段交接
阶段 D1：冻结标签定义
五个任务类使用未来窗口内第一个成功任务事件作为弱监督依据。
no_commitment 仅用于可观察到持续无任务推进、且没有接近任何成功事件的可靠样本；规则和采样条件写入 label_definitions.md。
unknown/invalid 用于数据损坏、窗口不足以可靠判断、事件冲突等情况，设置 classification_mask=false，不作为第七类训练。
每类人工抽查正例与相邻负例；特别抽查 no_commitment 与短暂走位、改向、被堵路的边界。
阶段 D2：构造历史与未来监督窗口
model_input[t] = observable_history[t-W+1:t] label_source[t] = first_successful_task_event[t+1:t+H] # future states / event / steps_to_event 只能生成监督，禁止进入模型输入
允许进入模型
只能用于监督或分析
历史位置、朝向、动作、手持物
未来状态与未来成功事件
当前锅、台面、订单和双方相对位置
intent_target、steps_to_event
history_mask、layout 编码
最终奖励、人工标签备注

阶段 D3：拆分与质量控制
先按 episode、玩家/策略、种子划分 train/validation/test，再生成重叠窗口。
若训练与测试使用同一程序化玩家生成器，至少留出未见参数组合或行为风格；在报告中限制泛化结论。
先生成 smoke 版本：每类约 200 条，用于全链路；通过后再扩大 formal 版本。
quality_report.json 必须报告六类分布、unknown 比例、泄漏、NaN/Inf、输入白名单和人工抽查结果。
阶段 D4：冻结训练接口
字段
形式
features
float32 [W, F]
history_mask
bool [W]
intent_target
int64，0..5；unknown 为 -1
classification_mask
bool；unknown/invalid 为 false
episode_id / subject_id
仅用于拆分和审计

正式交接与验收
交接包名称：cole_task_intent_v0.2    接收方：Agent 组
必须交接的文件
cole_task_intent_v0.2/ ├── smoke/ │   ├── train.parquet │   ├── validation.parquet │   └── test.parquet ├── formal/  # smoke 通过后再生成 ├── manifest.json ├── stats.json ├── quality_report.json ├── label_map.json ├── label_definitions.md ├── no_commitment_audit.jsonl ├── feature_spec.md ├── split_protocol.md ├── sequence_dataset.py ├── validate_dataset.py ├── sample_batch.npz └── HANDOFF_REPORT.md
接收方现场验收
运行 validate_dataset.py，quality_report 为 PASS。
读取 train/validation/test 各一个 batch，shape、mask 与 label_map 一致。
各抽查一个明确任务类和三个 no_commitment 边界案例。
证明未来事件、标签和 steps_to_event 未进入输入白名单。
证明 episode、主体/策略和指定留出风格不跨 split。
交接完成判据  接收方在新终端或新环境中按 README 运行成功，机器可读验证脚本通过，并在 HANDOFF_REPORT.md 中记录版本、已通过项、未完成项和已知限制。

如何让 AI 按当前进度指导
必须上传  将本组目录中的 `AGENTS.md` 与当前代码一起上传。若工具支持仓库级 AGENTS.md，应把它放在本组工作目录根部；普通对话则手动上传。

上传本组 AGENTS.md。
上传本指南和上一组的 HANDOFF_REPORT.md。
上传当前代码、配置和目录树，不要只描述文件名。
上传最近一次完整命令、完整首个报错及工作目录。
上传当前阶段证据：环境交接包、标签定义、no_commitment 抽查、manifest/stats/quality_report、生成与验证日志。
要求 AI 先判断当前阶段和缺失证据，本轮只推进一个验收门槛。
推荐开场消息
请先读取 02_数据组/AGENTS.md、本工作指南和上一组 HANDOFF_REPORT。 请根据实际文件判断我当前阶段，指出缺失证据；本轮只推进下一个验收门槛。 修改前先读代码，修改后运行最小测试，并列出本次新增或更新的交接文件。
