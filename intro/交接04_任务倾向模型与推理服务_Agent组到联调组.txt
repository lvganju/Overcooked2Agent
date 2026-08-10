HANDOFF 04
任务倾向模型与推理服务交接指南
训练六输出任务倾向模型，并交付不会把 no_commitment 高置信度误当 Assist 许可的接口。
负责单元
模型与推理负责人
输入门槛
数据质量报告 PASS
核心输出
agent_handoff_v2
交接对象
联调展示组

交接目标  服务输出任务概率和不确定度；它不直接决定 Assist / Support / Observe。模型“很确定没有任务承诺”时，应明确告诉控制器 no_commitment 很高，而不是返回高介入许可。

所需资料与权威来源
COLE Platform 官方仓库 — 经典 Overcooked 人机实验平台、轨迹保存与冻结策略入口
COLE 论文介绍页 — 核对 COLE 方法边界、实验设定与论文信息
COLE 论文 PDF — 方法和实验细节
Human-Aware RL — 经典 Overcooked 策略及旧 TensorFlow 接口
Overcooked-AI — 状态、动作、布局、规划和事件定义
PyTorch GRU — 序列模型接口
scikit-learn 模型评估 — 分类和概率评估
从环境开始：本组开工操作
先做这一件事  先把自己的运行环境和上一交接包的最小测试跑通，再开始改代码。不要在无法复现输入的情况下直接开发。

1. 建立本组独立环境
新建并激活 `cole-intent（Python 3.10）`；把 Python、Conda、操作系统和关键包版本写入 setup_log.md。
复制上一组交接包到只读输入目录；记录交接包版本、文件哈希和来源提交，不直接改写上一组文件。
阅读 README、HANDOFF_REPORT、schema/config 和 known_issues，列出缺失文件与不一致项。
执行第一个最小测试：验证 smoke 数据，读取一个 batch，并运行多数类 baseline
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
阶段 A1：先证明训练管线正确
运行多数类 baseline 和单帧 MLP，确认指标脚本、标签顺序和 mask 正确。
在极小子集上过拟合；失败时先查标签错位、mask 和 shape，不先扩大模型。
只从 manifest 的 model_input_allowlist 读取特征。
阶段 A2：训练任务倾向 GRU
每帧特征编码 → GRU → 六分类头 输出顺序固定：get_ingredient, put_in_pot, get_plate, plate_food, deliver, no_commitment
先在 smoke 数据完成训练、评估、导出和服务全链路，再决定 formal 规模。
报告 Macro-F1、六类召回、混淆矩阵、NLL、Brier、ECE 和可靠性曲线。
单独报告 no_commitment 的 precision/recall，以及它与五个任务类的混淆。
阈值和温度缩放只使用 validation；test 不参与调参。
阶段 A3：冻结推理响应
{   probabilities: {五个任务类..., no_commitment},   predicted_task: 'get_plate' | null,   p_task_max: 0.68,   p_no_commitment: 0.22,   normalized_entropy: 0.47,   model_valid: true,   model_version: 'intent_gru_v0.2',   latency_ms: 7.1 }
接口禁令  不要只返回 max_probability，也不要返回由六类最大概率直接计算的 mode。predicted_task 在 no_commitment 为主或输入无效时应为 null。

阶段 A4：交付阈值候选与验证证据
assist_enter_task: 0.70 assist_exit_task: 0.55 no_commitment_max_for_assist: 0.25 support_enter_task: 0.45 stable_steps: 3 service_timeout_ms: 100
这些只是验证集候选值，最终模式还要检查环境一致性和互补 Option。
服务超时、非法 shape、全零 mask 或模型异常时 model_valid=false，由控制器安全回退。
提供固定请求覆盖五类任务、no_commitment、模糊分布和输入无效四类情况。
正式交接与验收
交接包名称：agent_handoff_v2    接收方：联调展示组
必须交接的文件
agent_handoff_v2/ ├── intent_model.pt ├── model_config.yaml ├── label_map.json ├── normalization.json ├── threshold_candidates.yaml ├── inference.py ├── inference_service.py ├── requirements.txt ├── evaluation.json ├── confusion_matrix.png ├── reliability_curve.png ├── no_commitment_cases.jsonl ├── test_inference.py ├── example_requests/ ├── example_responses/ └── HANDOFF_REPORT.md
接收方现场验收
干净环境安装并启动 /health 与 /predict_intent。
检查概率和为 1、六类顺序固定、p_task_max 与 p_no_commitment 正确。
发送 no_commitment 主导样例，predicted_task 为 null，接口不返回 Assist。
发送模糊与非法输入，model_valid/fallback 语义正确。
核对 checkpoint、数据版本、评估 JSON 和服务版本一致。
交接完成判据  接收方在新终端或新环境中按 README 运行成功，机器可读验证脚本通过，并在 HANDOFF_REPORT.md 中记录版本、已通过项、未完成项和已知限制。

如何让 AI 按当前进度指导
必须上传  将本组目录中的 `AGENTS.md` 与当前代码一起上传。若工具支持仓库级 AGENTS.md，应把它放在本组工作目录根部；普通对话则手动上传。

上传本组 AGENTS.md。
上传本指南和上一组的 HANDOFF_REPORT.md。
上传当前代码、配置和目录树，不要只描述文件名。
上传最近一次完整命令、完整首个报错及工作目录。
上传当前阶段证据：数据交接包、配置、训练日志、checkpoint、六类评估、可靠性和推理样例。
要求 AI 先判断当前阶段和缺失证据，本轮只推进一个验收门槛。
推荐开场消息
请先读取 03_Agent组/AGENTS.md、本工作指南和上一组 HANDOFF_REPORT。 请根据实际文件判断我当前阶段，指出缺失证据；本轮只推进下一个验收门槛。 修改前先读代码，修改后运行最小测试，并列出本次新增或更新的交接文件。
