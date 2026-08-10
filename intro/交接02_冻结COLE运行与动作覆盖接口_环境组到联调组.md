HANDOFF 02
冻结 COLE 运行与动作覆盖接口交接指南
环境组向联调展示组交付稳定运行底座、控制器插入点和双动作日志。
负责单元
环境运行接口负责人
输入门槛
交接 01 的 schema 已冻结
核心输出
platform_runtime_handoff_v2
交接对象
联调展示组

交接目标  联调组能够在完全不修改 COLE 权重的情况下读取历史、取得 cole_action、选择 final_action，并在控制器关闭时保持原始行为等价。

所需资料与权威来源
COLE Platform 官方仓库 — 经典 Overcooked 人机实验平台、轨迹保存与冻结策略入口
COLE 论文介绍页 — 核对 COLE 方法边界、实验设定与论文信息
COLE 论文 PDF — 方法和实验细节
Human-Aware RL — 经典 Overcooked 策略及旧 TensorFlow 接口
Overcooked-AI — 状态、动作、布局、规划和事件定义
从环境开始：本组开工操作
先做这一件事  先把自己的运行环境和上一交接包的最小测试跑通，再开始改代码。不要在无法复现输入的情况下直接开发。

1. 建立本组独立环境
新建并激活 `cole-platform（Python 3.7 旧栈）`；把 Python、Conda、操作系统和关键包版本写入 setup_log.md。
复制上一组交接包到只读输入目录；记录交接包版本、文件哈希和来源提交，不直接改写上一组文件。
阅读 README、HANDOFF_REPORT、schema/config 和 known_issues，列出缺失文件与不一致项。
执行第一个最小测试：运行 frozen_cole_smoke.py，固定输入得到合法 cole_action
测试通过后保存完整命令、退出码和日志；失败时保存完整首个 traceback，不用截图代替文本日志。
git rev-parse HEAD python --version conda env export > environment_lock.yml
分步任务与阶段交接
阶段 R1：建立默认透传插入点
cole_action = frozen_cole.predict(state) controller_input = {state, history, cole_action} final_action, decision_meta = controller.decide(controller_input) # controller disabled 时必须 final_action == cole_action
先实现 controller disabled 的透传行为，回归比较一批固定状态。
再实现 debug_force_action，只用于验证覆盖路径；默认配置必须关闭。
动作执行前检查合法性；非法覆盖回退 cole_action 或 safe_stay，并记录原因。
阶段 R2：冻结每步调用合同
输入/输出
必须字段
ControllerInput
state, recent_history, cole_action, timestep, layout_id
ControllerOutput
final_action, override_cole, override_reason, latency_ms
Decision log
cole_action 与 final_action 必须同时存在；保留 controller_version
Failure behavior
意图服务未接入或超时时默认透传/安全 Observe，不阻塞游戏

阶段 R3：给联调组提供可替换假服务
提供 mock_intent_service，能返回明确任务、no_commitment、低置信度、超时四种固定响应。
提供 scripted_controller，证明联调组可在不等待模型的情况下开发 Option 和 UI。
记录端口、超时、重试和服务断开时的回退配置。
正式交接与验收
交接包名称：platform_runtime_handoff_v2    接收方：联调展示组
必须交接的文件
platform_runtime_handoff_v2/ ├── README_RUNTIME.md ├── HANDOFF_REPORT.md ├── repo_commit.txt ├── environment_lock.yml ├── observation_spec.json ├── action_spec.json ├── controller_contract.json ├── decision_log_schema.json ├── controller_hook.py ├── mock_intent_service.py ├── frozen_cole_smoke.py ├── test_passthrough.py ├── test_action_override.py └── logs/
接收方现场验收
启动冻结 COLE 并运行 frozen_cole_smoke.py。
运行 test_passthrough.py，控制器关闭时动作逐项一致。
运行 test_action_override.py，覆盖动作生效且双动作日志完整。
模拟意图服务超时，游戏不中断并执行约定回退。
确认联调组不需要导入 COLE 的训练代码即可调用运行接口。
交接完成判据  接收方在新终端或新环境中按 README 运行成功，机器可读验证脚本通过，并在 HANDOFF_REPORT.md 中记录版本、已通过项、未完成项和已知限制。

如何让 AI 按当前进度指导
必须上传  将本组目录中的 `AGENTS.md` 与当前代码一起上传。若工具支持仓库级 AGENTS.md，应把它放在本组工作目录根部；普通对话则手动上传。

上传本组 AGENTS.md。
上传本指南和上一组的 HANDOFF_REPORT.md。
上传当前代码、配置和目录树，不要只描述文件名。
上传最近一次完整命令、完整首个报错及工作目录。
上传当前阶段证据：透传回归日志、覆盖测试、接口 schema、mock 服务输出和故障回退日志。
要求 AI 先判断当前阶段和缺失证据，本轮只推进一个验收门槛。
推荐开场消息
请先读取 01_环境组/AGENTS.md、本工作指南和上一组 HANDOFF_REPORT。 请根据实际文件判断我当前阶段，指出缺失证据；本轮只推进下一个验收门槛。 修改前先读代码，修改后运行最小测试，并列出本次新增或更新的交接文件。
