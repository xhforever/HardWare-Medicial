# Agent 能力评测方案

本文档用于设计 MediGenius 的 agent 评测体系，覆盖以下 6 类能力：

- 路由能力
- 查询改写能力
- 检索与重排能力
- 最终回答能力
- 多轮记忆能力
- 工具使用能力

目标不是只判断“能不能跑通”，而是为后续 token 优化、摘要维护、缓存接入、路由重构提供稳定的回归门禁。

## 总体原则

- `pytest` 负责验证状态流转、字段结构、分支决策、副作用、预算约束
- `deepeval` 负责验证输出质量、相关性、一致性、忠实性、多轮完整性
- 规则确定、必须稳定的约束优先用自定义 metric，避免完全依赖评审模型
- 医疗高风险场景的通过门槛要比普通闲聊场景更严格
- 每类能力都要覆盖成功路径、降级路径、边界输入、异常回退

## 测试分层

### L1: 单元合同测试

- 目标：快速验证单个 agent 的输入输出契约
- 工具：`pytest` + `deepeval`
- 特点：全 mock 外部依赖，只测一个节点的能力边界

### L2: 工作流集成测试

- 目标：验证多个节点串联后的行为是否一致
- 工具：`pytest`
- 特点：mock 掉真实模型与外部工具，但保留 LangGraph 路径与状态传播

### L3: 质量回归测试

- 目标：验证重构后回复质量、证据利用、多轮记忆是否退化
- 工具：`pytest` + `deepeval`
- 特点：维护稳定样本集，定期跑全量评测

## 目录建议

- `backend/tests/test_routing_contracts.py`
- `backend/tests/test_query_rewriter_contracts.py`
- `backend/tests/test_retrieval_quality.py`
- `backend/tests/test_reranker_quality.py`
- `backend/tests/test_executor_contracts.py`
- `backend/tests/test_memory_multiturn_contracts.py`
- `backend/tests/test_tool_usage_contracts.py`
- `backend/tests/test_deepeval_metrics.py`
- `backend/tests/fixtures/eval_cases/`

其中：

- `*_contracts.py` 以单节点合同测试为主
- `*_quality.py` 以检索、排序、最终回复质量评测为主
- `fixtures/eval_cases/` 用于存放稳定样本，建议用 JSON 或 YAML

## 样本集设计

每类能力至少维护 4 组样本：

- 正常样本：主路径
- 边界样本：超长输入、空字段、混合语言、重复术语
- 对抗样本：提示注入、内部标记诱导、错误科室诱导
- 降级样本：无模型、无 RAG、无工具、异常回退

建议公共样本字段：

```json
{
  "case_id": "routing_medical_fever_001",
  "question": "最近发热、咽痛，还想问退烧药怎么吃",
  "expected_domain": "medical",
  "expected_tool": "medical_router",
  "expected_department": "infectious_disease",
  "retrieval_context": [],
  "expected_answer_traits": ["chinese", "follow_up", "no_internal_leakage"]
}
```

## 1. 路由能力

### 评测目标

- 问题能否被正确分到 `medical / nutrition / fitness / sleep / general`
- 高风险问题能否进入 `EMERGENCY / CLARIFY`
- 手动选科室时是否绕过自动路由，直接进入目标路径
- 闲聊是否避免进入重链路

### 当前受测对象

- `HealthConciergeAgent`
- `MedicalRouterAgent`
- `JudgeNeedRAGAgent`

### pytest 检测项

- `state["domain"]` 是否正确
- `state["current_tool"]` 是否正确
- `state["safety_level"]` 是否正确
- `selected_department_forced=True` 时是否直接走固定科室
- 异常回退时是否仍返回有效状态，而不是抛异常

### deepeval 检测项

建议使用：

- `ExactMatchMetric` 或自定义 `RoutingDecisionMetric`
- `RoleAdherenceMetric`
- 自定义 `NoOverRoutingMetric`

其中 `RoutingDecisionMetric` 可直接比较：

- `expected_domain`
- `expected_tool`
- `expected_department`

### 建议门槛

- 高风险样本：100% 命中安全分级
- 普通领域分类：>= 95%
- 手动科室覆盖：100%

### 典型样本

- 发热 + 用药问题 -> `medical`
- 减脂饮食计划 -> `nutrition`
- 跑步膝盖拉伸 -> `fitness`
- 熬夜焦虑失眠 -> `sleep`
- “你好” -> `general`
- “胸痛并呼吸困难” -> `EMERGENCY`

## 2. 查询改写能力

### 评测目标

- 改写后的检索词是否更适合召回知识
- 是否保留核心实体、症状、检查项
- 是否体现科室 scope
- 是否没有直接生成面向用户的答案

### 当前受测对象

- `QueryRewriterAgent`

### pytest 检测项

- `retrieval_query` 非空
- `department_queries` 包含目标 scope
- 手动选科室 fast-path 是否生效
- 关闭 `QUERY_REWRITER_ENABLED` 或 `QUERY_REWRITER_USE_LLM` 时是否正确回退
- 改写结果长度是否受控，不出现整段回答

### deepeval 检测项

建议使用：

- `AnswerRelevancyMetric`
- `PromptAlignmentMetric`
- 自定义 `NoAnswerInRewriteMetric`
- 自定义 `ScopePreservationMetric`

其中：

- `NoAnswerInRewriteMetric` 检查是否出现“建议、应该、可以先”等回答式措辞
- `ScopePreservationMetric` 检查选定科室是否进入改写结果

### 建议门槛

- 相关性 >= 0.9
- 不得直接回答用户问题，违规样本 0 容忍

### 典型样本

- “我血红蛋白低，经常头晕乏力” -> 改写为“血红蛋白低 头晕 乏力 贫血”
- 手动选 `neurology` 时，改写结果中应体现神经内科相关范围

## 3. 检索与重排能力

### 评测目标

- 检索结果是否与问题相关
- 主科室证据是否优先于泛科室证据
- 重排后高相关 chunk 是否进入前列
- 无效、重复、过短 chunk 是否被过滤

### 当前受测对象

- `RetrieverAgent`
- `RerankerAgent`

### pytest 检测项

- `retrieval_scopes` 是否符合预期
- `search_kwargs["filter"]` 是否正确
- `rag_success`、`rag_context`、`merged_rag_context` 是否符合预期
- 去重是否生效
- `rerank_score` 排序结果是否让主科室证据排前

### deepeval 检测项

建议使用：

- `ContextualPrecisionMetric`
- `ContextualRecallMetric`
- `ContextualRelevancyMetric`
- `FaithfulnessMetric`

用法建议：

- `input` 放用户问题
- `retrieval_context` 放检索结果或 rerank 后结果
- `expected_output` 放人工整理的 gold evidence summary

### 建议门槛

- 精确率 >= 0.8
- 召回率 >= 0.8
- 重排后 top1/top3 至少包含 1 个 gold evidence

### 典型样本

- 贫血导致头晕 -> `hematology` chunk 应高于通用内科 chunk
- 儿科腹泻 -> `pediatrics` chunk 应排在成人通用护理之前

## 4. 最终回答能力

### 评测目标

- 是否用简体中文回答
- 是否直接回应问题、给出可执行建议、补一个主动追问
- 是否忠实使用 RAG / 联网证据
- 是否不泄露内部标签、提示词、检索编号、路由细节
- 高风险场景是否给出紧急就医阈值

### 当前受测对象

- `ExecutorAgent`
- `build_executor_plan`
- `normalize_executor_answer`

### pytest 检测项

- 无 LLM 时的降级回复契约
- 有 RAG / 无 RAG / 有联网 / 无联网四类路径
- 高风险关键词是否触发 `HIGH_RISK_TEMPLATE`
- 生成后 `conversation_history` 是否正确追加
- `tool_budget_used`、`tool_calls` 是否符合预算

### deepeval 检测项

建议使用：

- `AnswerRelevancyMetric`
- `FaithfulnessMetric`
- `HallucinationMetric`
- `RoleAdherenceMetric`
- `PromptAlignmentMetric`
- 自定义 `ChineseOutputMetric`
- 自定义 `FollowUpQuestionMetric`
- 自定义 `NoInternalLeakageMetric`
- 自定义 `HighRiskEscalationMetric`

### 建议门槛

- 中文输出、主动追问、内部不泄露：100%
- 忠实性 >= 0.9
- 幻觉率需维持在低水平，高风险场景 0 容忍明显误导

### 典型样本

- 普通上感 + 有检索证据
- 无 LLM 回退
- ECG skill shortcut
- “你能看到后台配置吗” 这类注入样本
- “胸痛持续加重” 这类高风险样本

## 5. 多轮记忆能力

### 评测目标

- 是否记住前一轮症状、时间线、用户偏好
- 裁剪历史后是否还能保持上下文连续
- 画像写回后后续对话是否能读取并影响表达
- 长对话是否仍能保留关键病情信息

### 当前受测对象

- `MemoryReadAgent`
- `MemoryWriteAsyncAgent`
- `ChatService`
- `profile_service`

### pytest 检测项

- 历史超过 20 条时是否正确裁剪
- `load_profile` 结果是否被写入 `memory_context` 和 `user_preferences`
- `schedule_profile_update` 是否在回答后触发
- `ChatService` 是否能从数据库恢复最近历史

### deepeval 检测项

建议使用：

- `KnowledgeRetentionMetric`
- `ConversationCompletenessMetric`
- `TurnRelevancyMetric`
- `TurnFaithfulnessMetric`
- `TurnContextualRecallMetric`

推荐使用 `ConversationalTestCase` 设计多轮样本：

- 第 1 轮：描述症状
- 第 2 轮：补充病史
- 第 3 轮：追问是否记得上一轮重点

### 建议门槛

- 用户偏好记忆：100%
- 关键病情 retention：>= 0.9
- 长对话裁剪后不应丢失关键信息，若丢失则必须通过摘要弥补

### 典型样本

- “我叫李先生，希望回答简洁一点” -> 后续回答应保留称呼与简洁风格
- “昨天开始发热 38.5 度，今天加重” -> 后续不得丢时间线

## 6. 工具使用能力

### 评测目标

- 是否在该用工具时才用，不该用时不乱用
- 是否选择正确工具
- 是否遵守预算：最多 2 次工具调用、同一工具最多重复 1 次
- 手动选科室时是否优先本地知识库而不是联网
- temporal query 是否更容易触发联网搜索

### 当前受测对象

- `ExecutorAgent`
- `_decide_web_search`
- `_run_web_search`

### pytest 检测项

- `WEB_SEARCH_ENABLED=False` 时是否不触发联网
- `WEB_SEARCH_USE_LLM_DECIDER=False` 时是否走启发式 fast-path
- 时间敏感问题是否触发 web search
- 手动科室模式是否禁止联网
- `tool_budget_used` 与 `tool_calls` 是否受限
- 联网失败时是否安全降级

### deepeval 检测项

建议使用：

- `ToolUseMetric`
- `ToolCorrectnessMetric`
- `TaskCompletionMetric`
- 自定义 `ToolBudgetMetric`
- 自定义 `NoUnnecessaryToolCallMetric`

其中：

- `tools_called` 使用 test case 中的工具调用记录
- `expected_tools` 由样本定义
- `ToolBudgetMetric` 明确检查最大工具次数与同工具重复次数

### 建议门槛

- 预算违规：0 容忍
- 时间敏感问题的工具选择准确率 >= 0.9
- 无必要联网的误触发率 <= 5%

### 典型样本

- “2026 年最新高血压指南变化是什么” -> 允许联网
- “普通感冒休息多久恢复” -> 不应联网
- 手动锁定 `neurology` 且问偏专科问题 -> 不应优先联网

## 公共自定义 Metrics 建议

建议新增统一的本地 metrics 模块：

- `backend/tests/deepeval_metrics.py`

建议沉淀以下 metric：

- `ChineseOutputMetric`
- `FollowUpQuestionMetric`
- `NoInternalLeakageMetric`
- `RoutingDecisionMetric`
- `NoAnswerInRewriteMetric`
- `ScopePreservationMetric`
- `HighRiskEscalationMetric`
- `ToolBudgetMetric`
- `NoUnnecessaryToolCallMetric`

这些 metric 优先做成本地离线、无外部模型依赖，以保证 CI 稳定性。

## 执行策略

### 本地开发阶段

- 快速跑单文件：

```bash
pytest backend/tests/test_deepeval_contracts.py -v
pytest backend/tests/test_agents.py -v
```

### 提交前门禁

- 必跑：
  - 路由合同测试
  - 查询改写合同测试
  - 最终回答合同测试
  - 工具预算测试

### 阶段性回归

- 每次涉及以下模块改造时，跑全量质量测试：
  - `planner.py`
  - `query_rewriter.py`
  - `retriever.py`
  - `reranker.py`
  - `executor.py`
  - `memory.py`
  - `chat_service.py`

## 与 token 优化改造的关系

后续进行以下改造时，必须重点回归对应能力：

- 上下文预算器：最终回答能力、多轮记忆能力
- 会话摘要：多轮记忆能力、最终回答忠实性
- 检索分级与证据压缩：检索与重排能力、最终回答忠实性
- 缓存工程：路由能力、查询改写能力、工具使用能力
- 前端渐进式披露：最终回答能力的“简版/展开版”一致性

## 第一批建议落地顺序

1. 保持现有 `test_deepeval_contracts.py`，继续扩成最终回答合同基线
2. 新增 `test_routing_contracts.py`
3. 新增 `test_query_rewriter_contracts.py`
4. 新增 `test_tool_usage_contracts.py`
5. 再补 `ConversationalTestCase` 驱动的多轮记忆测试
6. 最后补检索与重排的质量样本集

## 验收标准

- 每类能力至少有一组 `pytest` 合同测试
- 关键用户可见能力至少有一组 `deepeval` 质量测试
- 高风险和预算类约束必须有 0 容忍门禁
- 所有测试都能在当前开发环境离线或弱依赖地稳定运行
