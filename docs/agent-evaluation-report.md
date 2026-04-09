# Agent 当前能力评估报告

评估更新时间：2026-03-22

本文档记录 MediGenius 当前 6 类 agent 能力的评估结论，以及每类能力在仓库中的具体评估方式。目标不是只说明“测过了”，而是把测试入口、样本设计、断言方式和当前局限固定下来，方便后续做 token 优化、摘要维护、缓存工程和工作流改造时作为回归门禁。

## 评估框架

- `pytest` 负责验证状态流转、分支选择、异常回退、副作用和预算约束
- `deepeval` 负责验证输出质量、字段契约、忠实性和多轮保持
- 当前优先使用本地自定义 metric，尽量避免强依赖外部 judge 模型
- 结论优先基于可重复、离线、稳定的合同测试，而不是单次在线主观体验

## 当前测试入口

- 路由能力：`backend/tests/test_routing_contracts.py`
- 查询改写能力：`backend/tests/test_query_rewriter_contracts.py`
- 检索与重排能力：`backend/tests/test_retrieval_quality.py`
- 最终回答能力：`backend/tests/test_executor_contracts.py`、`backend/tests/test_deepeval_contracts.py`
- 多轮记忆能力：`backend/tests/test_memory_multiturn_contracts.py`
- 工具使用能力：`backend/tests/test_tool_usage_contracts.py`
- 公共 metric：`backend/tests/deepeval_metrics.py`
- 查询改写 gold 数据集：`backend/tests/fixtures/eval_cases/query_rewriter_gold_cases.json`

## 1. 路由能力

结论：强

### 当前评估方式

- 受测对象：`HealthConciergeAgent`、`MedicalRouterAgent`、`JudgeNeedRAGAgent`
- 对应实现文件：
  - `backend/app/agents/planner.py`
  - `backend/app/agents/medical_router.py`
  - `backend/app/agents/judge_need_rag.py`
  - `backend/app/core/state.py`
- 主要测试文件：`backend/tests/test_routing_contracts.py`
- 对应评估文件：
  - `backend/tests/test_routing_contracts.py`
  - `backend/tests/deepeval_metrics.py`
- 样本类型：
  - 医疗问题进入 `medical_router`
  - 闲聊进入 `general/judge_need_rag`
  - 手动选科室时跳过自动路由
  - 高风险问题升级到 `EMERGENCY`
  - 血液科问题回退到合理主科室
- `pytest` 断言字段：
  - `domain`
  - `current_tool`
  - `safety_level`
  - `primary_department`
  - `need_rag`
  - `search_query`
- `deepeval` 方式：
  - 将关键状态字段序列化为 JSON
  - 使用本地 `JsonFieldsMatchMetric` 做精确字段匹配
- 当前特点：
  - 这是强合同测试，不依赖外部模型评分
  - 重点检验“走哪条分支”和“是否命中安全分流”

### 当前局限

- 目前以规则路径和代表性样本为主，还不是大样本分类准确率评测
- 复杂跨科室、多意图混合问题仍缺少更大的离线 gold 路由集

## 2. 查询改写能力

结论：良好

### 当前评估方式

- 受测对象：`QueryRewriterAgent`
- 对应实现文件：
  - `backend/app/agents/query_rewriter.py`
  - `backend/app/core/medical_taxonomy.py`
  - `backend/app/core/state.py`
- 主要测试文件：`backend/tests/test_query_rewriter_contracts.py`
- 对应评估文件：
  - `backend/tests/test_query_rewriter_contracts.py`
  - `backend/tests/deepeval_metrics.py`
  - `backend/tests/fixtures/eval_cases/query_rewriter_gold_cases.json`
- 数据集：
  - 已接入 30 条本地 gold 样本
  - 来源于当前真实有内容的 3 个知识域：`general_medical`、`pediatrics`、`infectious_disease`
  - `user_query` 分布为 `20` 条纯中文、`10` 条中英混合
- 样本字段：
  - `user_query`
  - `expected_scope`
  - `rewrite_mode`
  - `gold_retrieval_queries`
  - `must_keep_terms`
  - `required_department_query_terms`
  - `expected_rewrite_reason`
- 测试设计：
  - 单独保留 1 组 mocked LLM 合同测试，验证 LLM 改写输出结构
  - 单独保留 fast-path、禁用 query rewriter、禁用 LLM 的回退测试
  - gold 数据集评估时强制走离线确定性路径：`QUERY_REWRITER_USE_LLM=False`
- `pytest` 断言字段：
  - `retrieval_query`
  - `department_queries`
  - `rewrite_reason`
- `deepeval` 方式：
  - `ContainsTermsMetric`：检查核心症状、实体、检查项是否保留
  - `GoldQueryAlignmentMetric`：按词项覆盖率对比 gold 检索 query，而不是做脆弱的整句精确匹配
  - `NoAnswerStyleMetric`：确保改写结果保持检索短语风格，不直接回答用户
  - `JsonFieldContainsTermsMetric`：检查目标 scope 的 `department_queries`
  - `JsonFieldsMatchMetric`：检查 `rewrite_reason`
- 当前特点：
  - 评估已经从“结构契约”升级为“知识库驱动的 gold query 对齐”
  - 该评估完全离线，无需 judge 模型

### 当前局限

- 当前 gold 对齐仍是词项覆盖，不是向量语义等价评测
- 30 条样本已能做稳定回归，但仍不足以覆盖大量口语化、省略式、错别字式查询

## 3. 检索与重排能力

结论：良好

### 当前评估方式

- 受测对象：`RetrieverAgent`、`RerankerAgent`
- 对应实现文件：
  - `backend/app/agents/retriever.py`
  - `backend/app/agents/reranker.py`
  - `backend/app/tools/vector_store.py`
  - `backend/app/core/state.py`
- 主要测试文件：`backend/tests/test_retrieval_quality.py`
- 对应评估文件：
  - `backend/tests/test_retrieval_quality.py`
  - `backend/tests/deepeval_metrics.py`
- 样本设计：
  - 使用 mock retriever 构造可控文档集
  - 构造重复文档，验证去重
  - 构造主科室与通用科室证据混合样本，验证重排提升主科室证据
- `pytest` 断言字段：
  - `documents`
  - `rag_context`
  - `merged_rag_context`
  - `rerank_score`
- `deepeval` 方式：
  - `JsonListTopFieldMetric`：检查 top1 结果是否来自预期 scope
  - `JsonListContainsTermsMetric`：检查检索内容是否含有问题核心术语
- 当前特点：
  - 重点验证召回结果结构是否正确、是否去重、top 排序是否合理
  - 测试是稳定可重复的，不依赖真实在线向量检索环境

### 当前局限

- 当前主要是合成文档和合同测试，不是大规模真实知识库 precision/recall 评测
- 暂未接入基于真实知识库 chunk 的 top-k 命中率统计

## 4. 最终回答能力

结论：强

### 当前评估方式

- 受测对象：`ExecutorAgent`、欢迎语生成逻辑、紧急场景 shortcut
- 对应实现文件：
  - `backend/app/agents/executor.py`
  - `backend/app/services/greeting_service.py`
  - `backend/app/core/state.py`
- 主要测试文件：
  - `backend/tests/test_executor_contracts.py`
  - `backend/tests/test_deepeval_contracts.py`
- 对应评估文件：
  - `backend/tests/test_executor_contracts.py`
  - `backend/tests/test_deepeval_contracts.py`
  - `backend/tests/deepeval_metrics.py`
- 样本类型：
  - 有 RAG 上下文时的标准回答
  - 无 LLM 时的降级回答
  - 空会话欢迎语
  - `EMERGENCY` 高风险问题
- `pytest` 断言字段：
  - `generation`
  - `source`
  - 降级路径是否返回有效回复
- `deepeval` 方式：
  - `ChineseOutputMetric`：检查回答保持简体中文
  - `FollowUpQuestionMetric`：检查回答末尾有主动追问
  - `NoInternalLeakageMetric`：检查不泄露内部标签、提示词、RAG/WEB 标记
  - `HighRiskEscalationMetric`：检查高风险问题是否给出就医升级语言
- 当前特点：
  - 评估重点是“用户可见契约”而不是主观文风喜好
  - 对安全和泄露约束采用本地 0 容忍 metric

### 当前局限

- 当前重点还是输出契约和安全性，不是大规模医学正确性 benchmark
- 对真实在线模型的忠实性、幻觉率和证据引用质量仍需后续扩充

## 5. 多轮记忆能力

结论：基础可用

### 当前评估方式

- 受测对象：`MemoryReadAgent`、`MemoryWriteAsyncAgent`、`ChatService`
- 对应实现文件：
  - `backend/app/agents/memory.py`
  - `backend/app/services/chat_service.py`
  - `backend/app/services/profile_service.py`
  - `backend/app/core/state.py`
- 主要测试文件：`backend/tests/test_memory_multiturn_contracts.py`
- 对应评估文件：
  - `backend/tests/test_memory_multiturn_contracts.py`
  - `backend/tests/deepeval_metrics.py`
- 样本设计：
  - 构造超过 20 条历史，验证裁剪
  - mock 用户画像，验证 `memory_context` 和 `user_preferences` 写入
  - 验证回答结束后会调度画像异步写回
  - mock 持久化聊天记录，验证多轮历史恢复
- `pytest` 断言字段：
  - `conversation_history`
  - `memory_context`
  - `user_preferences`
  - `schedule_profile_update` 调用参数
- `deepeval` 方式：
  - `JsonFieldsMatchMetric`：验证裁剪长度和偏好字段
  - `JsonFieldContainsTermsMetric`：验证长期记忆内容是否被读取
  - `HistoryContainsMetric`：验证恢复历史中仍保留关键上下文
- 当前特点：
  - 当前评估覆盖的是“记忆读写链路”和“历史恢复契约”
  - 重点在状态存在性和内容保留，而不是复杂多轮推理

### 当前局限

- 目前还没有真正的 `ConversationalTestCase` 式多轮语义一致性评测
- 长对话下的摘要保持、旧信息遗失补偿、风格持续一致性仍未系统评测

## 6. 工具使用能力

结论：强

### 当前评估方式

- 受测对象：`ExecutorAgent`、`_run_web_search`
- 对应实现文件：
  - `backend/app/agents/executor.py`
  - `backend/app/tools/tavily_search.py`
  - `backend/app/tools/duckduckgo_search.py`
  - `backend/app/core/state.py`
- 主要测试文件：`backend/tests/test_tool_usage_contracts.py`
- 对应评估文件：
  - `backend/tests/test_tool_usage_contracts.py`
  - `backend/tests/deepeval_metrics.py`
- 样本类型：
  - 时间敏感问题触发联网搜索
  - 手动锁定科室时抑制联网
  - 同一工具重复调用达到上限时被拦截
- `pytest` 断言字段：
  - `tool_calls`
  - `tool_budget_used`
  - `source`
  - web search 是否实际被调用
- `deepeval` 方式：
  - `ToolBudgetMetric`：检查总工具次数和同工具重复次数
  - `JsonFieldsMatchMetric`：检查 source 或预算字段是否符合预期
- 当前特点：
  - 工具使用评估是强约束测试，重点是预算、是否误用、是否越界
  - 特别适合给 token 优化和工具决策改造做回归门禁

### 当前局限

- 当前还没有“大样本该联网/不该联网”准确率统计
- 对复杂工具选择错误类型的细分还不够，比如“该联网没联网”和“错用本地知识库”尚未分层统计

## 当前综合判断

当前系统在工程合同层面已经具备较好的稳定性，尤其是以下几类能力已经具备明确、可重复的回归方式：

- 路由分支决策
- 查询改写输出契约与 gold 对齐
- 最终回答安全约束
- 工具预算控制

当前最需要继续加强的部分仍然是：

- 基于真实知识库的大样本检索评估
- 多轮记忆的摘要级、多轮语义级评测
- 最终回答的医学正确性、忠实性和幻觉率评估

## 后续建议

1. 继续扩充查询改写 gold 数据集，覆盖错别字、省略式表达和跨科室问题。
2. 为检索与重排增加真实知识库 chunk 级别的 top-k 命中评测。
3. 为多轮记忆增加 `ConversationalTestCase`，覆盖时间线记忆和用户偏好延续。
4. 在保持离线 metric 稳定的前提下，逐步引入可选在线 judge 评测，补语义层质量判断。
