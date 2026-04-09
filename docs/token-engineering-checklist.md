# Token 工程化改造清单

本清单用于后续 token 优化改造落地，目标是在不破坏现有医疗回复契约的前提下，降低上下文体积、减少重复调用，并补齐可观测性与回滚能力。

## 改造目标

- 降低长会话和 RAG 场景下的平均 prompt token 消耗
- 保持中文输出、主动追问、安全提示、内部实现不泄露等现有回复契约
- 将“硬编码截断”升级为“预算驱动 + 摘要维护 + 可观测 + 缓存”的工程化方案

## Phase 1: 可观测性基线

- [ ] 在 `backend/app/tools/llm_client.py` 增加统一调用封装，记录每次调用的模型、节点名、耗时、估算 prompt/completion token
- [ ] 在 `backend/app/services/chat_service.py` 和执行链路中补齐 node 级别的 token/latency 埋点
- [ ] 将关键指标落到日志或结构化 trace，至少包含 `rag_used`、`web_used`、`cache_hit`、`summary_used`
- [ ] 建立改造前基线：平均 prompt token、长会话 prompt token、RAG chunk 数、p95 延迟

## Phase 2: 上下文预算管理

- [ ] 新增 `backend/app/services/token_budget_service.py`
- [ ] 用统一预算器替代 `executor.py`、`memory.py`、`retriever.py` 中分散的字符级截断
- [ ] 预算按区块分配：system instructions、user question、recent history、rolling summary、profile memory、RAG evidence、web evidence、completion reserve
- [ ] 为高风险场景保留独立预算策略，避免安全信息被摘要或裁剪掉

## Phase 3: 摘要维护

- [ ] 在 `AgentState` 中增加 `conversation_summary`、`summary_updated_at` 等字段
- [ ] 新增 `backend/app/services/session_summary_service.py`
- [ ] 长会话达到阈值后，用轻量模型异步生成滚动摘要
- [ ] 最终 prompt 改为“最近原文 + 滚动摘要”，而不是单纯只保留最近若干条消息
- [ ] ECG 结构化摘要继续保留，但不再充当通用会话摘要

## Phase 4: RAG 分级检索与证据压缩

- [ ] 将固定 `k` 和固定 chunk 数改为预算驱动的分级检索
- [ ] 主科室先检索，分数不足时再扩展候选科室或通用知识库
- [ ] 对进入最终 prompt 的 RAG 片段做 token 级限额，而不是只按条数限额
- [ ] 在 `backend/app/agents/reranker.py` 中保留可解释的排序原因，便于后续调优

## Phase 5: 缓存工程

- [ ] 新增 `backend/app/services/cache_service.py`
- [ ] 先落地进程内 TTL cache，后续再按需要接 Redis
- [ ] 优先缓存低风险、高重复、确定性强的节点结果：`judge_need_rag`、`medical_router`、`query_rewriter`、检索结果、天气、联网搜索摘要
- [ ] 不做跨用户最终医疗回答缓存
- [ ] 记录 cache hit/miss 指标，避免“加了缓存但不知道是否有效”

## Phase 6: 前端渐进式披露

- [ ] 在 `frontend/src/App.jsx` 增加“简要回答 / 展开详情”模式
- [ ] 默认先请求简版回答，将机制解释、更多证据、扩展建议放到按需触发的二次请求
- [ ] 对 RAG 证据、联网资料、流程 trace 使用抽屉或折叠区，避免一次性进入主回答

## 配置与回滚

- [ ] 所有新能力都加开关：`TOKEN_BUDGET_ENABLED`、`SESSION_SUMMARY_ENABLED`、`CACHE_ENABLED`
- [ ] 每个阶段都支持单独回滚，不让多个改动捆绑上线
- [ ] 新逻辑默认灰度启用，先在开发和测试环境验证

## 测试门禁

- [x] 新增 `pytest + deepeval` 合同测试，覆盖中文输出、主动追问、内部标记不泄露等核心回复契约
- [ ] 将新合同测试纳入后续重构阶段的必跑清单
- [ ] 在摘要、预算器、缓存接入后补齐对应单元测试与集成测试
- [ ] 重构完成后重新采集 token 基线，确认优化收益与行为稳定性
