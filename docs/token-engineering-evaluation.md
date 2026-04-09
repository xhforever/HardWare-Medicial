# Token 工程化改造评估

评估时间：2026-03-21

本次评估基于本地代码改造后的 `pytest` 回归和 `tiktoken` 估算结果生成，重点看两类收益：

- 长上下文场景下的最终回答 prompt 压缩收益
- 重复请求场景下的轻量节点缓存收益

## 当前评估模型

- 主模型：`gpt-5.4`
- 轻量模型：`gpt-5.4`

说明：

- 上述模型来自本地运行时配置读取结果，不是文档默认值推断
- token 对比采用本地 `tiktoken` 估算，属于工程侧 prompt 体积基线，不等同于线上供应商账单

## 已落地能力

- 统一 LLM 调用埋点：记录节点名、模型、耗时、估算 prompt/completion token
- Executor 上下文预算器：按 `profile / summary / recent_history / rag / web / completion reserve` 分配预算
- 长会话滚动摘要：在历史超过硬上限后维护 `conversation_summary`
- RAG 证据压缩：进入最终 prompt 的证据改为 token 级限额
- 进程内 TTL 缓存：接入 `judge_need_rag`、`medical_router`、`query_rewriter`、`retriever`
- 结构化 trace：补齐 `rag_used`、`web_used`、`cache_hit`、`summary_used`、`prompt_tokens`、`latency_ms`

## 基准结果

### 1. 长上下文最终回答 prompt

- 改造前：`10213` tokens
- 改造后：`1858` tokens
- 降幅：`81.81%`

测试场景：

- 28 条长多轮历史
- 8 个长 RAG chunk
- 带长期画像
- 不触发联网搜索

结论：

- 本次收益主要来自 `RAG evidence` 和 `recent history` 的预算压缩
- 滚动摘要补回了被裁掉的旧信息，但总体 prompt 仍显著下降

### 2. 重复路由/改写链路

- 首次请求：`343` prompt tokens
- 缓存命中后重复请求：`0` prompt tokens
- 降幅：`100%`

测试链路：

- `medical_router`
- `query_rewriter`

结论：

- 对完全重复或高重复问题，缓存可以直接消除重复的轻量模型 prompt 消耗
- 这部分收益更偏“重复流量优化”，不是单次请求压缩

## 回归结果

- 新增 token 工程测试：`4 passed`
- 后端全量测试：`133 passed`
- 现有 warning：`10` 条，仍为 SQLAlchemy `datetime.utcnow()` 弃用告警

## 风险与边界

- 目前的收益评估主要针对 prompt token，不包含 completion token 和第三方工具计费
- `SESSION_SUMMARY_USE_LLM` 默认关闭，当前摘要默认走启发式压缩；开启后可进一步提高摘要质量，但会引入额外轻量模型成本
- 检索阶段仍是固定 `k + rerank`，只是在进入 Executor prompt 前做了 token 级压缩；下一阶段仍可继续做预算驱动检索

## 建议下一步

1. 给 `SESSION_SUMMARY_USE_LLM` 做小流量灰度，对比摘要质量和新增成本
2. 继续把 Retriever 的召回阶段从固定 `k` 改成预算驱动
3. 在真实线上流量上补充 `平均 prompt tokens / p95 latency / cache hit rate` 的持续观测
