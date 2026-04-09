# 医枢智疗项目存储与数据处理分析

## 1. 文档目的

本文档从“存储与数据处理”视角系统梳理医枢智疗项目的实现方案，重点回答以下问题：

- 项目里有哪些数据类型，分别存在哪里
- 不同模块如何完成数据采集、清洗、持久化、检索和回写
- LangGraph 工作流如何驱动数据在不同存储之间流转
- JSON 画像存储为什么要使用“锁 + 原子写入”
- 当前设计的优势、边界和后续可演进方向是什么

本文档主要对应以下实现文件：

- `backend/app/api/v1/request_context.py`
- `backend/app/core/state.py`
- `backend/app/services/chat_service.py`
- `backend/app/services/database_service.py`
- `backend/app/services/profile_service.py`
- `backend/app/tools/vector_store.py`
- `backend/app/agents/query_rewriter.py`
- `backend/app/agents/retriever.py`
- `backend/app/agents/reranker.py`
- `backend/app/services/ecg_monitor_service.py`
- `backend/app/services/ecg_report_service.py`
- `backend/app/services/ecg_pdf_service.py`
- `hardware/fetch_latest_ecg_and_convert.py`
- `backend/app/services/flow_trace_service.py`

---

## 2. 总体架构概览

这个项目不是单一数据库应用，而是一个典型的多存储协同系统。项目把不同生命周期、不同结构的数据拆分到了不同介质：

| 数据类型 | 存储介质 | 主要文件/模块 | 用途 |
| --- | --- | --- | --- |
| 会话消息、ECG 报告 | SQLite | `database_service.py` | 持久化结构化业务数据 |
| 用户长期画像 | JSON 文件 | `profile_service.py` | 维护偏好、基础信息、上下文记忆 |
| 医疗知识向量索引 | ChromaDB | `vector_store.py` | 支持 RAG 检索 |
| ECG PDF、流程日志 | 文件系统 | `ecg_pdf_service.py`、`flow_trace_service.py` | 产出文件与可观测性记录 |
| 当前对话中间状态 | 内存 + TypedDict | `state.py`、`chat_service.py` | LangGraph 节点之间共享上下文 |
| ECG 原始采集数据 | XLS + 内存解析结果 | `fetch_latest_ecg_and_convert.py` | 硬件侧原始信号处理 |

项目的核心设计思想是：

1. 用 SQLite 管理稳定、结构明确、需要查询的业务数据。
2. 用 JSON 存储灵活变化、低频更新、以用户为中心的长期记忆。
3. 用 ChromaDB 持久化语义知识索引，承接医疗问答 RAG。
4. 用文件系统存放最终产物和追踪日志，避免挤占业务数据库。
5. 用内存状态承接 LangGraph 每轮推理过程中的临时数据。

---

## 3. 数据入口与身份隔离

### 3.1 请求上下文解析

入口在 `backend/app/api/v1/request_context.py`。

项目所有核心存储操作都依赖三个身份维度：

- `tenant_id`
- `user_id`
- `session_id`

`get_request_context()` 的处理顺序是：

1. 优先读取请求头 `X-Tenant-ID`、`X-User-ID`、`X-Session-ID`
2. 如果没有，再从 query 参数中读取
3. 如果还没有，再从 cookie session 中读取
4. 如果 `session_id` 仍然为空，则自动生成 UUID
5. 使用 `_sanitize_id()` 对标识符做清洗，过滤非法字符
6. 将解析后的三元组再写回 `request.session`

### 3.2 为什么要统一做身份解析

这是项目所有存储设计的根主键。后续的：

- SQLite 消息读写
- ECG 报告查询
- JSON 画像路径映射
- 内存会话态缓存
- ECG 任务状态访问控制

都依赖这套身份体系完成隔离。

### 3.3 数据隔离策略

项目并不是简单按 `session_id` 隔离，而是采用分层策略：

- 会话历史：按 `tenant_id + user_id + session_id`
- 用户画像：按 `tenant_id + user_id`
- ECG 任务状态：按 `tenant_id + user_id + session_id`

这意味着：

- 同一个用户不同会话可以共享长期记忆
- 不同用户即使 session_id 相同也不会互相看到数据
- 多租户场景下可以安全复用同一套服务逻辑

---

## 4. 结构化业务存储：SQLite

### 4.1 模块职责

结构化业务数据主要通过 `backend/app/services/database_service.py` 管理。

当前 SQLite 主要存两类核心数据：

- 聊天消息 `Message`
- ECG 报告 `ECGReport`

### 4.2 主要实现方法

#### 4.2.1 ORM 封装

项目使用 SQLAlchemy ORM，通过：

- `app.models.message`
- `app.models.ecg_report`

定义数据模型。

`DatabaseService` 对外暴露统一接口：

- `save_message()`
- `get_chat_history()`
- `get_all_sessions()`
- `delete_session()`
- `save_ecg_report()`
- `get_ecg_report()`

#### 4.2.2 会话历史写入

聊天消息写入时会持久化：

- `tenant_id`
- `user_id`
- `session_id`
- `role`
- `content`
- `source`

这让后端能够在任意时刻恢复当前会话的最近对话上下文。

#### 4.2.3 ECG 报告写入

ECG 报告保存时会落库：

- 风险等级 `risk_level`
- 报告正文 `report`
- 关键发现 `key_findings`
- 建议 `recommendations`
- 免责声明 `disclaimer`
- 原始输入 `raw_request`

其中 `key_findings`、`recommendations`、`raw_request` 被序列化为 JSON 字符串存入数据库。

### 4.3 轻量迁移兼容方案

项目没有引入 Alembic 等完整迁移框架，而是在 `init_db()` 中调用 `_ensure_identity_columns()`：

- 通过 `PRAGMA table_info(...)` 检查列是否存在
- 对旧表执行 `ALTER TABLE`
- 将历史空值行补齐为默认租户和匿名用户

这是一种适合原型和单机部署场景的轻量兼容升级方案。

### 4.4 为什么适合 SQLite

这个项目的会话和报告数据具备以下特点：

- 单条记录结构明确
- 查询模式简单
- 部署环境偏本地/单机
- 不需要高并发分布式事务

因此 SQLite 足够稳定、实现简单、维护成本低。

---

## 5. 用户长期画像存储：JSON 文件

### 5.1 模块职责

用户画像由 `backend/app/services/profile_service.py` 管理，用于保存：

- 基础信息：年龄、性别、身高、体重
- 偏好信息：偏好称呼、语言、沟通风格、详细程度
- 当前上下文：症状、用药、最近检查、最近 ECG 总结

### 5.2 为什么不用数据库而用 JSON

画像数据有几个典型特征：

- 字段会逐步扩展
- 更新频率不高
- 查询基本都是整份读取
- 需要更灵活的 schema 演进

因此用 JSON 文件比强行建表更轻量、更灵活。

### 5.3 文件命名与隔离方式

画像路径由 `_profile_path()` 生成：

- 文件名格式：`{tenant_id}__{user_id}.json`
- 明确忽略 `session_id`

这意味着项目把画像定义为“用户级长期记忆”，而不是“会话级短期记忆”。

### 5.4 Schema 约束与数据清洗

`PROFILE_SCHEMA` 定义了允许写入的字段集合。

写入前会调用 `_normalize_profile_updates()`：

1. 只保留 schema 中定义的字段
2. 根据规则做类型纠正
3. 无效值和未知字段直接丢弃

例如：

- `"29"` 会转成整数 `29`
- `"MALE"` 会转成合法枚举 `male`
- 空字符串不会覆盖已有值

### 5.5 画像写回来源

画像更新主要有两条路径：

#### 5.5.1 ECG 流程显式写入

`ECGMonitorService.start_monitor()` 在任务开始前就把前端填写的基础信息写入画像，例如：

- 年龄
- 性别
- 身高/体重
- 患者称呼

这样后续聊天可以直接复用这些信息。

#### 5.5.2 对话后异步抽取写入

在 `MemoryWriteAsyncAgent` 中，会调用 `schedule_profile_update()`：

1. 后台线程启动
2. 使用轻量模型从“用户问题 + 助手回答”中提取长期记忆
3. 校验并规范化提取结果
4. 调用 `update_profile()` 合并回 JSON 文件

这是一种“LLM 抽取 + Schema 约束 + 文件存储”的混合记忆方案。

---

## 6. 并发控制：锁 + 原子写入

### 6.1 设计目标

JSON 文件不是数据库，没有事务、没有行锁、没有 WAL。  
因此项目必须自行处理两个问题：

- 多线程同时更新同一份画像时，如何避免丢更新
- 写文件中途失败或被读取时，如何避免文件损坏

### 6.2 锁解决什么问题

全局锁定义为：

```python
_profile_lock = threading.Lock()
```

使用位置在 `update_profile()`：

```python
with _profile_lock:
    profile = load_profile(...)
    ...
    _atomic_save_profile(...)
```

锁保护的是整个“读 -> 合并 -> 写回”过程，而不是单独的“写文件”。

如果没有这把锁，可能出现这种并发覆盖：

1. 线程 A 读取旧画像
2. 线程 B 也读取同一个旧画像
3. A 写入自己的更新
4. B 再把自己基于旧数据生成的新结果写回
5. A 的更新被 B 覆盖

这就是典型的丢更新问题。

使用锁后，同一时刻只允许一个线程执行完整更新流程：

1. 线程 A 拿锁，读旧值、合并、写回
2. A 释放锁
3. 线程 B 再读取 A 更新后的新值继续合并

这样才能保证更新语义是“累积合并”，而不是“最后写入者覆盖”。

### 6.3 原子写入解决什么问题

`_atomic_save_profile()` 的实现是：

```python
path = _profile_path(...)
temp_path = f"{path}.tmp"
with open(temp_path, "w", encoding="utf-8") as f:
    json.dump(profile, f, ensure_ascii=True, indent=2)
os.replace(temp_path, path)
```

它采用的是：

1. 先把完整 JSON 写到临时文件
2. 再用 `os.replace()` 一次性替换正式文件

这个设计可以避免两类问题。

#### 6.3.1 防止读到半截文件

如果直接对正式文件 `open(path, "w")`：

- 旧文件会先被截断
- 新内容还没写完时，另一个读线程可能读到空文件或半截 JSON

而当前实现中，正式文件在 `os.replace()` 前始终保持旧版本，读者要么读到完整旧文件，要么读到完整新文件。

#### 6.3.2 防止中途崩溃导致文件损坏

如果程序在写正式文件时崩溃，正式文件可能只剩半截内容。  
而现在如果进程在 `os.replace()` 之前退出：

- 正式文件仍然是旧版本
- 最多只留下一个 `.tmp` 文件

### 6.4 为什么两者缺一不可

只有锁，没有原子写入：

- 可以防止并发覆盖
- 但不能防止写到一半时文件损坏
- 也不能防止读者读到半截文件

只有原子写入，没有锁：

- 可以保证文件始终是完整版本
- 但两个线程仍可能基于同一个旧画像分别生成新版本
- 最终还是可能出现最后一次覆盖前一次更新

因此：

- 锁负责“更新逻辑正确”
- 原子替换负责“落盘状态正确”

### 6.5 当前方案的边界

这套设计只保证单进程内线程安全，因为 `threading.Lock` 只能约束当前 Python 进程。

如果未来部署为：

- 多个 gunicorn worker
- 多进程任务消费者
- 多容器实例

就需要进一步引入：

- 文件锁
- 数据库画像表
- 或分布式存储方案

---

## 7. 当前对话状态管理：内存 + TypedDict

### 7.1 模块职责

LangGraph 运行时的临时状态由 `backend/app/core/state.py` 中的 `AgentState` 表达。

它管理的不是长期数据，而是本轮请求的中间态，例如：

- 当前问题
- 检索 query
- 科室候选
- RAG 上下文
- 工具调用次数
- 安全等级
- 当前节点 flow trace
- 最终答案

### 7.2 实现方法

在 `ChatService` 中，项目使用：

- `conversation_states: Dict[str, Dict]`
- `threading.Lock()`

按 `tenant::user::session` 组织内存会话态。

每次处理请求时：

1. 如果会话状态不存在，先初始化
2. 从 SQLite 恢复最近持久化历史
3. 用 `reset_query_state()` 重置本轮字段
4. 把当前消息、身份信息、科室选择写入 state

### 7.3 为什么要内存态

如果每个 LangGraph 节点都反复读数据库：

- 性能差
- 逻辑分散
- 节点之间状态传递复杂

使用 `AgentState` 后，所有节点共享同一个中间状态对象，工作流表达更清晰。

---

## 8. 向量存储与语义检索：ChromaDB

### 8.1 模块职责

语义知识库由 `backend/app/tools/vector_store.py` 管理，核心职责是：

- 初始化 embedding 模型
- 加载或创建 Chroma 向量库
- 对外提供 retriever

### 8.2 实现方法

#### 8.2.1 Embedding 懒加载

`get_embeddings()` 使用 `HuggingFaceEmbeddings`，并缓存到模块级单例：

- 避免每次请求都重新加载模型
- 降低 CPU 和内存开销

#### 8.2.2 向量库持久化

向量库存储在本地目录中，项目通过：

- 判断目录内是否已有 `sqlite3/index` 文件
- 决定加载已有库还是用文档重建

#### 8.2.3 元数据感知重建

项目特别处理了历史向量库不带 `department` 元数据的情况：

1. 如果新文档已经包含科室标签
2. 但旧 collection 中没有 `department`
3. 则自动删库并重建

这保证了后续的科室级过滤检索能够正常工作。

### 8.3 检索调用方式

`get_retriever()` 返回的 retriever 支持带过滤参数的检索，例如：

- 医疗场景：`{"department": "neurology"}`
- 非医疗领域：`{"domain": "sleep"}`

这使得项目并不是做全库召回，而是做“带标签约束的局部召回”。

---

## 9. 查询改写、召回与重排

### 9.1 Query Rewriter

模块：`backend/app/agents/query_rewriter.py`

实现方法：

1. 先从原问题里抽关键词，生成兜底检索 query
2. 如果用户手动锁定科室，则直接走 fast-path
3. 如果启用 LLM 改写，则让轻量模型输出：
   - `retrieval_query`
   - `department_queries`
   - `rewrite_reason`
4. 如果 LLM 不可用，则回退到关键词规则生成

这是一种典型的“规则兜底 + LLM 优化”方案。

### 9.2 Retriever

模块：`backend/app/agents/retriever.py`

实现方法：

1. 先根据 `domain / selected_department / primary_department / department_candidates` 决定检索 scope
2. 每个 scope 会尝试多种 query：
   - `department_queries[scope]`
   - `retrieval_query`
   - `search_query`
   - 原问题
3. 针对每个 scope 构建过滤条件
4. 调用 retriever 检索
5. 对结果做：
   - 最小长度过滤
   - 按内容/来源/页码去重
   - 统一封装为 chunk

最终产出：

- `documents`
- `merged_rag_context`
- `retrieval_results_by_scope`
- `rag_context`

### 9.3 Reranker

模块：`backend/app/agents/reranker.py`

项目没有使用大型重排模型，而是实现了轻量规则重排：

- 问题关键词覆盖度
- 检索 query 关键词覆盖度
- scope 优先级
- 主科室加权
- 原始召回 rank 加权

这样既保持了一定相关性，又控制了推理延迟和工程复杂度。

---

## 10. 聊天主流程的数据流

从数据处理角度，一次完整问答的流程如下：

1. 前端发起请求
2. `RequestContext` 解析身份和会话信息
3. `ChatService` 先把用户消息写入 SQLite
4. `MemoryReadAgent` 加载 JSON 画像并格式化成 prompt 文本
5. `HealthConciergeAgent` 进行安全分诊与领域识别
6. `MedicalRouterAgent` 决定主科室和候选科室
7. `QueryRewriterAgent` 生成更适合检索的 query
8. `RetrieverAgent` 从 ChromaDB 做多 scope 检索
9. `RerankerAgent` 对召回结果排序
10. `ExecutorAgent` 结合：
    - 用户问题
    - 最近对话
    - 长期画像
    - RAG 结果
    - 必要时联网搜索结果
    生成回答
11. 助手消息再写入 SQLite
12. `MemoryWriteAsyncAgent` 后台抽取长期记忆并回写 JSON
13. `flow_trace_service` 记录流程轨迹

这是一条“原始事件持久化 -> 中间态推理 -> 高价值信息沉淀”的完整数据链。

---

## 11. ECG 数据处理闭环

这是项目最重的数据处理模块，涉及远程抓取、原始文件解析、特征工程、报告生成和 PDF 落地。

### 11.1 ECG 任务管理

模块：`backend/app/services/ecg_monitor_service.py`

实现方法：

1. 前端提交基础信息
2. 后端创建 `task_id`
3. 把任务状态保存在内存 `_tasks` 字典
4. 用后台线程启动 worker
5. 用 `Condition` 变量通知状态更新

这样前端可以通过：

- 轮询接口
- SSE 事件流

持续获取任务状态。

### 11.2 动态加载硬件脚本

ECG 采集逻辑不直接写在主服务中，而是通过 `_load_hardware_fetch_module()` 动态加载：

- `hardware/fetch_latest_ecg_and_convert.py`

这样做的意义：

- 主服务与抓取脚本解耦
- 硬件侧变更不影响主业务初始化
- 可以单独替换或调试抓取逻辑

### 11.3 原始数据获取

硬件脚本完成以下流程：

1. 登录远程心电平台
2. 查询最新一条 ECG 记录
3. 下载对应 `XLS`
4. 解析导联数据和诊断信息

主要依赖：

- `requests`
- `xlrd`

### 11.4 信号处理与特征提取

脚本中使用：

- `numpy`
- `scipy.signal.find_peaks`

实现以下处理逻辑：

#### 11.4.1 基线与平滑

通过 `_moving_average()` 做滑动平均，辅助去基线漂移和噪声平滑。

#### 11.4.2 心率估计

`_estimate_heart_rate()` 的主要步骤：

1. 去除中位数基线
2. 计算慢变化基线
3. 构造包络
4. 通过 `find_peaks` 检测心搏峰值
5. 计算 RR 间期
6. 用 RR 中位数估算心率

#### 11.4.3 信号质量评估

`_quality_metrics()` 通过：

- 高频噪声占比
- 基线漂移占比

计算质量分数，并映射为：

- 质量良好
- 存在噪声干扰
- 信号质量较差

#### 11.4.4 诊断解析

`_parse_diagnosis()` 把平台返回的诊断文本切分成：

- 诊断编码 `diagnosis_codes`
- 中文诊断 `diagnosis_cn`

同时去重并映射常见 SCP 编码。

### 11.5 标准化请求载荷

在 `ECGMonitorService._worker()` 中，远程抓到的数据会被整理为统一的 `skill_payload`：

- `patient_info`
- `diagnosis_codes`
- `diagnosis_cn`
- `signal_quality`
- `features`
- `waveform`
- `notes`

这个标准载荷随后会被转换成 `ECGReportRequest`，用于后续报告生成。

这是一种典型的“原始异构数据 -> 标准中间层”的处理方法。

---

## 12. ECG 报告生成与 PDF 存储

### 12.1 报告生成

模块：`backend/app/services/ecg_report_service.py`

实现方法可以分为五层：

#### 12.1.1 风险分层

`_infer_risk_level()` 根据：

- 高风险诊断码
- 心率阈值
- 信号质量

判断 `high / medium / low`。

#### 12.1.2 关键信息提取

`_extract_key_findings()` 提取：

- 诊断结论
- 心率
- 心电轴
- 信号质量

#### 12.1.3 建议生成

`_build_recommendations()` 按风险等级选择模板化建议。

#### 12.1.4 LLM 报告生成

项目构造结构化 Prompt，要求模型输出四段：

- 临床信息
- 心电图所见
- 诊断结论
- 建议

同时禁止输出无关签名栏和占位符。

#### 12.1.5 降级回退

如果 LLM 调用失败，则用 `_fallback_report()` 输出模板化报告，避免整个链路中断。

### 12.2 ECG 报告的持久化

报告生成后会：

1. 把报告文本和原始请求落到 SQLite
2. 把摘要结果回写到 JSON 画像
3. 尝试生成 PDF 文件

### 12.3 PDF 生成

模块：`backend/app/services/ecg_pdf_service.py`

主要依赖：

- `matplotlib`
- `reportlab`

实现方法：

1. 从 `waveform` 中挑选 Lead II
2. 用 matplotlib 将波形绘制为 PNG
3. 用 reportlab 生成 PDF
4. 写入患者信息、报告文本、关键发现、建议和免责声明
5. 落盘到 `ECG_REPORT_PDF_DIR`

这种方案适合医疗报告这种“图形 + 文本”的混合输出场景。

---

## 13. SSE 与任务状态数据

### 13.1 聊天流式返回

`backend/app/api/v1/endpoints/chat.py` 中的 `/chat/stream` 使用 `StreamingResponse` 输出 SSE 帧：

- `start`
- `delta`
- `done`
- `error`

数据流上，`ChatService.process_message_stream()` 会逐步把 LLM token 输出给前端。

### 13.2 ECG 状态推送

`backend/app/api/v1/endpoints/ecg.py` 中的 `/monitor/{task_id}/events` 也是基于 SSE。

服务端通过：

- `Condition.wait_for()`
- 状态更新时间 `updated_at`

等待内存任务表 `_tasks` 的变化，并把最新任务状态推送给前端。

这是一个“内存状态 + 条件变量 + SSE 推送”的轻量任务通知机制。

---

## 14. 可观测性与链路追踪

模块：`backend/app/services/flow_trace_service.py`

### 14.1 存储内容

每次对话后，项目会把以下信息持久化到：

- `docs/flow-trace-record.jsonl`
- `docs/flow-trace-record.md`

内容包括：

- 时间戳
- session_id
- 用户问题
- flow_trace
- 最终 source
- 说明字段 notes

### 14.2 实现意义

这相当于一个轻量版链路追踪系统，可以回答：

- 这次请求经过了哪些 LangGraph 节点
- 是否走了安全分诊
- 是否触发了 RAG
- 最终答案来自哪个知识来源

对多 Agent 系统的调试和问题定位非常有帮助。

---

## 15. 关键设计优点

### 15.1 多存储分层明确

项目没有把所有东西硬塞进单一数据库，而是根据数据特性分层：

- 业务结构化数据交给 SQLite
- 灵活长期记忆交给 JSON
- 语义知识交给 ChromaDB
- 报告文件和追踪交给文件系统

这让每种存储都工作在自己最合适的场景里。

### 15.2 数据链路完整

从用户问题到：

- 历史恢复
- 画像召回
- 检索增强
- 答案生成
- 长期记忆回写

形成了闭环。

ECG 部分从：

- 远程抓取
- 原始文件解析
- 特征计算
- 风险评估
- 报告生成
- PDF 导出

也形成了闭环。

### 15.3 并发安全考虑到位

虽然画像采用 JSON 文件存储，但通过：

- 全局锁
- 原子替换写入

基本覆盖了单进程多线程场景下的正确性要求。

### 15.4 可观测性良好

flow trace、任务状态和流式事件设计，使系统不仅能工作，而且比较容易排查问题。

---

## 16. 当前设计的边界与风险

### 16.1 JSON 画像仅适合低频写入

如果画像字段数量快速扩展，或者出现高频并发更新需求，文件存储会变得笨重。

### 16.2 锁仅限单进程

`threading.Lock` 不适合多进程或多实例部署。

### 16.3 ChromaDB 更适合中小规模知识库

当前实现以本地持久化为主，适合单机医疗知识库；如果知识规模和并发量继续上升，可能需要迁移到更专业的向量数据库。

### 16.4 SQLite 更适合轻量部署

对于本项目当前规模足够，但若未来：

- 并发用户增长
- 查询复杂度提高
- 需要更细粒度事务和索引

则应考虑切换到 PostgreSQL 等更成熟数据库。

---

## 17. 后续可演进方向

### 17.1 画像存储升级

可将 JSON 画像迁移为数据库表，支持：

- 更稳定的并发控制
- 更灵活的查询
- 更可控的字段演进

### 17.2 ECG 原始数据归档

当前更偏实时处理，可进一步引入：

- 原始信号长期归档
- 特征表
- 任务结果快照表

便于后续审计、回溯和模型评估。

### 17.3 检索链路增强

可进一步加入：

- 更强的 reranker
- 分段召回统计
- 命中率监控
- 检索评测集

### 17.4 可观测性升级

当前 flow trace 已够用，但未来可引入：

- 结构化 tracing 系统
- 指标监控
- 请求级日志聚合

---

## 18. 一句话总结

医枢智疗在存储与数据处理上采用了“SQLite 管业务、JSON 管长期记忆、ChromaDB 管语义知识、文件系统管报告与追踪、内存状态管实时工作流”的混合架构，并通过 LangGraph 把会话恢复、画像召回、科室级检索、ECG 原始信号解析、结构化报告生成串成了一条完整可落地的数据闭环。

---

## 19. 最近修复：损坏 ChromaDB 持久化目录自动重建

### 19.1 问题背景

项目原本的 ChromaDB 初始化逻辑优先复用本地持久化目录 `backend/storage/vector_store`。  
只要目录下存在类似 `chroma.sqlite3` 的文件名，系统就会判定“已有数据库”，进入加载旧库分支。

但在当前仓库中，这个目录里保存的是 Git LFS 占位文件而不是真实数据库文件，因此会出现：

- 目录结构看起来像已有 ChromaDB
- 实际加载时报错：`file is not a database`

### 19.2 旧逻辑为什么不会自动生成新库

旧逻辑的问题在于：

1. 第一次启动时先尝试加载旧库
2. 加载失败后返回 `None`
3. 主流程随后解析知识库 PDF，成功得到新的 `documents`
4. 之后再次调用 `get_or_create_vectorstore(documents)` 时，由于坏的持久化目录还在，代码仍然进入“加载旧库”分支
5. 第二次加载再次失败，函数直接返回 `None`
6. 真正的 `Chroma.from_documents(...).persist()` 创建分支没有机会执行

因此，问题不在于 PDF 没解析成功，而在于坏的旧目录始终把逻辑带入错误路径。

### 19.3 修复方案

本次修复在 `backend/app/tools/vector_store.py` 中加入了“损坏旧库自动删目录并重建”的机制。

核心改动如下：

1. 抽出 `_create_vectorstore_from_documents()` 私有方法，统一封装 `Chroma.from_documents(...).persist()` 创建逻辑。
2. 当检测到旧库加载失败且当前调用同时拿到了新的 `documents` 时：
   - 记录日志
   - 使用 `shutil.rmtree(persist_dir, ignore_errors=True)` 删除损坏目录
   - 立即基于当前知识文档重建新的持久化 ChromaDB
3. 当检测到旧库可加载但 `count() == 0` 时，也同样执行删目录重建。
4. 原有“旧库缺失 `department` 元数据时重建”的逻辑保留不变，只是复用了新的创建 helper。

### 19.4 修复后的行为变化

修复前：

- 旧库损坏时，系统只会报错
- 即使新知识文档已经被成功切块，也不会落盘成新的 ChromaDB
- RAG 初始化可能退化为“无可用向量库”

修复后：

- 旧库损坏时，如果当前已经有新文档可用，系统会自动删除损坏目录
- 直接进入 `from_documents + persist` 重建流程
- 持久化目录恢复为真实可用的 ChromaDB

### 19.5 测试覆盖

本次修复同时补充了 `backend/tests/test_tools.py` 中的单元测试，覆盖以下场景：

- 旧向量库损坏时自动重建
- 旧向量库为空时自动重建

另外也回归执行了原有的覆盖测试，确认没有破坏既有分支行为。

### 19.6 当前仍然存在的边界

这次修复解决的是“旧持久化目录损坏导致无法重建”的问题，但不解决知识文档本身的解析问题。

例如当前知识库中的某些大 PDF 在 `pypdf` 解析阶段仍可能出现：

- 解压限制触发
- 单文件处理耗时较长
- 某个文件 ingest 失败

因此这个修复保证的是：

- 向量库目录损坏时，系统能够正确重建

但不保证：

- 每一份知识文档都一定能被成功解析

### 19.7 工程意义

这次修改实质上提升了 RAG 初始化的鲁棒性：

- 从“依赖旧持久化目录必须完好”
- 变成“旧目录损坏时也能基于新知识源自恢复”

这使得项目的本地部署、重启恢复和知识库更新流程更加稳定，也降低了 Git LFS 缺失、文件损坏、历史库格式失配等问题对系统启动的影响。
