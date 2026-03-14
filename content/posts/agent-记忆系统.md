---
title: "Agent 记忆系统设计"
date: 2026-03-09T20:00:00+08:00
draft: false
tags: ["记忆系统", "向量数据库", "上下文管理", "AI Agent"]
categories: ["AI Agent"]
description: "深入探讨 AI Agent 记忆架构的设计，涵盖短期记忆、长期记忆和情节记忆的实现方案。"
---

# Agent 记忆系统设计

Agent 的记忆能力是其智能性的核心。一个优秀的记忆系统需要高效存储、快速检索和智能管理。本文深入探讨 Agent 记忆架构的设计。

## 一、记忆分类

### 1.1 按时间维度

```
┌─────────────────────────────────────────┐
│           短期记忆 (Working Memory)       │
│  - 当前任务上下文                          │
│  - 最近交互历史                           │
│  - 快速访问，容量有限                      │
└─────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────┐
│          长期记忆 (Long-term Memory)      │
│  - 知识库                                 │
│  - 经验总结                               │
│  - 持久化存储，容量大                      │
└─────────────────────────────────────────┘
```

### 1.2 按内容类型

| 类型 | 描述 | 存储方式 | 检索方式 |
|------|------|----------|----------|
| 事实记忆 | 固定知识 | 向量数据库 | 相似度搜索 |
| 情景记忆 | 经历事件 | 关系型数据库 | 结构化查询 |
| 程序记忆 | 技能和习惯 | 元数据索引 | 元素匹配 |
| 反思记忆 | 经验总结 | 日志分析 | 关键词检索 |

## 二、记忆架构设计

### 2.1 三层架构

```python
class MemorySystem:
    def __init__(self):
        # 1. 工作记忆（缓存）
        self.working_memory = MemoryBuffer(max_size=100)

        # 2. 长期记忆（持久化）
        self.long_term_memory = MemoryStore(
            vector_db=ChromaDB(),
            sql_db=PostgreSQL(),
            semantic_cache=LanceDB()
        )

        # 3. 反思模块（持续优化）
        self.reflector = ReflectionEngine()

    def store(self, item: MemoryItem):
        """存储记忆"""
        # 先存入工作记忆
        self.working_memory.add(item)

        # 定期持久化到长期记忆
        if self.working_memory.should_persist():
            self.long_term_memory.add(item)
            self.working_memory.clear()

    def retrieve(self, query: str) -> List[MemoryItem]:
        """检索记忆"""
        # 1. 优先从工作记忆查找
        direct_match = self.working_memory.search(query)
        if direct_match:
            return direct_match

        # 2. 从长期记忆检索
        return self.long_term_memory.search(
            query,
            threshold=0.7
        )
```

### 2.2 记忆检索流程

```python
def retrieve_relevant_context(agent_state, query: str, k: int = 5):
    """
    检索 Agent 相关的上下文记忆
    """
    # 1. 构建检索查询
    retrieval_query = f"""
    Given task: {query}
    Task history: {agent_state.history[-10:]}

    Retrieve {k} relevant memories:
    - Previous similar tasks
    - Relevant knowledge bases
    - Relevant examples
    """

    # 2. 向量化查询
    embedding = embed_model.encode(retrieval_query)

    # 3. 检索
    results = vector_db.search(embedding, k=k)

    # 4. 重排序（可选）
    reranked = reranker.rerank(
        query=retrieval_query,
        documents=results
    )

    # 5. 格式化上下文
    return format_context(reranked)
```

## 三、关键组件

### 3.1 向量数据库

**选择指南**：

| 数据库 | 特点 | 适用场景 |
|--------|------|----------|
| ChromaDB | 轻量、本地 | 开发/原型 |
| Pinecone | 托管、可扩展 | 生产环境 |
| Weaviate | 富语义查询 | 复杂查询 |
| Milvus | 高性能 | 大规模数据 |

**使用示例**：

```python
import chromadb

client = chromadb.PersistentClient(path="./memory_db")

# 创建 collection
collection = client.get_or_create_collection(
    name="agent_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# 存储记忆
collection.add(
    documents=[
        "Agent 记忆系统是 AI 智能体的核心组件",
        "短期记忆用于存储当前任务上下文",
        "长期记忆保存知识和经验"
    ],
    metadatas=[
        {"type": "fact", "created": "2026-03-01"},
        {"type": "concept", "created": "2026-03-01"},
        {"type": "fact", "created": "2026-03-01"}
    ],
    ids=["mem_1", "mem_2", "mem_3"]
)

# 检索
results = collection.query(
    query_texts=["Agent 记忆架构"],
    n_results=3
)
```

### 3.2 知识图谱

用于存储结构化关系和推理链：

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687")

def build_knowledge_graph(agent_id: str):
    with driver.session() as session:
        # 1. 创建 Agent 节点
        session.run("""
            CREATE (a:Agent {id: $id, created: datetime()})
        """, id=agent_id)

        # 2. 存储知识
        session.run("""
            MATCH (a:Agent {id: $id})
            MERGE (k:Knowledge {content: $content})
            MERGE (a)-[:HAS_KNOWLEDGE]->(k)
        """, id=agent_id, content="什么是 RAG?")

        # 3. 存储关系
        session.run("""
            MATCH (a:Agent {id: $id})
            MATCH (k1:Knowledge {content: $concept})
            MATCH (k2:Knowledge {content: $detail})
            MERGE (k1)-[:EXPLAINS]->(k2)
        """, id=agent_id, concept="RAG", detail="检索增强生成")

        # 4. 查询推理路径
        result = session.run("""
            MATCH path = (a:Agent {id: $id})-[:HAS_KNOWLEDGE*0..3]->(k:Knowledge)
            RETURN path
        """, id=agent_id)

        for record in result:
            print(record["path"])
```

## 四、记忆管理策略

### 4.1 增量更新

```python
def update_memory_memory(memory, new_item):
    """智能更新记忆"""
    # 1. 检查是否已存在
    existing = search_memory(memory, new_item.id)
    if existing:
        # 2. 如果相似度>0.9，合并而非覆盖
        if similarity(existing, new_item) > 0.9:
            memory.update(existing.id, new_item.content)
        else:
            # 3. 否则添加新版本
            memory.add(new_item)
    else:
        memory.add(new_item)
```

### 4.2 记忆衰减

```python
class DecayMemory:
    def __init__(self, decay_rate=0.95):
        self.decay_rate = decay_rate
        self.memories = []

    def add(self, memory):
        self.memories.append(memory)

    def get_relevant(self, query, k=10):
        # 计算衰减分数
        for mem in self.memories:
            mem.decay_score *= self.decay_rate

        # 按衰减分数排序
        ranked = sorted(
            self.memories,
            key=lambda m: m.decay_score,
            reverse=True
        )

        return ranked[:k]
```

### 4.3 反思与总结

```python
def reflect_on_actions(agent):
    """定期反思和总结"""
    recent_actions = agent.get_recent_actions(hours=24)

    # 1. 分析成功/失败模式
    success_patterns = analyze_success_patterns(recent_actions)

    # 2. 生成经验总结
    experience_summary = generate_summary(success_patterns)

    # 3. 存储反思记忆
    agent.memory.add(ReflectionMemory(
        content=experience_summary,
        type="reflection",
        timestamp=datetime.now()
    ))

    # 4. 更新技能
    update_skills(agent, experience_summary)
```

## 五、性能优化

### 5.1 混合检索

结合关键词检索和语义检索：

```python
from haystack import Pipeline
from haystack.nodes import BM25Retriever, EmbeddingRetriever

pipeline = Pipeline()

# 1. 关键词检索
bm25 = BM25Retriever(index="documents")
pipeline.add_node(node=bm25, name="BM25")

# 2. 语义检索
embedding = EmbeddingRetriever(document_store=es_store)
pipeline.add_node(node=embedding, name="Embedding")

# 3. 合并结果
from haystack.nodes import FusionRetriever
fusion = FusionRetriever(fusion="reciprocal_rank_fusion")
pipeline.add_node(node=fusion, name="Fusion")

# 执行检索
results = pipeline.run(query="记忆系统设计", top_k=10)
```

### 5.2 缓存策略

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_response(query: str) -> str:
    # 检查缓存
    cached = cache.get(query)
    if cached:
        return cached

    # 执行查询
    result = execute_query(query)

    # 存入缓存
    cache.set(query, result)
    return result
```

## 六、评估指标

### 6.1 准确性指标

- **检索准确率**：返回的记忆是否相关
- **召回率**：是否遗漏重要记忆
- **精度**：相关记忆的比例

### 6.2 性能指标

- **检索延迟**：平均查询响应时间
- **存储效率**：单位存储空间的信息密度
- **内存占用**：工作记忆占用

### 6.3 质量指标

- **可解释性**：记忆来源是否可追溯
- **一致性**：是否与历史一致
- **更新频率**：是否及时更新

## 七、最佳实践

1. **分层存储**：热数据用内存，冷数据用磁盘
2. **智能过滤**：只存储有价值的信息
3. **版本控制**：保留记忆的历史版本
4. **隐私保护**：敏感信息加密存储
5. **可审计性**：所有访问可追踪

## 八、参考资源

- [MemGPT: 虚拟操作系统](https://memgpt.ai/)
- [Mem0: 简单的 AI 记忆库](https://github.com/mem0ai/mem0)
- [LangChain Memory 模块](https://python.langchain.com/modules/memory/)

---

**相关文章**：
- [AI Agent 系统概览](/posts/ai-agent-概览.html)
- [RAG 系统最佳实践](/posts/rag-最佳实践.html)
