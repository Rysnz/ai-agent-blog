---
title: "RAG 在 Agent 系统中的应用"
date: 2026-03-09T21:00:00+08:00
draft: false
---

# RAG 在 Agent 系统中的应用

检索增强生成（RAG）是解决 LLM 知识更新滞后、幻觉问题的核心技术。当 Agent 需要调用专业领域知识时，RAG 系统提供了解决方案。本文介绍 RAG 的核心概念和实际应用。

## 一、RAG 基础

### 1.1 为什么需要 RAG？

| 问题 | 无 RAG | 有 RAG |
|------|--------|--------|
| 知识更新 | 需重新训练 | 实时更新 |
| 幻觉问题 | 容易产生 | 大幅减少 |
| 上下文限制 | 固定窗口 | 动态检索 |
| 事实准确度 | 依赖训练数据 | 依赖外部数据 |
| 专业领域知识 | 需微调 | 简单检索 |

### 1.2 RAG 工作流程

```mermaid
graph LR
    A[用户查询] --> B[向量化查询]
    B --> C[检索相关文档]
    C --> D[构建上下文]
    D --> E[LLM 生成回答]
    E --> F[返回结果]
```

## 二、RAG 系统架构

### 2.1 核心组件

```python
class RAGSystem:
    def __init__(self):
        # 1. 文档处理器
        self.document_processor = DocumentProcessor()

        # 2. 向量化模型
        self.embedder = SentenceTransformer(
            model="all-MiniLM-L6-v2"
        )

        # 3. 向量数据库
        self.vector_store = VectorStore(
            dimension=384  # all-MiniLM-L6-v2 输出维度
        )

        # 4. 重排序器（可选）
        self.reranker = CrossEncoderModel(
            model="BAAI/bge-reranker-v2-m3"
        )

        # 5. LLM
        self.llm = LLMClient()

    async def query(
        self,
        query: str,
        k: int = 5,
        temperature: float = 0.7
    ) -> str:
        """执行 RAG 查询"""
        # 1. 检索相关文档
        docs = await self.retrieve_documents(query, k)

        # 2. 重排序（可选）
        reranked_docs = await self.rerank_documents(
            query, docs
        )

        # 3. 构建上下文
        context = self.build_context(reranked_docs)

        # 4. 生成回答
        answer = await self.llm.generate(
            prompt=self.build_prompt(query, context),
            temperature=temperature
        )

        return answer
```

### 2.2 文档处理管道

```python
class DocumentProcessor:
    async def process(self, file_path: str) -> List[Document]:
        """处理文档并生成向量"""
        # 1. 读取文档
        content = await self.read_document(file_path)

        # 2. 清洗和分段
        segments = self.clean_and_segment(content)

        # 3. 向量化
        embeddings = self.embed_segments(segments)

        # 4. 创建 Document 对象
        documents = [
            Document(
                text=segment,
                embedding=embedding,
                metadata={
                    "source": file_path,
                    "segment_id": i
                }
            )
            for i, (segment, embedding) in enumerate(
                zip(segments, embeddings)
            )
        ]

        return documents

    def clean_and_segment(self, text: str) -> List[str]:
        """清洗和分段"""
        # 去除 Markdown 标记
        text = re.sub(r"#{1,6}\s+", "", text)

        # 按段落或标题分割
        segments = re.split(r"\n\n+", text)

        # 过滤空段落
        return [s.strip() for s in segments if s.strip()]

    def embed_segments(self, segments: List[str]) -> List[List[float]]:
        """向量化"""
        return self.embedder.encode(
            segments,
            convert_to_numpy=True,
            show_progress_bar=True
        )
```

## 三、高级 RAG 技术

### 3.1 混合检索

结合关键词检索和语义检索：

```python
class HybridRetriever:
    def __init__(self):
        self.semantic_retriever = SemanticRetriever()
        self.keyword_retriever = BM25Retriever()

    async def retrieve(
        self,
        query: str,
        k: int = 10,
        alpha: float = 0.7
    ) -> List[Document]:
        # 1. 语义检索
        semantic_results = await self.semantic_retriever.search(
            query, k=k
        )

        # 2. 关键词检索
        keyword_results = await self.keyword_retriever.search(
            query, k=k
        )

        # 3. 融合排序（RRF）
        fused = self.reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
            alpha
        )

        return fused[:k]

    def reciprocal_rank_fusion(
        self,
        list1: List[Document],
        list2: List[Document],
        alpha: float
    ) -> List[Document]:
        """Reciprocal Rank Fusion"""
        scores = {}

        # 初始化分数
        for doc in list1 + list2:
            scores[doc.id] = 0

        # RRF 算法
        for rank, doc in enumerate(list1, 1):
            scores[doc.id] += 1 / (alpha + rank)

        for rank, doc in enumerate(list2, 1):
            scores[doc.id] += 1 / ((1 - alpha) + rank)

        # 按分数排序
        sorted_docs = sorted(
            list1 + list2,
            key=lambda d: scores.get(d.id, 0),
            reverse=True
        )

        return sorted_docs
```

### 3.2 查询重写

优化检索质量：

```python
class QueryRewriter:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def rewrite(self, original_query: str) -> List[str]:
        """重写查询"""
        prompts = [
            # 1. 添加澄清
            f"Clarify the query: {original_query}\n"
            f"Ask questions if needed",
            # 2. 添加上下文
            f"Add context to: {original_query}\n"
            f"Think about what information might be missing",
            # 3. 变换表述
            f"Rewrite the query with different wording:\n{original_query}",
            # 4. 扩展查询
            f"Generate variations of: {original_query}"
        ]

        queries = []
        for prompt in prompts:
            response = await self.llm.generate(prompt)
            queries.extend([q.strip() for q in response.split("\n") if q.strip()])

        # 去重
        queries = list(set(queries))

        return queries
```

### 3.3 上下文压缩

减少 Token 消耗：

```python
class ContextCompressor:
    async def compress(
        self,
        query: str,
        documents: List[Document],
        max_length: int = 2000
    ) -> List[Document]:
        """压缩上下文"""
        # 1. 生成压缩提示
        prompt = f"""
        Query: {query}

        Documents:
        {self.format_documents(documents)}

        Summarize each document concisely (max 100 words) while keeping key information.
        """

        # 2. 调用 LLM 压缩
        compressed_docs = await self.llm.generate(prompt)

        # 3. 返回压缩结果
        return self.parse_compressed_docs(compressed_docs)
```

## 四、Agent 集成

### 4.1 RAG Agent

```python
class RAGAgent:
    def __init__(self, rag_system: RAGSystem):
        self.rag = rag_system

    async def execute(
        self,
        task: str,
        max_docs: int = 5,
        use_rag: bool = True
    ) -> str:
        """执行任务"""
        if use_rag:
            # 使用 RAG 检索信息
            context = await self.rag.retrieve_context(
                task,
                k=max_docs
            )
        else:
            context = None

        # 生成回答
        answer = await self.llm.generate(
            prompt=self.build_prompt(task, context),
            temperature=0.3  # 较低温度确保事实准确
        )

        return answer

    async def retrieve_context(
        self,
        query: str,
        k: int = 5
    ) -> str:
        """检索上下文"""
        docs = await self.rag.vector_store.search(query, k=k)
        return "\n\n".join(
            f"{'='*50}\nSource: {doc.metadata['source']}\n{'='*50}\n{doc.text}"
            for doc in docs
        )
```

### 4.2 多阶段 RAG

```python
class MultiStageRAG:
    async def query(
        self,
        query: str,
        stages: List[str] = ["retrieve", "refine", "answer"]
    ) -> str:
        """多阶段 RAG"""

        if "retrieve" in stages:
            # 阶段1：检索
            retrieved_docs = await self.retrieve(query)
            context = self.build_context(retrieved_docs)

        if "refine" in stages:
            # 阶段2：精炼
            refined_docs = await self.refine(
                query,
                context,
                n_refinements=2
            )
            context = self.build_context(refined_docs)

        if "answer" in stages:
            # 阶段3：生成答案
            answer = await self.generate_answer(
                query,
                context
            )

        return answer

    async def refine(
        self,
        query: str,
        initial_docs: List[Document],
        n_refinements: int = 2
    ) -> List[Document]:
        """精炼检索结果"""
        current_docs = initial_docs

        for i in range(n_refinements):
            # 1. 构建精炼查询
            refine_query = f"""
            Given query: {query}

            Current retrieved documents:
            {self.format_documents(current_docs)}

            Identify additional documents that would be most helpful.
            Generate search query to find them.
            """

            # 2. 执行精炼查询
            refined_query = await self.llm.generate(refine_query)
            new_docs = await self.vector_store.search(
                refined_query,
                k=len(current_docs)
            )

            # 3. 合并结果
            current_docs = await self.merge_documents(
                current_docs,
                new_docs
            )

        return current_docs[:len(initial_docs)]
```

## 五、性能优化

### 5.1 索引优化

```python
class OptimizedVectorStore:
    def __init__(self):
        self.store = ChromaDB(
            collection_name="rag_index",
            metadata={"hnsw:space": "cosine"}
        )

        # 启用分片
        self.store._persist = True

    async def batch_insert(
        self,
        documents: List[Document],
        batch_size: int = 100
    ):
        """批量插入"""
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            await self.store.add_documents(batch)
```

### 5.2 缓存策略

```python
class RAGCache:
    def __init__(self):
        self.cache = LRUCache(maxsize=1000)
        self.cache_file = "rag_cache.pkl"

    async def get(self, query: str) -> Optional[str]:
        """获取缓存结果"""
        cached = self.cache.get(query)
        if cached:
            return cached

        # 检查持久化缓存
        try:
            with open(self.cache_file, "rb") as f:
                persisted = pickle.load(f)
                if query in persisted:
                    result = persisted[query]
                    self.cache.put(query, result)
                    return result
        except FileNotFoundError:
            pass

        return None

    async def set(self, query: str, result: str):
        """设置缓存"""
        self.cache.put(query, result)

        # 持久化缓存
        with open(self.cache_file, "wb") as f:
            pickle.dump(dict(self.cache), f)
```

## 六、评估指标

### 6.1 检索质量

- **Recall@k**：前k个结果中包含相关文档的比例
- **Precision@k**：前k个结果中相关文档的比例
- **MRR**：平均倒数排名（考虑文档顺序）

### 6.2 生成质量

- **Hallucination Rate**：幻觉问题比例
- **Factuality Score**：事实准确性评分
- **User Satisfaction**：用户满意度调查

## 七、部署与监控

```python
class RAGMonitoring:
    def __init__(self):
        self.metrics = {
            "query_count": 0,
            "avg_retrieval_time": 0,
            "avg_generation_time": 0,
            "cache_hit_rate": 0
        }

    async def monitor_query(self, query: str, duration: float):
        """监控查询"""
        self.metrics["query_count"] += 1
        self.metrics["avg_retrieval_time"] = (
            self.metrics["avg_retrieval_time"] * (self.metrics["query_count"] - 1) + duration
        ) / self.metrics["query_count"]

        # 记录详细日志
        logger.info(
            "rag_query",
            extra={
                "query": query[:100],
                "duration_ms": duration * 1000,
                "metrics": self.metrics
            }
        )
```

## 八、最佳实践

1. **文档质量**：定期更新和维护知识库
2. **分片策略**：合理的文档分段长度（300-500 tokens）
3. **查询优化**：使用查询重写和扩展
4. **评估驱动**：持续评估检索和生成质量
5. **渐进式改进**：从简单 RAG 开始，逐步添加高级特性

## 九、参考资源

- [LangChain Retrieval](https://python.langchain.com/docs/data_connection/)
- [LlamaIndex](https://docs.llamaindex.ai/)
- [Weaviate RAG Guide](https://weaviate.io/developers/weaviate/tutorials/rag-guide)

---

**相关文章**：
- [AI Agent 系统概览](/posts/ai-agent-概览.html)
- [RAG 系统最佳实践](/posts/rag-最佳实践.html)
