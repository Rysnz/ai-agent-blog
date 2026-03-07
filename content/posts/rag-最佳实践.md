---
title: "RAG 系统最佳实践"
date: 2026-03-07T12:00:00+08:00
draft: false
tags: ["RAG", "Vector DB", "Retrieval"]
categories: ["AI Agent"]
---

# RAG 系统最佳实践

## 什么是 RAG？

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合外部知识检索与 LLM 生成能力的技术。它解决了 LLM 知识截止、幻觉等问题。

## RAG 系统架构

### 核心组件

```
┌─────────────┐
│   用户输入   │
└──────┬──────┘
       ↓
┌──────────────────────────────┐
│      文本预处理与分块        │
└──────┬───────────────────────┘
       ↓
┌──────────────────────────────┐
│      向量化与嵌入            │
└──────┬───────────────────────┘
       ↓
┌──────────────────────────────┐
│    向量数据库存储             │
└──────┬───────────────────────┘
       ↓
┌──────────────────────────────┐
│   检索（检索相关文档片段）    │
└──────┬───────────────────────┘
       ↓
┌──────────────────────────────┐
│     上下文拼接与生成          │
└──────┬───────────────────────┘
       ↓
┌──────────────────────────────┐
│     添加引用和答案            │
└──────────────────────────────┘
```

## 1. 文本分块策略

### 1.1 分块原则

**Chunk Size**：
- 通常 500-1000 tokens
- 适中大小：平衡检索精度和上下文完整性

**Chunk Overlap**：
- 10-20% 重叠
- 保证上下文连续性

### 1.2 分块方法

#### 固定大小分块（Simple）
```python
def simple_chunking(text, chunk_size=512, overlap=50):
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks
```

#### 语义分块（Semantic）
基于语义相似度自动分块：
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)

chunks = splitter.split_text(text)
```

#### 递归分块（Recursive）
逐层尝试不同分隔符，保证内容完整性：
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=[
        "\n\n",  # 段落
        "\n",    # 换行
        "。",    # 中文句号
        ".",     # 英文句号
        "!",     # 感叹号
        "?",     # 问号
        " ",     # 空格
        ""       # 字符
    ]
)
```

## 2. 向量化模型选择

### 2.1 Embedding 模型对比

| 模型 | 维度 | 语言 | 性能 |
|------|------|------|------|
| text-embedding-3-small | 1536 | 多语言 | ⭐⭐⭐⭐ |
| text-embedding-3-large | 3072 | 多语言 | ⭐⭐⭐⭐⭐ |
| bge-m3 | 1024 | 中文/英文 | ⭐⭐⭐⭐ |
| m3e-base | 768 | 中文 | ⭐⭐⭐ |

### 2.2 中文优化

推荐使用中文优化的模型：
```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)
```

## 3. 检索策略优化

### 3.1 相似度搜索

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# 创建向量存储
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings()
)

# 搜索
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 5}  # 返回top 5
)
```

### 3.2 混合检索

结合关键词和语义检索：
```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# 关键词检索
bm25_retriever = BM25Retriever.from_documents(chunks)

# 语义检索
faiss_retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

# 混合检索
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]  # 权重
)
```

### 3.3 重排序（Rerank）

检索后重新排序，提高精度：
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

## 4. 上下文优化

### 4.1 动态上下文
根据查询动态调整检索内容：
```python
def build_context(query, top_k=3):
    # 检索相关文档
    docs = retriever.get_relevant_documents(query)
    
    # 提取和查询相关的部分
    relevant_parts = extract_relevant_parts(docs, query)
    
    # 格式化为提示词
    context = "\n\n".join([
        f"文档 {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(relevant_parts)
    ])
    
    return context
```

### 4.2 引用格式化
```python
def format_response(llm_response, retrieved_docs):
    """
    添加引用信息到回答中
    """
    citations = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get('source', '未知来源')
        citations.append(f"[{i}] {source}")
    
    return f"""
    {llm_response}

    参考来源:
    {' | '.join(citations)}
    """
```

## 5. 常见问题与解决方案

### 5.1 幻觉问题

**原因**：检索不准确、LLM 泛化过度

**解决方案**：
- 使用更强的检索模型
- 增加检索数量（k值）
- 在提示词中强调"仅使用提供的上下文"

### 5.2 检索准确性不足

**解决方案**：
- 使用重排序（Rerank）
- 优化分块策略
- 考虑混合检索
- 使用专门针对任务微调的模型

### 5.3 上下文过长

**解决方案**：
- 动态调整上下文长度
- 使用摘要压缩
- 分阶段检索和回答

## 6. 监控与优化

### 6.1 关键指标

- **检索精度**：召回率和准确率
- **回答质量**：用户满意度、幻觉率
- **性能**：响应时间、资源使用

### 6.2 持续优化

```python
def evaluate_rag_system(query, ground_truth, rag_response):
    """
    评估 RAG 系统质量
    """
    # 1. 检索准确性评估
    retrieved_docs = retriever.get_relevant_documents(query)
    retrieval_score = calculate_retrieval_score(retrieved_docs, ground_truth)
    
    # 2. 生成质量评估
    generation_score = evaluate_llm_output(
        rag_response,
        ground_truth,
        context=retrieved_docs
    )
    
    return retrieval_score * 0.4 + generation_score * 0.6
```

## 最佳实践总结

1. **分块策略**：使用递归分块，保证语义完整性
2. **检索优化**：混合检索 + 重排序
3. **模型选择**：中文任务优先考虑 bge-m3
4. **上下文管理**：动态检索和引用
5. **持续优化**：监控指标，迭代改进

通过遵循这些最佳实践，可以构建高质量、可维护的 RAG 系统。
