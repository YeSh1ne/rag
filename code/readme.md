# 论文 RAG 问答系统

---

## 项目简介

本项目以 40 篇大模型与引文评估相关学术论文为数据源，搭建轻量化本地 RAG 问答框架。可接收自然语言中文问题，从论文知识库中匹配相关文本片段，结合大模型生成作答内容，同时输出**论文名称、页码、chunk\_id**三级引用来源，满足文献查阅、论文研读、问题溯源等使用场景。

## 技术选型

| 阶段         | 组件          | 技术选型                           |
| ---------- | ----------- | ------------------------------ |
| 1. 文档加载与解析 | PDF提取       | Marker                         |
| 2. 文本分块    | 文本分块        | tiktoken                       |
| 3. 向量化     | Embedding模型 | BGE-M3 / Qwen3-Embedding       |
|            | 批量处理        | SentenceTransformer.encode()   |
| 4. 向量存储    | 向量数据库       | ChromaDB / Qdrant              |
|            | 索引类型        | HNSW                           |
| 5. 检索      | 向量检索        | ChromaDB.similarity\_search()  |
|            | 检索框架        | LlamaIndex Retriever           |
| 6. 重排序     | 重排序模型       | BGE-Reranker-v2-m3             |
|            | 重排序策略       | Cross-Encoder                  |
| 7. 提示构建    | Prompt模板    | LangChain PromptTemplate       |
|            | 上下文压缩       | ContextualCompressionRetriever |
| 8. 生成      | 本地LLM       | Qwen2.5-3B-Instruct            |
|            | 生成框架        | LangChain LLMChain             |

## 实验变量

对比测试 chunk\_size、top-k 检索数、embedding模型三类参数，完成多组消融实验

| chunk\_size | top-k | embedding模型 |
| ----------- | ----- | ----------- |
|             |       |             |
|             |       |             |

## 评测指标

Recall@k、MRR 平均倒数排名、回答正确率、引用准确