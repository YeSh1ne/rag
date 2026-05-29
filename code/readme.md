# 论文 RAG 问答系统

---

## 项目简介

本项目以 40 篇大模型与引文评估相关学术论文为数据源，搭建轻量化本地 RAG 问答框架。可接收自然语言中文问题，从论文知识库中匹配相关文本片段，结合大模型生成作答内容，同时输出**论文名称、页码、chunk\_id**三级引用来源，满足文献查阅、论文研读、问题溯源等使用场景。

## 技术选型

| 阶段         | 组件          | 技术选型                                |
| ---------- | ----------- | ----------------------------------- |
| 1. 文档加载与解析 | PDF提取       | Marker                              |
| 2. 文本分块    | 文本分块        | tiktoken                            |
| 3. 向量化     | Embedding模型 | BGE-M3 / Qwen3-Embedding            |
|            | 批量处理        | SentenceTransformer.encode()        |
| 4. 向量存储    | 向量数据库       | ChromaDB / Qdrant                   |
|            | 索引类型        | HNSW                                |
| 5. 检索      | 向量检索        | ChromaDB.similarity\_search()       |
|            | 检索框架        | LlamaIndex Retriever                |
| 6. 重排序     | 重排序模型       | BGE-Reranker-v2-m3                  |
|            | 重排序策略       | Cross-Encoder                       |
| 7. 提示构建    | Prompt模板    | LangChain PromptTemplate            |
|            | 上下文压缩       | ContextualCompressionRetriever      |
| 8. 生成      | 硅基流动LLM     | deepseek-ai/DeepSeek-V4-Flash       |
| 9.评测指标     | 回答正确率       | Qwen/Qwen2.5-7B-Instruct as a Judge |
|            | recall, mrr | 语义相似度                               |
|            | 引用准确率       | F1score                             |

## 实验变量

对比测试 chunk\_size、top-k 检索数、embedding模型, LLM四类参数，完成多组消融实验

| chunk\_size | top-k | embedding模型 |        LLM       | Recall@1 (Binary) | Recall@3(Coverage) | Recall@5(Coverage) |  MRR | 引用准确率 | 回答正确率 |
| :---------: | :---: | :---------: | :--------------: | :---------------: | :----------------: | :----------------: | :--: | :---: | :---: |
|     512     |   5   |    BGE-M3   | deepseekV4-flash |        0.61       |        0.68        |        0.70        | 0.69 |  0.57 |  0.81 |
|             |       |             |                  |                   |                    |                    |      |       |       |
|             |       |             |                  |                   |                    |                    |      |       |       |
|             |       |             |                  |                   |                    |                    |      |       |       |

## 评测指标

**Recall@k**（本报告采用二元命中定义，也称 Hit@k）

- **核心定义**：对于单个问题，若前 k 个检索结果（chunk）中**至少有一个 chunk 与任意一条 Gold Evidence 语义匹配**，则判定该问题检索成功（得分为 1），否则为 0。
- **评测逻辑**：
  1. 提取该问题的所有 Gold Evidence 文本。
  2. 依次检查重排结果 Top‑k 中的每个 chunk 文本。
  3. 计算每个 chunk 与所有 Gold Evidence 的语义相似度（使用相似度模型）。
  4. 若相似度超过预设阈值（如 0.75），则判定该 chunk 命中。
  5. 若 Top‑k 中**存在至少一个命中的 chunk**，则该问题的 Recall@k \= 1；否则 Recall@k \= 0。
- **计算公式**：

$$
\text{Recall@k} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left( \bigvee_{j=1}^{k} \text{sim}(c_{ij}, G_i) > \tau \right)
$$

- $N$：测试问题总数
- $c_{ij}$：第 $i$ 个问题的前 $j$ 个检索结果
- $G_i$：第 $i$ 个问题的 Gold Evidence 集合
- $\text{sim}(c, G_i) = \max_{g \in G_i} \text{similarity}(c, g)$
- $\tau$：语义匹配阈值
- $\mathbf{1}(\cdot)$：指示函数（条件为真时取值 1）
- **最终指标**：对所有测试问题取算术平均，得到系统整体的 Recall@k。

## **MRR 平均倒数排名**

- **核心定义**：衡量系统把第一个相关证据排在了第几位。
- **评测逻辑**：
  1. 按重排结果的原始排序，从第1个chunk开始逐个检查。
  2. 对每个chunk，计算其与所有Gold Evidence条目的最高语义相似度。
  3. 一旦某个chunk的相似度超过阈值，立即停止，记录其排名rank。
  4. **公式**：`MRR = 1 / rank`（若遍历完所有chunk均未命中，则为0）

## **回答正确率**

- **核心定义**：生成的答案是否在语义上正确回答了问题。
- **评测逻辑**：
  - **指标**：`Answer Correctness F1` (RAGas 风格)
  - **评测逻辑**：Prompt 要求 LLM 对比【标准答案】与【模型回答】，直接输出知识点级别的 `TP, FP,FN`数量, 套用F1Score公式

## **引用准确率 (Citation Accuracy)**

- **核心定义**：对于单个问题，模型生成的引用中，正确命中 Gold Evidence 的引用比例。每条 Gold Evidence 只计一次正确命中（重复命中不计分），未命中任何 Gold 的引用记为错误。
- **评测逻辑**：
  1. 从模型生成的回答中提取所有引用（如 `chunk_id`）。
  2. 对每个引用，通过向量数据库获取其原文文本。
  3. 计算引用原文与所有 Gold Evidence 的语义相似度。
  4. 若相似度超过预设阈值（如 0.75），则判定该引用命中；若该 Gold 尚未被命中过，则计为正确引用（TP），否则不计分。
  5. 若相似度未超过阈值，或找不到原文，则计为错误引用（FP）。
  6. 问题的引用准确率 \= TP / (TP + FP)（若没有生成任何引用，则得分为 0）。
  7. 对所有问题取算术平均，得到系统整体的引用准确率。
- 计算公式

$$
\text{CitationAcc} = \frac{1}{N} \sum_{i=1}^{N} \frac{TP_i}{TP_i + FP_i}
$$

&#x20;其中：

- $N$：测试问题总数
- $TP_i$：问题 $i$ 中正确命中 Gold Evidence 的引用数量（每个 Gold 仅计一次）
- $FP_i$：问题 $i$ 中未命中任何 Gold Evidence 的引用数量
- 若 $TP_i + FP_i = 0$，则 $\frac{TP_i}{TP_i + FP_i} = 0$

##