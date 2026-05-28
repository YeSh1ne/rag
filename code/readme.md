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
| 8. 生成      | 硅基流动LLM     | deepseek-ai/DeepSeek-V4-Flash  |
| 9.评测指标     | 硅基流动LLM     | Qwen/Qwen2.5-7B-Instruct       |

## 实验变量

对比测试 chunk\_size、top-k 检索数、embedding模型, LLM四类参数，完成多组消融实验

| chunk\_size | top-k | embedding模型 | LLM              | Recall@1 | Recall@3 | Recall@5 | MRR  | 引用准确率 | 回答正确率 |
| ----------- | ----- | ----------- | :--------------- | :------- | :------- | :------- | :--- | :---- | :---- |
| 512         | 5     | BGE-M3      | deepseekV4-flash | 0.61     | 0.68     | 0.70     | 0.69 | 0.57  | 0.81  |
|             |       |             |                  |          |          |          |      |       |       |
|             |       |             |                  |          |          |          |      |       |       |
|             |       |             |                  |          |          |          |      |       |       |

## 评测指标

Recall@k

- **核心定义**：在重排返回的前k个chunk中，有多少比例的Gold Evidence被成功找到。
- **评测逻辑**：
  1. 将Excel中的Gold Evidence文本解析为独立的证据条目列表。
  2. 依次检查重排结果Top-k中的每个chunk文本。
  3. 计算该chunk与每一条尚未被命中的Gold Evidence的语义相似度。
  4. 若相似度超过预设阈值（如0.75），则判定该条Gold Evidence被召回。
  5. **公式**：`Recall@k = (Top-k中命中的Gold Evidence数量) / (Gold Evidence总数量)`

MRR 平均倒数排名

- **核心定义**：衡量系统把第一个相关证据排在了第几位。
- **评测逻辑**：
  1. 按重排结果的原始排序，从第1个chunk开始逐个检查。
  2. 对每个chunk，计算其与所有Gold Evidence条目的最高语义相似度。
  3. 一旦某个chunk的相似度超过阈值，立即停止，记录其排名rank。
  4. **公式**：`MRR = 1 / rank`（若遍历完所有chunk均未命中，则为0）

回答正确率

- **核心定义**：生成的答案是否在语义上正确回答了问题。
- **评测逻辑**：
  - **指标**：`Answer Correctness F1` (RAGas 风格)
  - **评测逻辑**：Prompt 要求 LLM 对比【标准答案】与【模型回答】，直接输出知识点级别的 `TP, FP,FN`数量, 套用F1Score公式

引用准确率

- **核心定义**：生成答案中标注的引用来源，是否真正支撑了答案内容且与Gold Evidence一致。
- **评测逻辑**：
  1. 提取生成答案中附带的所有引用来源（sources）的文本内容。
  2. 对每个引用来源，计算其与Gold Evidence列表的语义相似度。
  3. 若相似度超过阈值，则该引用被视为命中, 按此计算`TP, FP, FN`数量。
  4. 套用F1score公式

F1score

$$
F1 = \frac{TP}{TP + 0.5 \times FP + 0.5 \times FN}
$$

- **TP（命中）**：模型正确提到的标准答案知识点数量 / 正确引用了 Gold Evidence。
- **FP（幻觉）**：模型捏造的、与标准答案明确矛盾的错误信息数量（Prompt 中特别强调了补充细节和论文名不算 FP）/ 引用了无关内容、查无此 ID、或重复引用同一个 Gold。
- **FN（遗漏）**：标准答案中有，但模型没提到的知识点数量 / 遗漏了应该引用的 Gold Evidence。

##