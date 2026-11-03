# 论文 RAG 问答系统

## 项目简介

本项目以 40 篇大模型与引文评估相关学术论文为数据源，搭建轻量化本地 RAG 问答框架。可接收自然语言中文问题，从论文知识库中匹配相关文本片段，结合大模型生成作答内容，同时输出**论文名称、页码、chunk\_id**三级引用来源，满足文献查阅、论文研读、问题溯源等使用场景。

## 项目结构

```
code/
├── parsePaper.py            # 阶段1+2：PDF解析为Markdown + 按Token分片
├── build_vector_db.py       # 阶段3：Embedding编码 + ChromaDB向量存储
├── rag_pipeline.py          # 核心Pipeline：检索→重排序→Prompt构建→LLM生成
├── embedding_utils.py       # APIEmbedder适配器：封装硅基流动API，兼容SentenceTransformer接口
├── run_evaluation.py        # 评测运行脚本：串联RAG Pipeline与评测器
├── evaluator/               # 评测模块
│   ├── evaluator.py         # 评测器主类：基于内容匹配的RAG评测
│   ├── metrics.py           # 评测指标：Recall@K、MRR、引用准确率
│   ├── scorer.py            # LLM评分：RAGas风格知识点拆解(TP/FP/FN) + Claim验证
│   ├── utils.py             # 评测工具：语义相似度计算、Gold Evidence解析、Markdown清洗
│   └── excel_handler.py     # Excel输出：逐题追加结果 + 平均值公式行
├── web_app_flask/           # Flask Web应用
│   ├── app.py               # Flask后端：路由、SSE流式推送、PDF引用映射
│   ├── templates/
│   │   └── index.html       # 前端页面模板
│   └── static/
│       ├── script.js        # 前端交互逻辑（SSE接收、对话管理）
│       ├── style.css        # 页面样式
│       └── papers/          # 论文PDF文件（供在线预览和页码跳转）
├── parsed_markdown_raw/     # 阶段1输出：PDF解析后的原始Markdown（JSON格式）
├── parsed_output_256/       # 阶段2输出：chunk_size=256的分片结果
├── parsed_output_512/       # 阶段2输出：chunk_size=512的分片结果
├── parsed_output_1024/      # 阶段2输出：chunk_size=1024的分片结果
├── vector_db/               # 阶段3输出：ChromaDB向量数据库
│   ├── bge-m3/              # BGE-M3 Embedding 的向量库
│   └── qwen3-embedding-8b/  # Qwen3-Embedding-8B 的向量库
├── experiment/              # 对比实验结果（Excel文件）
└── readme.md                # 本文件
```

## 快速启动

### 环境要求

- Python 3.10+
- GPU（推荐，用于模型推理）或 CPU（速度较慢）
- 硅基流动 API Key（用于 LLM 生成和可选的 API Embedding）

### 安装依赖

```bash
pip install flask chromadb sentence-transformers torch openai tiktoken modelscope
pip install marker-pdf  # PDF解析
pip install pandas openpyxl  # 评测相关
```

### 启动 Web 问答界面

```bash
cd e:\rag_project\code
python web_app_flask/app.py
```

启动后在浏览器访问 **http://localhost:5000** 即可使用。

> 首次启动会自动加载 Embedding 模型、Reranker 模型和向量数据库，需要等待约 1-2 分钟。

### 命令行交互模式

```bash
cd e:\rag_project\code
python rag_pipeline.py
```

### 运行评测

```bash
cd e:\rag_project\code
python run_evaluation.py
```

## 技术选型

| 阶段         | 组件          | 技术选型                                 |
| ---------- | ----------- | ------------------------------------ |
| 1. 文档加载与解析 | PDF提取       | Marker                               |
| 2. 文本分块    | 文本分块        | tiktoken                             |
| 3. 向量化     | Embedding模型 | BGE-M3 / Qwen3-Embedding-8B          |
| 4. 向量存储    | 向量数据库       | ChromaDB                             |
| 5. 检索      | 向量检索        | ChromaDB.similarity\_search()        |
|            | 检索框架        | LlamaIndex Retriever                 |
| 6. 重排序     | 重排序模型       | BGE-Reranker-v2-m3                   |
|            | 重排序策略       | Cross-Encoder 结合 MMR多样性重排序           |
| 7. 生成      | 硅基流动LLM     | deepseek-ai/DeepSeek-V4-Pro          |
| 8. 评测指标    | 回答正确率       | Qwen/Qwen2.5-32B-Instruct as a Judge |
|            | recall, mrr | 语义相似度                                |
|            | 引用准确率       | 语义相似度                                |

## 对比实验

对比测试 chunk\_size、top-k 检索数、embedding模型, LLM四类参数，完成多组消融实验

相似度阈值:  recall:0.75, cite:0.75

| chunk\_size | top-k |   embedding模型   |       LLM      | Recall@1 | Recall@3 | Recall@5 |  MRR  | 引用准确率 | 回答正确率 |
| :---------: | :---: | :-------------: | :------------: | :------: | :------: | :------: | :---: | :---: | :---: |
|     512     |   5   |      BGE-M3     | deepseekV4-Pro |   0.881  |   0.952  |   0.952  | 0.909 | 0.931 | 0.924 |
|     512     |   10  |      BGE-M3     | deepseekV4-Pro |   0.929  |   0.952  |   0.952  | 0.937 | 0.897 | 0.875 |
|     256     |   5   |      BGE-M3     | deepseekV4-Pro |   0.786  |   0.929  |   0.976  | 0.857 | 0.913 | 0.910 |
|     256     |   10  |      BGE-M3     | deepseekV4-Pro |   0.810  |   0.952  |   0.976  | 0.875 | 0.845 | 0.915 |
|     1024    |   5   |      BGE-M3     | deepseekV4-Pro |   0.738  |   0.857  |   0.929  | 0.809 | 0.793 | 0.849 |
|     1024    |   10  |      BGE-M3     | deepseekV4-Pro |   0.738  |   0.857  |   0.929  | 0.809 | 0.716 | 0.866 |
|     512     |   5   | Qwen3-Embedding | deepseekV4-Pro |   0.929  |   0.952  |   0.976  | 0.945 | 0.913 | 0.903 |
|     512     |   10  | Qwen3-Embedding | deepseekV4-Pro |   0.952  |   0.952  |   0.976  | 0.957 | 0.875 | 0.900 |
|     256     |   5   | Qwen3-Embedding | deepseekV4-Pro |   0.890  |   0.929  |   0.976  | 0.864 | 0.881 | 0.922 |
|     256     |   10  | Qwen3-Embedding | deepseekV4-Pro |   0.810  |   0.929  |   0.976  | 0.860 | 0.855 | 0.922 |
|     1024    |   5   | Qwen3-Embedding | deepseekV4-Pro |   0.738  |   0.881  |   0.929  | 0.807 | 0.726 | 0.903 |
|     1024    |   10  | Qwen3-Embedding | deepseekV4-Pro |   0.738  |   0.881  |   0.929  | 0.808 | 0.720 | 0.894 |

## 评测指标

**Recall@k**（本报告采用二元命中定义，也称 Hit@k）

- **核心定义**：对于单个问题，若前 k 个检索结果（chunk）中**至少有一个 chunk 与任意一条 Gold Evidence 语义匹配**，则判定该问题检索成功（得分为 1），否则为 0。
- **评测逻辑**：
  1. 提取该问题的所有 Gold Evidence 文本。
  2. 依次检查重排结果 Top‑k 中的每个 chunk 文本。
  3. 采用混合匹配计算每个 chunk 与所有 Gold Evidence 的语义相似度（使用相似度模型）。
  4. 若相似度超过预设阈值（此系统采用 0.75），则判定该 chunk 命中。
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
  - AnswerCorrectness \= TP / （TP + 0.5 × FP + 0.5 × FN ）
  - TP True Positives（命中） 模型回答中 正确提到 的标准答案知识点数量&#x20;
  - FP False Positives（幻觉） 模型回答中 捏造的 或 与标准答案矛盾的 错误信息数量&#x20;
  - FN False Negatives（遗漏） 标准答案中 有但回答完全没提到 的知识点数量

## **引用准确率 (Citation Accuracy)**

- **核心定义**：对于单个问题，模型生成的引用中，正确命中 Gold Evidence 的引用比例。每条 Gold Evidence 只计一次正确命中（重复命中不计分），未命中任何 Gold 的引用记为错误。
- **评测逻辑**：
  1. 从模型生成的回答中提取所有引用（如 `chunk_id`）。
  2. 对每个引用，通过向量数据库获取其原文文本。
  3. 采用混合匹配计算引用原文与所有 Gold Evidence 的语义相似度。
  4. 若相似度超过预设阈值（此系统采用0.75），则判定该引用命中；若该 Gold 尚未被命中过，则计为正确引用（TP），否则不计分。
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

## **MMR（最大边际相关性）**

重排序阶段采用 MMR 多样性策略，在保证与 Query 相关性的同时，提升结果集中 chunk 的多样性，解决跨论文比较问题中重排被单篇论文主导的困境。

**公式**：

$$
\text{MMR} = \lambda \cdot \text{sim}(q, d) - (1 - \lambda) \cdot \max_{d_j \in S} \text{sim}(d, d_j)
$$

| 参数 | 含义 |
|------|------|
| $\text{sim}(q, d)$ | chunk $d$ 与 Query $q$ 的语义相似度（相关性项） |
| $\max_{d_j \in S} \text{sim}(d, d_j)$ | chunk $d$ 与已选集合 $S$ 中所有 chunk 的最大相似度（多样性惩罚项） |
| $\lambda$ | 权重系数，$\lambda = 1$ 为纯相关性，$\lambda = 0$ 为纯多样性 |

**为什么有效**：

- **单论文问题**：相关 chunks 少，MMR 不会强行引入无关论文（相关性分数低）。
- **跨论文比较问题**：某篇论文的最佳 chunk 被选中后，另一篇论文的最佳 chunk 因与已选集合差异大、相关性也够高，会被优先选中。
