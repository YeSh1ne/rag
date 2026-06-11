import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import json
import time
import gc
from pathlib import Path
from typing import Optional

import torch
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from embedding_utils import APIEmbedder


# ========== 模型下载辅助函数 ==========
def download_from_modelscope(model_name: str, cache_dir: str = None) -> str:
    """
    从 ModelScope 下载模型，如果失败则返回原始模型名
    """
    # 使用绝对路径，避免从不同目录启动时重复下载
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
    
    try:
        from modelscope import snapshot_download
        print(f"📥 正在通过 ModelScope 下载模型: {model_name}...")
        model_path = snapshot_download(model_name, cache_dir=cache_dir)
        print(f"✅ ModelScope 下载完成，缓存至: {model_path}")
        return model_path
    except ImportError:
        print(f"⚠️ 未安装 modelscope（pip install modelscope），将使用 HuggingFace 源")
        return model_name
    except Exception as e:
        print(f"⚠️ ModelScope 下载失败: {e}")
        print(f"   回退到 HuggingFace 源: {model_name}")
        return model_name


# ========== 配置 ==========
EMBEDDING_MODEL = "BAAI/bge-m3"  # 本地: BAAI/bge-m3 | API: Qwen/Qwen3-Embedding-8B
USE_API_EMBEDDING = False  # True=使用硅基流动API, False=使用本地模型
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # 重排序模型（对比实验保持不变）
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# 向量数据库路径：与 build_vector_db.py 保持一致
MODEL_SHORT_NAME = EMBEDDING_MODEL.split("/")[-1].lower().replace("_", "-")
CHUNK_SIZE = "512"  # 与 build_vector_db.py 中使用的 chunk_size 一致
VECTOR_DB_DIR = rf"E:\rag_project\code\vector_db\{MODEL_SHORT_NAME}\chunk_{CHUNK_SIZE}"
COLLECTION_NAME = f"rag_papers_{CHUNK_SIZE}"

RETRIEVE_TOP_K = 20
RERANK_TOP_K = 5
MMR_LAMBDA = 0.55  # MMR多样性权重, 1.0=纯相关性, 0.0=纯多样性 (0.5=平衡)

LLM_MODEL = "deepseek-ai/DeepSeek-V4-Pro"  # 生成回答用的模型
SCORING_MODEL = "Qwen/Qwen2.5-32B-Instruct"  # 评分用的模型（14B，评判能力更强）
SILICONFLOW_API_KEY = "sk-budorxggodzqkjiedqprgkhffzuggepgrmwcakelpgexfqrb"  # 替换为您的硅基流动API Key
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
LLM_MAX_NEW_TOKENS = 1024
LLM_TEMPERATURE = 0.3


# ========== 1. 模型管理（懒加载 + 显存管理） ==========
class RAGModels:
    """懒加载模型，按需加载，避免同时占用显存"""
    def __init__(self):
        self._embedder = None
        self._reranker = None
        self._llm_client = None

    @property
    def embedder(self):
        if self._embedder is None:
            if USE_API_EMBEDDING:
                # 使用硅基流动 API（新增）
                print(f"📡 使用 API Embedding 模型: {EMBEDDING_MODEL}")
                self._embedder = APIEmbedder(
                    model_name=EMBEDDING_MODEL,
                    api_key=SILICONFLOW_API_KEY,
                    base_url=SILICONFLOW_BASE_URL
                )
            else:
                # 使用本地模型（原有代码保留）
                device = "cuda" if torch.cuda.is_available() else "cpu"
                # 尝试从 ModelScope 下载
                model_path = download_from_modelscope(EMBEDDING_MODEL)
                print(f"📦 加载 Embedding 模型: {model_path}")
                self._embedder = SentenceTransformer(
                    model_path,
                    device=device,
                    model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else {},
                )
        return self._embedder

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # 尝试从 ModelScope 下载
            model_path = download_from_modelscope(RERANKER_MODEL)
            print(f"📦 加载 Reranker 模型: {model_path}")
            self._reranker = CrossEncoder(
                model_path,
                device=device,
                model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else {},
            )
        return self._reranker

    @property
    def llm_client(self) -> OpenAI:
        """获取硅基流动API客户端"""
        if self._llm_client is None:
            self._llm_client = OpenAI(
                api_key=SILICONFLOW_API_KEY,
                base_url=SILICONFLOW_BASE_URL
            )
        return self._llm_client
    
    def generate_score(self, messages: list[dict], max_retries: int = 3) -> str:
        """使用硅基流动API生成评分（带重试机制）"""
        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat.completions.create(
                    model=SCORING_MODEL,
                    messages=messages,
                    max_tokens=200,
                    temperature=0.1,
                    top_p=0.9,
                    timeout=30,
                )
                return response.choices[0].message.content.strip()
            except KeyboardInterrupt:
                print("\n⚠️ 评分请求被中断")
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避: 1s, 2s, 4s
                    print(f"\n⚠️ 评分请求失败 (第{attempt+1}次): {e}")
                    print(f"   等待 {wait_time} 秒后重试...")
                    import time
                    time.sleep(wait_time)
                else:
                    print(f"\n⚠️ 评分请求失败 (已重试{max_retries}次): {e}")
                    raise

    def generate(self, messages: list[dict]) -> str:
        """使用硅基流动API生成回答"""
        response = self.llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE,
            top_p=0.9,
            timeout=120,  # 120秒超时（生成回答需要更长时间）
        )
        return response.choices[0].message.content.strip()
    
    def preload(self):
        """预加载所有模型到 GPU，避免首次查询时延迟"""
        print("🔄 预加载 Embedding 模型...")
        _ = self.embedder
        print("🔄 预加载 Reranker 模型...")
        _ = self.reranker
        print("✅ 所有模型预加载完成！")

# ========== 2. 向量检索 ==========
def retrieve(query: str, chroma_collection, embedder: SentenceTransformer, top_k: int = RETRIEVE_TOP_K) -> list[dict]:
    """
    向量检索：将查询编码后在 ChromaDB 中检索
    """
    query_with_prefix = BGE_QUERY_PREFIX + query
    
    query_embedding = embedder.encode(
        query_with_prefix,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()
    
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "page": results["metadatas"][0][i]["page"],
            "paper_id": results["metadatas"][0][i]["paper_id"],
            "title": results["metadatas"][0][i]["title"],
            "distance": results["distances"][0][i],
        })
    
    return retrieved


# ========== 3. 重排序 ==========
def enhance_query_for_rerank(query: str) -> str:
    """
    查询增强: 只对定义类问题使用增强,其他类型使用原始查询
    """
    # 检测是否是定义类问题
    definition_keywords = ["定义", "是什么", "什么是", "含义", "指", "define", "what is", "refers to"]
    is_definition = any(kw in query.lower() for kw in definition_keywords)
    
    if is_definition:
        enhanced = (
            f"请找到最相关且包含【明确定义】的段落。"
            f"优先选择包含概念解释、定义句式的段落 (如 'X 是指...', 'X refers to...', 'we define X as...')。\n\n"
            f"问题: {query}"
        )
        return enhanced
    else:
        # 其他类型问题使用原始查询,不增强
        return query


def rerank(query: str, retrieved_docs: list[dict], reranker: CrossEncoder, top_k: int = RERANK_TOP_K, embedder: Optional[SentenceTransformer] = None) -> list[dict]:
    """
    交叉编码器重排序 + MMR多样性选择
    """
    if not retrieved_docs:
        return []
    
    # 使用增强后的查询
    enhanced_query = enhance_query_for_rerank(query)
    
    pairs = [(enhanced_query, doc["text"]) for doc in retrieved_docs]
    scores = reranker.predict(pairs, show_progress_bar=False)
    
    for doc, score in zip(retrieved_docs, scores):
        doc["rerank_score"] = float(score)
    
    # MMR多样性重排序 (使用全局embedder避免重复加载)
    reranked = mmr_rerank(retrieved_docs, top_k, lambda_weight=MMR_LAMBDA, embedder=embedder)
    
    # 显示重排后的分布
    new_counts = {}
    for doc in reranked:
        new_counts[doc["title"]] = new_counts.get(doc["title"], 0) + 1
    print(f"   重排后分布: {', '.join([f'{k}: {v}' for k, v in new_counts.items()])}")
    
    return reranked


def mmr_rerank(docs: list[dict], top_k: int, lambda_weight: float = 0.7, embedder: Optional[SentenceTransformer] = None) -> list[dict]:
    """
    MMR (Maximal Marginal Relevance) 多样性重排序
    
    MMR = λ * sim(q, d) - (1-λ) * max_{dj∈S} sim(d, dj)
    
    :param docs: 已计算 rerank_score 的文档列表
    :param top_k: 返回的文档数量
    :param lambda_weight: λ 权重, 1.0=纯相关性, 0.0=纯多样性
    :param embedder: Embedding模型 (可选,避免重复加载)
    :return: 重排序后的文档列表
    """
    if not docs:
        return []
    
    import numpy as np
    
    # 使用传入的 embedder 或临时加载
    if embedder is None:
        try:
            embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        except:
            print("   [MMR] Embedding模型加载失败,降级为纯相关性排序")
            return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
    
    # 计算所有 chunk 的 embedding
    texts = [doc["text"] for doc in docs]
    embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    
    # 计算相似度矩阵
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)
    
    selected = []
    remaining = list(range(len(docs)))
    
    while len(selected) < top_k and remaining:
        best_score = -float('inf')
        best_idx = -1
        
        for idx in remaining:
            # 相关性得分 (rerank_score)
            relevance_score = docs[idx]["rerank_score"]
            
            # 多样性惩罚 (与已选 chunks 的最大相似度)
            if selected:
                max_sim_to_selected = max(sim_matrix[idx, s] for s in selected)
            else:
                max_sim_to_selected = 0.0
            
            # MMR 分数
            mmr_score = lambda_weight * relevance_score - (1 - lambda_weight) * max_sim_to_selected
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    return [docs[i] for i in selected]


# ========== 4. 构建 Prompt ==========
def build_prompt(query: str, context_docs: list[dict]) -> list[dict]:
    """
    构建对话消息列表（适配 Qwen2.5 chat template）
    """
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        context_parts.append(
            f"[{i}] {doc['title']} (第{doc['page']}页, {doc['chunk_id']})\n{doc['text']}"
        )
    
    context = "\n\n---\n\n".join(context_parts)
    system_prompt = (
        "你是一个学术论文问答助手。请严格基于提供的论文片段回答用户问题。\n\n"
        "【输出格式】\n"
        "1. 先输出回答正文（准确、简洁的中文），后输出引用\n\n"
        "2. 每个引用必须单独一行，格式：来自: [论文名, 第X页, chunk_id]\n\n"
        "【示例】\n"
        "该模型通过引入安全对齐机制提升了鲁棒性。\n"
        "来自: [SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model, 第2页, SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model_chunk_004]\n\n"
        "【引用原则】\n"
        "- 宁缺毋滥：只引用直接支持回答内容的chunk\n"
        "- 每个引用必须单独一行，不能多个引用连在一起\n"
        "- 每个引用必须与回答中的具体陈述对应\n"
        "- 1个准确引用 > 3个模糊引用\n"
        "- 论文名、页码、chunk_id必须与上下文完全一致\n\n"
        "【跨论文比较要求】\n"
        "- 如果问题要求比较不同论文的方法/结果，请分别引用各论文对应的chunk\n"
        "- 生成完所有回答正文再生成对应的引用,不要生成部分回答又生成相应引用又继续生成回答, 注意生成完正文后要换行再生成引用\n"
        "- 每个比较点必须明确指出来自哪篇论文\n"
        "- 不要编造论文之间没有的关联\n"
        "【注意】\n"
        "- 你的任务是基于上下文回答问题。如果问题要求定义某个术语，请仅输出该术语的核心定义，使用一句话，不要包含任何例子、构建方法、步骤或额外解释。\n"
        "- 请尽量从上下文中提取信息来回答问题。只有当上下文完全没有相关信息时，才输出:'根据已有信息，无法回答此问题。'\n"
        "- 如果上下文包含相关但不完整的信息，请基于已有信息给出部分回答，并说明信息有限。\n"
        "- 绝不能编造不存在的论文名、页码或chunk_id\n"
        "- 绝不能只输出引用而没有回答内容！\n"
        "- 绝不能无法回答还输出引用！\n"
    )
    user_content = (
        f"以下是相关论文片段：\n\n{context}\n\n"
        f"用户问题：{query}\n\n"
        f"请回答："
    )
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


# ========== 5. 主流程 ==========
def ask(question: str, models: RAGModels, chroma_collection) -> dict:
    """
    完整的 RAG 问答流程
    """
    print(f"\n{'='*60}")
    print(f"❓ 问题: {question}")
    print(f"{'='*60}")
    
    # Step 1: 向量检索
    t0 = time.time()
    print(f"\n[1/4] 向量检索 (top-k={RETRIEVE_TOP_K})...")
    retrieved = retrieve(question, chroma_collection, models.embedder)
    print(f"   召回 {len(retrieved)} 条，耗时: {time.time()-t0:.2f}秒")
    for i, doc in enumerate(retrieved):
        print(f"   [{i+1}] {doc['title']} (p.{doc['page']}) dist={doc['distance']:.4f}")
    
    # Step 2: 重排序
    t0 = time.time()
    print(f"\n[2/4] 重排序 (保留 top-{RERANK_TOP_K})...")
    reranked = rerank(question, retrieved, models.reranker, embedder=models.embedder)
    print(f"   耗时: {time.time()-t0:.2f}秒")
    for i, doc in enumerate(reranked):
        print(f"   [{i+1}] {doc['title']} (p.{doc['page']}) score={doc['rerank_score']:.4f}")
    
    # Step 3: 构建 Prompt
    print(f"\n[3/4] 构建 Prompt...")
    messages = build_prompt(question, reranked)
    
    # Step 4: LLM 生成
    t0 = time.time()
    print(f"\n[4/4] LLM 生成回答 ({LLM_MODEL})...")
    answer = models.generate(messages)
    print(f"   生成耗时: {time.time()-t0:.2f}秒")
    
    # ================= 新增：引用兜底补全逻辑 =================
    # 检查回答中是否包含 "来自:" 或 "来自："
    if "来自:" not in answer and "来自：" not in answer:
        # 如果模型没写引用，且不是拒答，我们强制把排名 Top-1 的 chunk 作为引用拼接到末尾
        if "无法回答此问题" not in answer and len(reranked) > 0:
            top_doc = reranked[0]
            forced_citation = f"\n\n来自: [{top_doc['title']}, 第{top_doc['page']}页, {top_doc['chunk_id']}]"
            answer = answer + forced_citation
            print("   ⚠️ 模型未生成引用，已自动补全 Top-1 引用。")
    # =======================================================
    
    sources = [
        {"paper_id": doc["paper_id"], "title": doc["title"], "page": doc["page"], "chunk_id": doc["chunk_id"], "text": doc["text"]}
        for doc in reranked
    ]
    
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "context_docs": reranked,
    }


# ========== 交互入口 ==========
def main():
    print("=" * 60)
    print("🚀 论文 RAG 问答系统")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB)")
    else:
        print("⚠️  CPU 模式（速度较慢）")
    
    # 加载检索模型
    models = RAGModels()
    
    # 连接 ChromaDB
    print(f"\n📦 连接向量数据库: {VECTOR_DB_DIR}")
    chroma_client = chromadb.PersistentClient(
        path=VECTOR_DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print(f"   Collection: {COLLECTION_NAME}, 共 {collection.count()} 条记录")
    
    # 交互模式
    print(f"\n{'='*60}")
    print("💬 进入问答模式（输入 'quit' 退出）")
    print(f"{'='*60}")
    
    while True:
        question = input("\n请输入问题: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break
        if not question:
            continue
        
        try:
            result = ask(question, models, collection)
            print(f"\n{'='*60}")
            print("📝 回答:")
            print(f"{'='*60}")
            print(result["answer"])
        except Exception as e:
            print(f"\n❌ 出错: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()