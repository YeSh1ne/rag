"""
rag_pipeline.py - 阶段4：检索 + 重排 + 生成（全本地部署）

完整 RAG 问答流程：
1. 用户输入中文问题
2. BGE-M3 生成查询向量（带前缀）
3. ChromaDB 向量检索（top-k=10）
4. BGE-Reranker 重排序（精排取 top-5）
5. 构建 Prompt，调用本地 LLM 生成中文回答
6. 输出答案 + 引用来源（论文名、页码、chunk_id）

本地模型（8GB 显存适配）：
- Embedding: BAAI/bge-m3 (fp16, ~2.2GB)
- Reranker: BAAI/bge-reranker-v2-m3 (fp16, ~2.2GB)
- LLM: Qwen/Qwen2.5-3B-Instruct (fp16, ~6GB)
  模型按需加载，不会同时占用显存
"""

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
from transformers import AutoTokenizer, AutoModelForCausalLM


# ========== 模型下载辅助函数 ==========
def download_from_modelscope(model_name: str, cache_dir: str = "./model_cache") -> str:
    """
    从 ModelScope 下载模型，如果失败则返回原始模型名
    """
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
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# 向量数据库路径：与 build_vector_db.py 保持一致
MODEL_SHORT_NAME = EMBEDDING_MODEL.split("/")[-1].lower().replace("_", "-")
CHUNK_SIZE = "512"  # 与 build_vector_db.py 中使用的 chunk_size 一致
VECTOR_DB_DIR = rf"E:\rag_project\code\vector_db\{MODEL_SHORT_NAME}\chunk_{CHUNK_SIZE}"
COLLECTION_NAME = f"rag_papers_{CHUNK_SIZE}"

RETRIEVE_TOP_K = 10
RERANK_TOP_K = 5

LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LLM_MAX_NEW_TOKENS = 1024
LLM_TEMPERATURE = 0.3


# ========== 1. 模型管理（懒加载 + 显存管理） ==========
class RAGModels:
    """懒加载模型，按需加载，避免同时占用显存"""
    def __init__(self):
        self._embedder = None
        self._reranker = None
        self._llm_tokenizer = None
        self._llm_model = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
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

    def load_llm(self):
        """加载 LLM 到 GPU"""
        if self._llm_model is not None:
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 尝试从 ModelScope 下载
        model_path = download_from_modelscope(LLM_MODEL)
        print(f"📦 加载 LLM: {model_path} (fp16)")
        
        self._llm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._llm_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        
        if device == "cpu":
            self._llm_model = self._llm_model.to(device)
        self._llm_model.eval()
        print(f"   ✅ LLM 加载完成")

    def unload_llm(self):
        """卸载 LLM，释放显存"""
        if self._llm_model is not None:
            del self._llm_model
            del self._llm_tokenizer
            self._llm_model = None
            self._llm_tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            print("   🗑️  LLM 已卸载，显存已释放")

    def generate(self, messages: list[dict]) -> str:
        """使用 LLM 生成回答"""
        self.load_llm()
        
        text = self._llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self._llm_tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._llm_model.generate(
                **inputs,
                max_new_tokens=LLM_MAX_NEW_TOKENS,
                temperature=LLM_TEMPERATURE,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        
        # 只解码新生成的部分
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self._llm_tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return answer.strip()


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
def rerank(query: str, retrieved_docs: list[dict], reranker: CrossEncoder, top_k: int = RERANK_TOP_K) -> list[dict]:
    """
    交叉编码器重排序
    """
    if not retrieved_docs:
        return []
    
    pairs = [(query, doc["text"]) for doc in retrieved_docs]
    scores = reranker.predict(pairs, show_progress_bar=False)
    
    for doc, score in zip(retrieved_docs, scores):
        doc["rerank_score"] = float(score)
    
    reranked = sorted(retrieved_docs, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


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
        "你是一个学术论文问答助手。请根据提供的论文片段，用中文回答用户的问题。\n"
        "要求：\n"
        "1. 回答要准确、简洁，基于提供的上下文\n"
        "2. 如果上下文不足以回答问题，请明确说明\n"
        "3. 回答末尾必须列出引用来源，格式为：[论文名, 页码, chunk_id]\n"
        "4. 如果问题与论文无关或无法从上下文中找到答案，请回答：'根据提供的论文内容，无法回答此问题。'"
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
    reranked = rerank(question, retrieved, models.reranker)
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
    
    sources = [
        {"paper_id": doc["paper_id"], "title": doc["title"], "page": doc["page"], "chunk_id": doc["chunk_id"]}
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
    print("🚀 论文 RAG 问答系统（全本地部署）")
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
            print(f"\n📚 引用来源:")
            for i, src in enumerate(result["sources"], 1):
                print(f"   [{i}] {src['title']} | 第{src['page']}页 | {src['chunk_id']}")
        except Exception as e:
            print(f"\n❌ 出错: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()