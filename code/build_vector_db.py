"""
build_vector_db.py - 阶段3：BGE-M3 Embedding + ChromaDB 向量存储

功能：
- ✅ 从 parsed_output_* 目录加载已分片的 JSON 文件
- ✅ 使用 BAAI/bge-m3 模型生成 1024 维向量
- ✅ 存入 ChromaDB（每个 chunk_size 一个 collection）
- ✅ GPU 加速 + 批量处理
- ✅ 支持增量更新（跳过已有 collection）
- ✅ 元数据：paper_id, chunk_id, page, title, text
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import json
import re
import time
from pathlib import Path
import torch
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from embedding_utils import APIEmbedder

CHUNK_DIRS = {
    "256": r"E:\rag_project\code\parsed_output_256",
    "512": r"E:\rag_project\code\parsed_output_512",
    #"1024": r"E:\rag_project\code\parsed_output_1024",
}
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"  # 本地: BAAI/bge-m3 | API: Qwen/Qwen3-Embedding-8B
USE_API_EMBEDDING = True  # True=使用硅基流动API, False=使用本地模型
SILICONFLOW_API_KEY = "sk-sikigylnjewmvxoilaeihressilakdjxgmckrsqavluinily"  # 替换为您的硅基流动API Key
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_SHORT_NAME = EMBEDDING_MODEL.split("/")[-1].lower().replace("_", "-")
BATCH_SIZE = 16
SKIP_EXISTING = True


def clean_chunk_metadata(text: str) -> str:
    """清理chunk中的元信息（标题、作者、机构、Abstract等），保留正文内容"""
    lines = text.split('\n')
    cleaned_lines = []
    in_body = False
    
    for line in lines:
        line_stripped = line.strip()
        
        if not line_stripped:
            if in_body:
                cleaned_lines.append(line)
            continue
        
        if not in_body:
            body_starters = [
                'introduction', 'abstract', '1 introduction', '1.',
                'i. introduction', '1. introduction', 'background',
                'related work', 'method', 'approach', 'proposed'
            ]
            if any(starter in line_stripped.lower() for starter in body_starters):
                in_body = True
                cleaned_lines.append(line)
            continue
        
        if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+(?:, [A-Z][a-z]+ [A-Z][a-z]+)+\s*$', line_stripped):
            continue
        if re.search(r'(?:University|Institute|Laboratory|Lab|Department|College)', line_stripped, re.IGNORECASE):
            continue
        if re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', line_stripped):
            continue
        if re.match(r'^\s*(Abstract|Keywords?)\s*:?\s*$', line_stripped, re.IGNORECASE):
            continue
        if len(cleaned_lines) == 0 and len(line_stripped) < 100:
            continue
        
        cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    return result if result else text


def load_chunks(json_dir: str) -> list[dict]:
    """从指定目录加载所有 JSON 中的 chunks"""
    all_chunks = []
    json_files = sorted(Path(json_dir).glob("*.json"))
    
    print(f"   从 {json_dir} 加载 {len(json_files)} 个 JSON 文件...")
    
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        paper_id = data["paper_id"]
        title = data.get("title", paper_id)
        
        for chunk in data.get("chunks", []):
            all_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "page": chunk["page"],
                "paper_id": chunk["paper_id"],
                "title": title,
                "token_count": chunk.get("token_count", 0),
            })
    
    return all_chunks


def build_collection(chunk_size: str, json_dir: str, model: SentenceTransformer):
    """为一个 chunk_size 构建 ChromaDB collection"""
    vector_db_dir = rf"E:\rag_project\code\vector_db\{MODEL_SHORT_NAME}\chunk_{chunk_size}"
    collection_name = f"rag_papers_{chunk_size}"
    
    print(f"\n📦 向量数据库路径: {vector_db_dir}")
    
    os.makedirs(vector_db_dir, exist_ok=True)
    chroma_client = chromadb.PersistentClient(
        path=vector_db_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    
    existing_collections = [c.name for c in chroma_client.list_collections()]
    if SKIP_EXISTING and collection_name in existing_collections:
        print(f"  ⏭️  Collection '{collection_name}' 已存在，跳过")
        coll = chroma_client.get_collection(collection_name)
        print(f"    已有 {coll.count()} 条记录")
        return
    
    print(f"\n{'='*60}")
    print(f"🔨 构建 Collection: {collection_name}")
    print(f"{'='*60}")
    
    t0 = time.time()
    chunks = load_chunks(json_dir)
    print(f"   共 {len(chunks)} 个 chunks")
    
    if not chunks:
        print("   ⚠️  没有 chunks，跳过")
        return
    
    if collection_name in existing_collections:
        chroma_client.delete_collection(collection_name)
        print(f"   🗑️  已删除旧 collection")
    
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={
            "chunk_size": chunk_size,
            "embedding_model": EMBEDDING_MODEL,
            "description": f"RAG 论文 chunks (chunk_size={chunk_size})",
        },
    )
    
    texts = [c["text"] for c in chunks]
    
    print(f"   清理chunk中的元信息（标题、作者、机构等）...")
    cleaned_texts = [clean_chunk_metadata(t) for t in texts]
    
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {
            "paper_id": c["paper_id"],
            "title": c["title"],
            "page": c["page"],
            "token_count": c["token_count"],
        }
        for c in chunks
    ]
    
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"   第一步：生成 Embedding（共 {total_batches} 批，batch_size={BATCH_SIZE}）...")
    
    all_embeddings = []
    for i in range(0, len(cleaned_texts), BATCH_SIZE):
        batch_texts = cleaned_texts[i : i + BATCH_SIZE]
        batch_embeddings = model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=BATCH_SIZE,
        ).tolist()
        all_embeddings.extend(batch_embeddings)
        
        batch_num = i // BATCH_SIZE + 1
        print(f"   [{batch_num}/{total_batches}] 已生成 {min(i + BATCH_SIZE, len(texts))}/{len(texts)} 条")
    
    print(f"\n   第二步：一次性写入 ChromaDB（{len(all_embeddings)} 条）...")
    collection.add(
        ids=ids,
        embeddings=all_embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    
    elapsed = time.time() - t0
    print(f"\n   ✅ Collection '{collection_name}' 构建完成！")
    print(f"      总记录: {collection.count()}")
    print(f"      总耗时: {elapsed:.2f} 秒")
    print(f"      速度: {len(chunks) / elapsed:.1f} chunks/秒")
    
    print("   ⏳ 等待 ChromaDB 后台 Compaction 完成...")
    import time as time_module
    time_module.sleep(3)


def main():
    print("=" * 60)
    print("🚀 阶段3：Embedding + ChromaDB 向量存储")
    print("=" * 60)
    
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = "cpu"
        print("⚠️  CPU 模式（速度较慢）")
    
    print(f"\n📦 加载 Embedding 模型: {EMBEDDING_MODEL}")
    t0 = time.time()
    
    if USE_API_EMBEDDING:
        # 使用硅基流动 API
        print(f"📡 使用 API Embedding 模型: {EMBEDDING_MODEL}")
        model = APIEmbedder(
            model_name=EMBEDDING_MODEL,
            api_key=SILICONFLOW_API_KEY,
            base_url=SILICONFLOW_BASE_URL
        )
        print(f"✅ API 客户端初始化完成")
        print(f"   向量维度: {model.get_sentence_embedding_dimension()}")
        print(f"   最大序列长度: {model.max_seq_length}")
    else:
        # 使用本地模型（原有代码保留）
        MODEL_PATH = EMBEDDING_MODEL
        try:
            from modelscope import snapshot_download
            modelscope_id = "BAAI/bge-m3"
            print(f"📥 正在通过 ModelScope 下载模型: {modelscope_id}...")
            MODEL_PATH = snapshot_download(modelscope_id, cache_dir="./model_cache")
            print(f"✅ ModelScope 下载完成，缓存至: {MODEL_PATH}")
        except ImportError:
            print("⚠️ 未安装 modelscope，将使用 HuggingFace 源")
        except Exception as e:
            print(f"⚠️ ModelScope 下载失败: {e}")
            print(f"   回退到 HuggingFace 源: {EMBEDDING_MODEL}")
            MODEL_PATH = EMBEDDING_MODEL

        print(f"\n📦 加载 Embedding 模型: {MODEL_PATH} (fp16)")
        model = SentenceTransformer(
            MODEL_PATH,
            device=device,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16},
        )
        print(f"✅ 模型加载完成，耗时: {time.time() - t0:.2f} 秒")
        print(f"   向量维度: {model.get_sentence_embedding_dimension()}")
        print(f"   最大序列长度: {model.max_seq_length}")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"   GPU 显存已占用: {allocated:.2f} GB")
    
    for chunk_size, json_dir in CHUNK_DIRS.items():
        if not os.path.isdir(json_dir):
            print(f"\n⚠️  目录不存在，跳过: {json_dir}")
            continue
        build_collection(chunk_size, json_dir, model)
    
    print("\n" + "=" * 60)
    print("🎉 阶段3完成！")
    print("=" * 60)
    print("\n📂 生成的向量数据库：")
    for chunk_size in CHUNK_DIRS.keys():
        db_path = rf"E:\rag_project\code\vector_db\{MODEL_SHORT_NAME}\chunk_{chunk_size}"
        if os.path.exists(db_path):
            print(f"   {db_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()