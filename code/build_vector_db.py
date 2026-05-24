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
# 必须在 import sentence_transformers 之前设置，否则无效
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import json
import time
from pathlib import Path
import torch
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ========== 配置 ==========
CHUNK_DIRS = {
    #"256": r"E:\rag_project\code\parsed_output_256",
    "512": r"E:\rag_project\code\parsed_output_512",
    #"1024": r"E:\rag_project\code\parsed_output_1024",
}
# BGE-M3 模型配置
EMBEDDING_MODEL = "BAAI/bge-m3"
# 向量库目录：同时包含模型名和 chunk_size，保证数据隔离
MODEL_SHORT_NAME = EMBEDDING_MODEL.split("/")[-1].lower().replace("_", "-")
# 注意：VECTOR_DB_DIR 会在 main() 中根据 chunk_size 动态生成
BATCH_SIZE = 16           # RTX 4060 8GB 安全批次（fp16 下实测约占用 ~3GB）
SKIP_EXISTING = True      # 跳过已有 collection


def load_chunks(json_dir: str) -> list[dict]:
    """
    从指定目录加载所有 JSON 中的 chunks
    
    Args:
        json_dir: parsed_output_* 目录路径
    
    Returns:
        所有 chunk 的列表，每个 chunk 携带 paper 级别元数据
    """
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


def build_collection(
    chunk_size: str,
    json_dir: str,
    model: SentenceTransformer,
):
    """
    为一个 chunk_size 构建 ChromaDB collection
    
    Args:
        chunk_size: 如 "256", "512", "1024"
        json_dir: 对应的 parsed_output 目录
        model: 已加载的 SentenceTransformer 模型
    """
    # 为每个 chunk_size 生成独立的数据库路径
    vector_db_dir = rf"E:\rag_project\code\vector_db\{MODEL_SHORT_NAME}\chunk_{chunk_size}"
    collection_name = f"rag_papers_{chunk_size}"
    
    print(f"\n📦 向量数据库路径: {vector_db_dir}")
    
    # 初始化 ChromaDB（持久化模式）
    os.makedirs(vector_db_dir, exist_ok=True)
    chroma_client = chromadb.PersistentClient(
        path=vector_db_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    
    # 检查是否已存在
    existing_collections = [c.name for c in chroma_client.list_collections()]
    if SKIP_EXISTING and collection_name in existing_collections:
        print(f"  ⏭️  Collection '{collection_name}' 已存在，跳过")
        # 打印统计信息
        coll = chroma_client.get_collection(collection_name)
        print(f"    已有 {coll.count()} 条记录")
        return
    
    print(f"\n{'='*60}")
    print(f"🔨 构建 Collection: {collection_name}")
    print(f"{'='*60}")
    
    # 加载 chunks
    t0 = time.time()
    chunks = load_chunks(json_dir)
    print(f"   共 {len(chunks)} 个 chunks")
    
    if not chunks:
        print("   ⚠️  没有 chunks，跳过")
        return
    
    # 删除旧 collection（如果存在）
    if collection_name in existing_collections:
        chroma_client.delete_collection(collection_name)
        print(f"   🗑️  已删除旧 collection")
    
    # 创建新 collection
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={
            "chunk_size": chunk_size,
            "embedding_model": EMBEDDING_MODEL,
            "description": f"RAG 论文 chunks (chunk_size={chunk_size})",
        },
    )
    
    # 第一步：分批生成所有 embedding（节省显存）
    texts = [c["text"] for c in chunks]
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
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_embeddings = model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=BATCH_SIZE,
        ).tolist()
        all_embeddings.extend(batch_embeddings)
        
        batch_num = i // BATCH_SIZE + 1
        print(f"   [{batch_num}/{total_batches}] 已生成 {min(i + BATCH_SIZE, len(texts))}/{len(texts)} 条")
    
    # 第二步：一次性写入 ChromaDB（避免 HNSW 索引分批写入损坏）
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
    
    # 等待 ChromaDB 后台 Compaction 完成
    print("   ⏳ 等待 ChromaDB 后台 Compaction 完成...")
    import time as time_module
    time_module.sleep(3)


def main():
    print("=" * 60)
    print("🚀 阶段3：BGE-M3 Embedding + ChromaDB 向量存储")
    print("=" * 60)
    
    # GPU 检测
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = "cpu"
        print("⚠️  CPU 模式（速度较慢）")
    
    # 优先从 ModelScope 下载模型（BGE-M3 在魔搭的路径是 BAAI/bge-m3）
    MODEL_PATH = EMBEDDING_MODEL
    try:
        from modelscope import snapshot_download
        modelscope_id = "BAAI/bge-m3"
        print(f"📥 正在通过 ModelScope 下载模型: {modelscope_id}...")
        MODEL_PATH = snapshot_download(modelscope_id, cache_dir="./model_cache")
        print(f"✅ ModelScope 下载完成，缓存至: {MODEL_PATH}")
    except ImportError:
        print("⚠️ 未安装 modelscope（pip install modelscope），将使用 HuggingFace 源")
    except Exception as e:
        print(f"⚠️ ModelScope 下载失败: {e}")
        print(f"   回退到 HuggingFace 源: {EMBEDDING_MODEL}")
        MODEL_PATH = EMBEDDING_MODEL

    # 加载 BGE-M3 模型（fp16 半精度，节省显存）
    print(f"\n📦 加载 Embedding 模型: {MODEL_PATH} (fp16)")
    t0 = time.time()
    model = SentenceTransformer(
        MODEL_PATH,
        device=device,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.float16},
    )
    # BGE-M3 最大序列长度 8192，但 chunks 远小于此，使用默认即可
    print(f"✅ 模型加载完成，耗时: {time.time() - t0:.2f} 秒")
    print(f"   向量维度: {model.get_sentence_embedding_dimension()}")
    print(f"   最大序列长度: {model.max_seq_length}")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"   GPU 显存已占用: {allocated:.2f} GB")
    
    # 为每个 chunk_size 构建 collection（每个 chunk_size 有独立的数据库）
    for chunk_size, json_dir in CHUNK_DIRS.items():
        if not os.path.isdir(json_dir):
            print(f"\n⚠️  目录不存在，跳过: {json_dir}")
            continue
        build_collection(chunk_size, json_dir, model)
    
    # 总结
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