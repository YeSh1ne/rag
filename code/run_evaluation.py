"""
RAG 评测运行脚本
"""

from rag_pipeline import ask, RAGModels, EMBEDDING_MODEL, CHUNK_SIZE, VECTOR_DB_DIR, COLLECTION_NAME
import chromadb
from chromadb.config import Settings
from evaluator import ContentBasedRAGEvaluator


def main():
    similarity_threshold = 0.75
    citation_similarity_threshold = 0.75
    
    # 打印当前配置
    print(f"📋 当前配置:")
    print(f"   Embedding 模型: {EMBEDDING_MODEL}")
    print(f"   Chunk Size: {CHUNK_SIZE}")
    print(f"   向量数据库: {VECTOR_DB_DIR}")
    print(f"   Collection: {COLLECTION_NAME}")
    print()
    
    # 初始化模型
    models = RAGModels()
    
    # 初始化向量数据库（自动根据配置生成路径）
    chroma_client = chromadb.PersistentClient(
        path=VECTOR_DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_collection(COLLECTION_NAME)
    
    # RAG pipeline 包装函数
    def rag_pipeline_wrapper(question: str) -> dict:
        result = ask(question, models, collection)
        return {
            'answer': result.get('answer', ''),
            'sources': result.get('sources', []),
            'context_docs': result.get('context_docs', [])  # 重排后的结果（用于评估）
        }
    
    # 初始化评测器
    evaluator = ContentBasedRAGEvaluator(
        test_excel="测试集.xlsx",
        similarity_threshold=similarity_threshold,  # 可根据实际情况调整
        citation_similarity_threshold = citation_similarity_threshold,
        auto_scoring=True,  # 启用 LLM 自动评分
        chroma_collection=collection  # 传入向量数据库，用于引用验证
    )
    
    # 复用 RAG pipeline 中的 LLM 作为评分模型
    evaluator.scoring_model = models
    
    # 运行评测
    results_df = evaluator.run_evaluation(
        rag_pipeline_func=rag_pipeline_wrapper,
        output_file="evaluation_results.xlsx"
    )


if __name__ == "__main__":
    main()