"""
RAG 评测运行脚本
"""

from rag_pipeline import ask, RAGModels
import chromadb
from chromadb.config import Settings
from evaluator import ContentBasedRAGEvaluator


def main():
    # 初始化模型
    models = RAGModels()
    
    # 初始化向量数据库
    chroma_client = chromadb.PersistentClient(
        path="E:\\rag_project\\code\\vector_db\\bge-m3\\chunk_512",
        settings=Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_collection("rag_papers_512")
    
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
        similarity_threshold=0.70,  # 可根据实际情况调整
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