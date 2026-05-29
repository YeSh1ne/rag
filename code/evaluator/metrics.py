from typing import List, Dict, Tuple
from .utils import (
    compute_similarity, check_retrieved_hit_gold, check_citation_hit_gold,
    clean_markdown, extract_citations_from_answer, get_chunk_text_from_db
)


def calculate_recall_at_k(sim_model, similarity_threshold: float,
                          reranked_chunks: List[Dict], 
                          gold_evidence_texts: List[str],
                          k_values: List[int] = [1, 3, 5, 10]) -> Dict[int, float]:
    """
    基于内容匹配计算Recall@k
    
    核心定义：若前 k 个检索结果中至少有一个 chunk 与任意一条 Gold Evidence 语义匹配，
    则判定该问题检索成功（得分为 1），否则为 0。
    
    :param sim_model: 语义相似度模型
    :param similarity_threshold: 检索评估的相似度阈值
    :param reranked_chunks: 重排后的chunk列表（按排名排序），每个包含'text'字段
    :param gold_evidence_texts: Gold Evidence文本列表
    :param k_values: 要计算的k值列表
    """
    if not gold_evidence_texts:
        return {k: None for k in k_values}
    
    recall_results = {}
    
    for k in k_values:
        top_k_chunks = reranked_chunks[:k]
        
        is_any_hit = False
        max_sim_overall = 0.0
        hit_chunk_idx = -1
        
        for c_idx, chunk in enumerate(top_k_chunks):
            chunk_text = chunk.get('text', '') or chunk.get('content', '')
            chunk_text = clean_markdown(chunk_text)
            
            for gold_text in gold_evidence_texts:
                is_hit, sim = check_retrieved_hit_gold(sim_model, similarity_threshold, chunk_text, [gold_text])
                max_sim_overall = max(max_sim_overall, sim)
                if is_hit:
                    is_any_hit = True
                    hit_chunk_idx = c_idx
                    break
            if is_any_hit:
                break
        
        chunk_preview = top_k_chunks[hit_chunk_idx].get('text', '')[:50].replace('\n', ' ') if hit_chunk_idx >= 0 else ''
        if is_any_hit:
            print(f"    ✅ [Recall@{k}] Top-{k} 命中: chunk#{hit_chunk_idx+1} sim={max_sim_overall:.4f} | {chunk_preview}...")
        else:
            print(f"    ❌ [Recall@{k}] Top-{k} 未命中: 最高sim={max_sim_overall:.4f} (阈值={similarity_threshold})")
        
        recall_results[k] = 1.0 if is_any_hit else 0.0
    
    return recall_results


def calculate_mrr(sim_model, similarity_threshold: float,
                  reranked_chunks: List[Dict], 
                  gold_evidence_texts: List[str]) -> float:
    """
    基于内容匹配计算MRR
    找到第一个命中任意Gold Evidence的chunk的排名
    """
    if not gold_evidence_texts:
        return None
    
    for rank, chunk in enumerate(reranked_chunks, start=1):
        chunk_text = chunk.get('text', '') or chunk.get('content', '')
        # 清理markdown格式，提高与Gold Evidence的匹配度
        chunk_text = clean_markdown(chunk_text)
        is_hit, sim = check_retrieved_hit_gold(sim_model, similarity_threshold, chunk_text, gold_evidence_texts)
        if is_hit:
            return 1.0 / rank
    
    return 0.0


def calculate_citation_accuracy(sim_model, citation_similarity_threshold: float,
                                chroma_collection,
                                predicted_answer: str,
                                predicted_sources: List[Dict],
                                gold_evidence_texts: List[str],
                                retrieved_chunks: List[Dict]) -> Tuple[float, Dict]:
    """
    计算引用准确率（Citation Accuracy）

    核心定义：模型生成的引用（"来自:[...]"）是否命中了Gold Evidence。
    与Faithfulness不同，此指标直接对比引用文本 vs Gold Evidence，不依赖LLM拆解。

    计算逻辑（交集判断）：
    1. 从回答中提取引用（chunk_id 等）
    2. 通过向量数据库或检索结果获取引用原文
    3. 对每个引用，判断其与任意 Gold Evidence 的语义相似度是否达标
    4. TP = 命中Gold Evidence的引用数（去重，每个Gold只算一次）
       FP = 未命中任何Gold Evidence的引用数
       FN = 未被任何引用覆盖的Gold Evidence数
    5. F1 = 2 * P * R / (P + R)

    :param sim_model: 语义相似度模型
    :param citation_similarity_threshold: 引用命中的相似度阈值
    :param chroma_collection: ChromaDB集合，用于按chunk_id查原文
    :param predicted_answer: 模型生成的回答
    :param predicted_sources: 模型返回的sources列表
    :param gold_evidence_texts: Gold Evidence文本列表（可包含N条）
    :param retrieved_chunks: 检索到的chunk列表
    :return: (f1_score, details_dict)
    """
    # 提取引用
    extracted_citations = extract_citations_from_answer(predicted_answer)

    if not extracted_citations:
        return 0.0, {"tp": 0, "fp": 0, "fn": len(gold_evidence_texts), "detail": "无引用"}

    if not gold_evidence_texts:
        return 0.0, {"tp": 0, "fp": len(extracted_citations), "fn": 0, "detail": "无Gold Evidence"}

    tp, fp = 0, 0
    matched_gold_indices = set()

    for citation in extracted_citations:
        chunk_id = citation.get("chunk_id", "")
        db_id = citation.get("db_id", chunk_id)

        # 从向量数据库查询引用原文（已包含 clean_markdown）
        src_text = get_chunk_text_from_db(chroma_collection, db_id)

        if not src_text:
            # 找不到原文，视为错误引用
            fp += 1
            continue

        # 与 Gold Evidence 进行语义相似度判断
        hit_idx, sim = check_citation_hit_gold(
            sim_model, citation_similarity_threshold, src_text, gold_evidence_texts
        )
        src_preview = src_text[:50].replace('\n', ' ')
        if hit_idx != -1:
            if hit_idx not in matched_gold_indices:
                tp += 1
                matched_gold_indices.add(hit_idx)
                print(f"    ✅ 引用[{chunk_id}] 命中Gold[{hit_idx+1}]: sim={sim:.4f} | {src_preview}...")
            else:
                print(f"    ℹ️ 引用[{chunk_id}] 重复命中Gold[{hit_idx+1}]: sim={sim:.4f}（不计分）")
            # 注意：引用命中已存在的Gold，不算FP（补充引用是合理的）
        else:
            fp += 1  # 未命中任何Gold Evidence
            print(f"    ❌ 引用[{chunk_id}] 未命中任何Gold: 最高sim={sim:.4f} (阈值={citation_similarity_threshold}) | {src_preview}...")

    fn = len(gold_evidence_texts) - len(matched_gold_indices)

    # 引用准确率只看 Precision：引用的内容是否都正确？
    # 不惩罚未覆盖的Gold（Recall），因为很难要求模型引用所有内容
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    details = {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4),
        "total_citations": len(extracted_citations),
        "total_gold": len(gold_evidence_texts),
        "matched_gold": len(matched_gold_indices),
        "detail": f"P={precision:.2f}"
    }

    return round(precision, 4), details
