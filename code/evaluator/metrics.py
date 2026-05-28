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
    
    :param sim_model: 语义相似度模型
    :param similarity_threshold: 检索评估的相似度阈值
    :param reranked_chunks: 重排后的chunk列表（按排名排序），每个包含'text'字段
    :param gold_evidence_texts: Gold Evidence文本列表
    :param k_values: 要计算的k值列表
    
    计算策略：
    - Recall@1: Binary Recall（0或1），表示第一个结果是否命中任意Gold Evidence
    - Recall@k (k>1): Coverage Recall（0到1），表示Top-K覆盖了多少比例的Gold Evidence
    """
    if not gold_evidence_texts:
        return {k: None for k in k_values}
    
    recall_results = {}
    
    for k in k_values:
        top_k_chunks = reranked_chunks[:k]
        
        if k == 1:
            # Binary Recall: 第一个结果是否命中任意一条Gold Evidence
            chunk_text = top_k_chunks[0].get('text', '') or top_k_chunks[0].get('content', '') if top_k_chunks else ''
            chunk_text = clean_markdown(chunk_text)
            
            is_hit = False
            for gold_text in gold_evidence_texts:
                hit, sim = check_retrieved_hit_gold(sim_model, similarity_threshold, chunk_text, [gold_text])
                if hit:
                    is_hit = True
                    break
            
            recall_results[k] = 1.0 if is_hit else 0.0
        else:
            # Coverage Recall: Top-K覆盖了多少比例的Gold Evidence
            gold_hit_flags = [False] * len(gold_evidence_texts)
            
            hits = 0
            for g_idx, gold_text in enumerate(gold_evidence_texts):
                for chunk in top_k_chunks:
                    chunk_text = chunk.get('text', '') or chunk.get('content', '')
                    chunk_text = clean_markdown(chunk_text)
                    is_hit, sim = check_retrieved_hit_gold(sim_model, similarity_threshold, chunk_text, [gold_text])
                    if is_hit:
                        gold_hit_flags[g_idx] = True
                        hits += 1
                        break
            
            recall_results[k] = hits / len(gold_evidence_texts)
    
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
                                retrieved_chunks: List[Dict]) -> Tuple[float, bool]:
    """
    计算引用准确率（F1 Score）
    从向量数据库中根据chunk_id查询原文，然后与Gold Evidence进行语义匹配
    """
    extracted_citations = extract_citations_from_answer(predicted_answer)
    
    if not extracted_citations:
        return 0.0, False
        
    if not gold_evidence_texts:
        return 0.0, True 
        
    tp, fp = 0, 0
    matched_gold_indices = set() 
    
    for citation in extracted_citations:
        chunk_id = citation.get('chunk_id', '')
        db_id = citation.get('db_id', chunk_id)  # 使用完整的数据库ID
        
        # 策略1: 优先从向量数据库中查询原文（最准确）
        src_text = get_chunk_text_from_db(chroma_collection, db_id)
        
        # 策略2: 如果向量数据库查询失败，尝试用原始chunk_id查询
        if not src_text and db_id != chunk_id:
            src_text = get_chunk_text_from_db(chroma_collection, chunk_id)
        
        # 策略3: 如果向量数据库查询失败，尝试从检索结果中找
        if not src_text:
            for chunk in retrieved_chunks:
                if chunk.get('chunk_id', '') == chunk_id or chunk.get('chunk_id', '') == db_id:
                    src_text = chunk.get('text', '')
                    break
        
        # 策略4: 如果还没找到，尝试从 predicted_sources 中获取
        if not src_text and predicted_sources:
            for source in predicted_sources:
                if source.get('chunk_id', '') == chunk_id:
                    src_text = source.get('text', '')
                    break
        
        # 与Gold Evidence进行匹配
        if src_text:
            hit_idx, sim = check_citation_hit_gold(
                sim_model, citation_similarity_threshold, src_text, gold_evidence_texts
            )
            
            if hit_idx != -1:
                if hit_idx not in matched_gold_indices:
                    tp += 1
                    matched_gold_indices.add(hit_idx)
                else:
                    fp += 1  # 重复引用同一个 Gold
            else:
                fp += 1      # 引用了无关内容
        else:
            fp += 1  # 找不到原文，视为错误引用

    fn = len(gold_evidence_texts) - len(matched_gold_indices) 

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return round(f1_score, 4), True