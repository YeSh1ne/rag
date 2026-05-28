import pandas as pd
import json
import re
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util


def compute_similarity(sim_model: SentenceTransformer, text1: str, text2: str) -> float:
    """
    计算两段文本的语义相似度
    """
    emb1 = sim_model.encode(text1, convert_to_tensor=True)
    emb2 = sim_model.encode(text2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    return score


def check_retrieved_hit_gold(sim_model: SentenceTransformer, 
                              similarity_threshold: float,
                              retrieved_chunk_text: str, 
                              gold_evidence_texts: List[str]) -> Tuple[bool, float]:
    """
    判断一个检索到的chunk是否命中了任意一条Gold Evidence（使用检索阈值）
    
    :param sim_model: 语义相似度模型
    :param similarity_threshold: 检索评估的相似度阈值
    :param retrieved_chunk_text: 检索到的chunk文本
    :param gold_evidence_texts: Gold Evidence文本列表（从Excel的Gold Evidence列解析）
    :return: (是否命中, 最高相似度)
    """
    if not gold_evidence_texts:
        return False, 0.0
    
    max_sim = 0.0
    for gold_text in gold_evidence_texts:
        sim = compute_similarity(sim_model, retrieved_chunk_text, gold_text)
        max_sim = max(max_sim, sim)
    
    is_hit = max_sim >= similarity_threshold
    return is_hit, max_sim


def check_citation_hit_gold(sim_model: SentenceTransformer,
                            citation_similarity_threshold: float,
                            citation_text: str, 
                            gold_evidence_texts: List[str]) -> Tuple[int, float]:
    """
    判断引用文本是否命中Gold Evidence（使用更严格的引用阈值）
    
    :param sim_model: 语义相似度模型
    :param citation_similarity_threshold: 引用准确率的相似度阈值
    :param citation_text: 引用文本
    :param gold_evidence_texts: Gold Evidence文本列表
    :return: (最佳匹配的索引, 最高相似度)
    """
    if not gold_evidence_texts:
        return -1, 0.0
    
    max_sim = 0.0
    best_match_idx = -1
    
    for idx, gold_text in enumerate(gold_evidence_texts):
        sim = compute_similarity(sim_model, citation_text, gold_text)
        if sim > max_sim:
            max_sim = sim
            best_match_idx = idx 
    
    if max_sim >= citation_similarity_threshold:
        return best_match_idx, max_sim
    else:
        return -1, max_sim


def parse_gold_evidence(gold_evidence_str: str) -> List[str]:
    """
    解析Gold Evidence列
    
    您的测试集中Gold Evidence是文本描述，可能包含多条证据
    支持格式：
    - 单条文本: "Zhang et al. (2024) proposed..."
    - 多条文本（用换行或分号分隔）: "证据1\n证据2" 或 "证据1; 证据2"
    - JSON数组: '["证据1", "证据2"]'
    """
    if pd.isna(gold_evidence_str) or str(gold_evidence_str).strip() == "":
        return []
    
    gold_str = str(gold_evidence_str).strip()
    
    # 尝试JSON解析
    try:
        parsed = json.loads(gold_str)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except:
        pass
    
    # 提取双引号包裹的内容
    quoted_parts = re.findall(r'"([^"]+)"', gold_str)
    
    if quoted_parts:
        # 有双引号，提取引号内的内容，合并内部的换行
        parts = []
        for part in quoted_parts:
            # 将引号内的换行符替换为空格，合并成一条
            cleaned = ' '.join(line.strip() for line in part.split('\n') if line.strip())
            if cleaned:
                parts.append(cleaned)
        return parts
    
    # 没有双引号，按换行或分号分隔
    if '\n' in gold_str:
        lines = gold_str.split('\n')
        # 如果每行都很短(<50字符)且行首是小写字母,说明是PDF换行,应该合并
        is_pdf_wrapping = all(
            len(line.strip()) < 50 and 
            (line.strip() == '' or line.strip()[0].islower())
            for line in lines
            if line.strip()
        )
        
        if is_pdf_wrapping:
            # PDF换行,合并成一条
            parts = [' '.join(line.strip() for line in lines)]
        else:
            # 真正的多行证据
            parts = [line.strip() for line in lines if line.strip()]
    else:
        parts = [gold_str]
    
    return [p.strip() for p in parts if p.strip()]


def clean_markdown(text: str) -> str:
    """
    清理markdown格式，保留纯文本内容
    """
    # 移除代码块
    text = re.sub(r'```[\s\S]*?```', '', text)
    # 移除行内代码
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # 移除图片标记
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
    # 移除链接，保留文本
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # 移除标题标记
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # 移除粗体/斜体标记
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    # 移除列表标记
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # 移除引用标记
    text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE)
    # 移除水平线
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # 移除多余的空行（保留单个换行）
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 去除首尾空白
    text = text.strip()
    
    return text


def extract_citations_from_answer(answer: str) -> List[dict]:
    """
    从 LLM 生成的回答中提取引用信息
    支持格式：
    - 来自: [论文名, 页码, chunk_id]
    - 来自: [论文名 (第X页, chunk_id)]
    """
    citations = []
    
    # 匹配 "来自: [xxx]" 格式
    pattern = r'来自:\s*\[([^\]]+)\]'
    matches = re.findall(pattern, answer)
    
    for match in matches:
        # 尝试匹配括号格式 "论文名 (第X页, chunk_id)"
        bracket_pattern = r'([^\(]+)\s*\(([^,]+),\s*([^\)]+)\)'
        bracket_match = re.search(bracket_pattern, match)
        if bracket_match:
            title = bracket_match.group(1).strip()
            page = bracket_match.group(2).strip()
            chunk_id = bracket_match.group(3).strip()
            # 组合成完整的数据库ID: 论文名_chunk_id
            db_id = f"{title}_{chunk_id}" if not chunk_id.startswith(title.split('_')[0]) else chunk_id
            citations.append({
                'title': title,
                'page': page,
                'chunk_id': chunk_id,
                'db_id': db_id  # 用于数据库查询的完整ID
            })
            continue
        
        # 逗号分隔格式 "论文名, 页码, chunk_id"
        parts = [p.strip() for p in match.split(',')]
        if len(parts) >= 3:
            title = parts[0]
            page = parts[1]
            chunk_id = parts[2]
            # 组合成完整的数据库ID: 论文名_chunk_id
            db_id = f"{title}_{chunk_id}" if not chunk_id.startswith(title.split('_')[0]) else chunk_id
            citations.append({
                'title': title,
                'page': page,
                'chunk_id': chunk_id,
                'db_id': db_id  # 用于数据库查询的完整ID
            })
    
    return citations


def get_chunk_text_from_db(chroma_collection, chunk_id: str) -> str:
    """
    从向量数据库中根据chunk_id查询原文
    
    :param chroma_collection: ChromaDB集合对象
    :param chunk_id: chunk的唯一标识
    :return: chunk的原始文本（markdown格式）
    """
    if chroma_collection is None:
        return ''
    
    try:
        # 使用 ChromaDB 的 get 方法根据 ID 查询
        result = chroma_collection.get(ids=[chunk_id])
        if result and result.get('documents') and len(result['documents']) > 0:
            raw_text = result['documents'][0]
            # 清理markdown格式
            return clean_markdown(raw_text)
    except Exception as e:
        print(f"⚠️ 从向量数据库查询 chunk_id={chunk_id} 失败: {e}")
    
    return ''