import re
import pandas as pd
from typing import Dict, List
from .utils import compute_similarity


def calculate_ragas_style_correctness(scoring_model, question: str, gold_answer: str, model_answer: str) -> Dict:
    """
    使用 LLM（Qwen2.5-14B）进行知识点拆解评分
    输出 TP/FP/FN 三个数字
    """
    if pd.isna(gold_answer) or str(gold_answer).strip() == "" or not model_answer:
        return {"f1_score": 0.0, "tp": 0, "fp": 0, "fn": 0}

    extraction_prompt = f"""你是一个公正的学术评测员。请对比【标准答案】和【模型回答】，输出三个数字。

【知识点的定义 - 重要】
知识点 = 标准答案中的【核心观点、关键概念、重要结论、主要方法/步骤】
- 一个完整的观点/定义/方法 = 1个知识点
- 同一观点下的并列概念/分类/举例 ≠ 独立知识点（算作同一个知识点的细节）
- 修饰性形容词/副词 ≠ 独立知识点（如"重要的"等）
- 举例、解释、背景说明 ≠ 独立知识点
- 论文名、作者、年份 ≠ 知识点
- 细节数据、具体数值 ≠ 独立知识点（除非是核心结论）

【评判标准】
TP（命中）：模型回答中正确提到的【标准答案中的】知识点数量
FP（幻觉）：模型回答中【捏造的、与标准答案明确矛盾的】错误信息数量
FN（遗漏）：【标准答案中有】但模型回答完全没提到的知识点数量

【FP判定规则 - 重要】
- 只有【完全捏造】或【与标准答案直接矛盾】才算FP
- 模型补充的合理细节、解释、举例 ≠ FP
- 论文名、引用来源、页码 ≠ FP
- 表述不同但意思相同 ≠ FP
- 同一概念的不同缩写/简称 ≠ FP
- 中英文表述同一概念 ≠ FP

【输出格式 - 重要】
按顺序输出三个整数：TP, FP, FN
用英文逗号分隔，例如：3,0,0
不要输出任何其他文字、标点符号或解释

【标准答案】
{gold_answer}

【模型回答】
{model_answer}

输出（三个数字，逗号分隔）："""

    try:
        messages = [
            {"role": "system", "content": "你是评测助手，只输出三个整数，用英文逗号分隔。例如：3,0,0"},
            {"role": "user", "content": extraction_prompt}
        ]
        
        # 调用 14B 模型评分（带重试）
        max_retries = 2
        raw_output = None
        
        for attempt in range(max_retries + 1):
            raw_output = scoring_model.generate_score(messages)
            
            # 解析：提取所有数字
            numbers = re.findall(r'\d+', raw_output)
            
            if len(numbers) >= 3:
                tp, fp, fn = int(numbers[0]), int(numbers[1]), int(numbers[2])
                break
            elif attempt < max_retries:
                print(f"⚠️ 第{attempt+1}次解析失败: '{raw_output}'，重试中...")
                messages[-1]["content"] += "\n\n【重要】只输出三个整数，用英文逗号分隔！例如：3,0,0"
            else:
                raise ValueError(f"输出格式不正确: {raw_output}")

    except Exception as e:
        print(f"⚠️ LLM 评分失败: {e}。回退到语义相似度...")
        sim = compute_similarity(scoring_model.sim_model, gold_answer, model_answer) if hasattr(scoring_model, 'sim_model') else 0.0
        return {"f1_score": sim, "tp": 0, "fp": 0, "fn": 0, "fallback": True}

    # 计算 F1 分数
    if tp == 0 and fp == 0 and fn == 0:
        f1 = 0.0
    else:
        f1 = tp / (tp + 0.5 * fp + 0.5 * fn)
        
    f1 = max(0.0, min(1.0, f1))

    return {
        "f1_score": round(f1, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


def extract_claims(scoring_model, predicted_answer: str) -> List[str]:
    """
    Claim Extraction：将模型回答拆解为原子事实
    
    :param scoring_model: LLM 评分模型
    :param predicted_answer: 模型生成的回答
    :return: 原子事实列表
    """
    if not predicted_answer or predicted_answer.strip() == "":
        return []
    
    # 快速判断：如果是拒答类回答，没有可拆解的事实
    refusal_keywords = ["无法回答", "不可回答", "无法提供", "没有足够信息"]
    if any(kw in predicted_answer for kw in refusal_keywords):
        return []
    
    extraction_prompt = f"""请将以下回答拆解为独立的原子事实（Atomic Claims）。

【规则 - 重要】
- 每个事实应该是一个不可再分的、可独立验证的陈述
- 每个事实必须是一个完整的陈述句，包含明确的主谓宾
- 一个复杂句子中的多个独立主张需要拆成多条
- 论文名、引用来源、页码、chunk_id 不算作事实
- 修饰性描述（如"重要的"、"显著的"）不算独立事实
- 举例、解释、补充说明不算独立事实
- 问题背景复述不算新事实

【拆解示例】
回答："论文提出了方法A，该方法通过引入注意力机制，在三个基准测试上提升了10%的性能。"
输出：
1. 论文提出了方法A
2. 方法A引入了注意力机制
3. 方法A在三个基准测试上提升了10%的性能

【回答】
{predicted_answer}

请按上面的格式输出，每条事实一行，用数字+英文句点开头（不要其他内容）：
1. """

    try:
        messages = [
            {"role": "system", "content": "你是一个严谨的文本分析助手。你只输出数字编号的列表，每条一行。"},
            {"role": "user", "content": extraction_prompt}
        ]
        
        raw_output = scoring_model.generate_score(messages)
        
        # 解析：匹配 "数字. 内容" 格式的行
        claims = []
        for line in raw_output.strip().split('\n'):
            line = line.strip()
            match = re.match(r'^\d+[\.\)、]\s*(.+)', line)
            if match:
                claim_text = match.group(1).strip().strip('"').strip("'")
                if claim_text and len(claim_text) > 3:
                    claims.append(claim_text)
        
        if claims:
            return claims
        
        # 备用：按句号拆解
        print(f"⚠️ Claim Extraction 编号解析失败，回退到按句号拆解...")
        sentences = re.split(r'[。；;]', predicted_answer)
        claims = [s.strip() for s in sentences if len(s.strip()) > 5]
        return claims[:10]  # 最多10条
        
    except Exception as e:
        print(f"⚠️ Claim Extraction 失败: {e}")
        return []


def verify_claims_grounding(scoring_model, claims: List[str], context_text: str) -> Dict[int, str]:
    """
    Grounding Verification：批量验证原子事实是否能从上下文中推导出来（NLI）
    
    :param scoring_model: LLM 评分模型
    :param claims: 原子事实列表
    :param context_text: 模型检索到的所有上下文（合并后的文本）
    :return: {claim_index: verdict} 其中 verdict 为 "entailment" / "neutral" / "contradiction"
    """
    if not claims:
        return {}
    
    # 限制上下文长度
    max_context_chars = 8000
    if len(context_text) > max_context_chars:
        context_text = context_text[:max_context_chars] + "\n...(上下文已截断)"
    
    # 构造带编号的 claims 列表
    claims_text = "\n".join([f"{i+1}. {claim}" for i, claim in enumerate(claims)])
    
    verification_prompt = f"""请逐个判断以下事实能否从提供的【上下文】中推导出来。

【上下文】（模型检索到的所有文本）
---
{context_text}
---

【待验证的事实列表】
{claims_text}

【判定标准】
- "entailment"：上下文明确支持该事实，或可以从上下文逻辑推导出来。
- "neutral"：上下文不包含该事实的相关信息。
- "contradiction"：上下文明确与该事实矛盾。

【输出格式 - 重要】
请按顺序输出每个事实的判定结果，每行一个（格式：编号:判定词），不要输出任何其他内容：
1:entailment
2:neutral
3:entailment"""

    try:
        messages = [
            {"role": "system", "content": "你是一个严谨的NLI评测助手。只输出'编号:判定词'格式的结果，每行一个。"},
            {"role": "user", "content": verification_prompt}
        ]
        
        raw_output = scoring_model.generate_score(messages)
        
        # 解析：匹配 "数字:verdict" 格式
        verdict_map = {}
        for line in raw_output.strip().split('\n'):
            line = line.strip()
            # 匹配格式: 1:entailment 或 1.entailment 或 1 entailment
            match = re.match(r'^(\d+)\s*[:.\s]\s*(entailment|neutral|contradiction)', line, re.IGNORECASE)
            if match:
                idx = int(match.group(1)) - 1  # 转0-index
                verdict = match.group(2).lower()
                verdict_map[idx] = verdict
            else:
                # 备选匹配：行中包含数字和关键词
                alt_match = re.search(r'(\d+).*?(entailment|neutral|contradiction)', line, re.IGNORECASE)
                if alt_match:
                    idx = int(alt_match.group(1)) - 1
                    verdict = alt_match.group(2).lower()
                    if idx not in verdict_map:
                        verdict_map[idx] = verdict
        
        if verdict_map:
            return verdict_map
        
        # 备用：在整个输出中搜索关键词按顺序匹配
        all_verdicts = re.findall(r'(entailment|neutral|contradiction)', raw_output, re.IGNORECASE)
        if len(all_verdicts) == len(claims):
            return {i: v.lower() for i, v in enumerate(all_verdicts)}
        
        print(f"⚠️ Grounding Verification 解析失败: '{raw_output[:100]}...'")
        return {}
        
    except Exception as e:
        print(f"⚠️ Grounding Verification 失败: {e}")
        return {}