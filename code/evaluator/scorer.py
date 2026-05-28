import re
import pandas as pd
from typing import Dict
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