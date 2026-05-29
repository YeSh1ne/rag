import pandas as pd
import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .utils import parse_gold_evidence
from .metrics import calculate_recall_at_k, calculate_mrr, calculate_citation_accuracy
from .scorer import calculate_ragas_style_correctness
from .excel_handler import append_to_excel, append_averages_to_excel, print_summary


class ContentBasedRAGEvaluator:
    """
    基于内容匹配的RAG评测器
    不依赖chunk_id，通过文本相似度判断检索结果是否命中Gold Evidence
    """
    
    def __init__(self, test_excel: str, similarity_threshold: float = 0.65,
                 citation_similarity_threshold: float = 0.65,
                 auto_scoring: bool = True, scoring_model=None,
                 chroma_collection=None):
        """
        :param test_excel: 测试集Excel路径
        :param similarity_threshold: 检索评估的相似度阈值
        :param citation_similarity_threshold: 引用准确率的相似度阈值
        :param auto_scoring: 是否启用 LLM 自动评分
        :param scoring_model: 评分模型（可选，如果为 None 且 auto_scoring=True 则自动加载）
        :param chroma_collection: ChromaDB集合对象，用于根据chunk_id查询原文
        """
        self.sheets = pd.read_excel(test_excel, sheet_name=None)
        self.similarity_threshold = similarity_threshold
        self.citation_similarity_threshold = citation_similarity_threshold
        self.auto_scoring = auto_scoring
        self.scoring_model = scoring_model
        self.chroma_collection = chroma_collection  # 保存向量数据库引用
        
        # 加载语义相似度模型（用于判断内容是否匹配）
        print("🔄 加载语义相似度模型...")
        
        # 尝试从 ModelScope 下载
        sim_model_name = 'BAAI/bge-m3'
        try:
            from modelscope import snapshot_download
            print(f"📥 正在通过 ModelScope 下载模型: {sim_model_name}...")
            model_path = snapshot_download(sim_model_name, cache_dir="./model_cache")
            print(f"✅ ModelScope 下载完成，缓存至: {model_path}")
            sim_model_name = model_path
        except ImportError:
            print("⚠️ 未安装 modelscope，将使用 HuggingFace 源")
        except Exception as e:
            print(f"⚠️ ModelScope 下载失败: {e}")
            print(f"   回退到 HuggingFace 源")
        
        self.sim_model = SentenceTransformer(sim_model_name)  # 中文推荐
        # 如果是英文论文，可以用: 'sentence-transformers/all-MiniLM-L6-v2'
        
        # 合并所有问题
        self.all_questions = []
        for q_type, df in self.sheets.items():
            df['question_type'] = q_type
            self.all_questions.append(df)
        self.test_df = pd.concat(self.all_questions, ignore_index=True)
        
        # 去重问题类型（防止Excel中有重复的sheet name）
        self.question_types = list(dict.fromkeys(self.sheets.keys()))
        
        print(f"✅ 初始化完成，共 {len(self.test_df)} 个问题")
        print(f"   检索评估阈值: {similarity_threshold}")
        print(f"   自动评分: {'启用' if auto_scoring else '禁用'}")
    
    def evaluate_single_question(self, question: str, question_type: str,
                                  gold_answer: str, gold_evidence_str: str,
                                  predicted_answer: str, predicted_sources: List[Dict],
                                  reranked_chunks: List[Dict],
                                  manual_score: float = None,
                                  ragas_metrics: Dict = None) -> Dict:
        """
        评估单个问题（基于内容匹配）
        
        :param reranked_chunks: 重排后的chunk列表（用于生成回答的上下文）
        :param manual_score: 人工评分（0-1），如果提供则使用人工评分
        :param ragas_metrics: RAGas 评测结果（包含 tp/fp/fn），如果提供则直接使用
        """
        # 解析Gold Evidence
        gold_evidence_texts = parse_gold_evidence(gold_evidence_str)
        
        # 判断是否为不可回答问题
        is_unanswerable = (question_type == "不可回答" or 
                          not gold_evidence_texts or 
                          (len(gold_evidence_texts) == 1 and 
                           any(keyword in gold_evidence_texts[0].lower() 
                              for keyword in ['无法回答', '不可回答', '无', 'none', 'unanswerable'])))
        
        if is_unanswerable:
            # 不可回答类问题：只计算回答正确率
            # 判断模型是否正确拒答
            is_correct_refusal = ("无法回答此问题" in predicted_answer or 
                                 "不可回答" in predicted_answer or
                                 "无法提供" in predicted_answer or
                                 "没有足够信息" in predicted_answer)
            
            # 如果Gold Answer也是拒答，且模型也拒答 → TP
            gold_is_refusal = ("无法回答" in str(gold_answer) or 
                              "不可回答" in str(gold_answer) or
                              "无法提供" in str(gold_answer))
            
            if gold_is_refusal and is_correct_refusal:
                # 正确拒答：TP=1, FP=0, FN=0
                correctness_score = 1.0
                tp, fp, fn = 1, 0, 0
            elif gold_is_refusal and not is_correct_refusal:
                # 模型强行回答（幻觉）：TP=0, FP=1, FN=0
                correctness_score = 0.0
                tp, fp, fn = 0, 1, 0
            elif not gold_is_refusal and is_correct_refusal:
                # 模型错误拒答：TP=0, FP=0, FN=1
                correctness_score = 0.0
                tp, fp, fn = 0, 0, 1
            else:
                # 两者都有答案，用LLM评分
                correctness_result = calculate_ragas_style_correctness(
                    self.scoring_model, question, str(gold_answer), predicted_answer
                )
                correctness_score = correctness_result['f1_score']
                tp, fp, fn = correctness_result['tp'], correctness_result['fp'], correctness_result['fn']
            
            return {
                "question": question,
                "question_type": question_type,
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "recall_at_1": None,  # 不可回答问题不计算Recall
                "recall_at_3": None,
                "recall_at_5": None,
                "mrr": None,  # 不可回答问题不计算MRR
                "citation_accuracy": None,
                "answer_correctness": manual_score if manual_score is not None else correctness_score,
                "is_unanswerable": True,
                "is_correct_refusal": is_correct_refusal,
                "tp": tp,
                "fp": fp,
                "fn": fn
            }
        
        # 可回答问题：正常计算所有指标
        # 计算重排后的Recall@k
        recall_at_k = calculate_recall_at_k(
            self.sim_model, self.similarity_threshold,
            reranked_chunks, gold_evidence_texts, k_values=[1, 3, 5]
        )
        
        # 计算重排后的MRR
        mrr = calculate_mrr(
            self.sim_model, self.similarity_threshold,
            reranked_chunks, gold_evidence_texts
        )
        
        # 计算引用准确率（引用 vs Gold Evidence 交集判断）
        citation_accuracy, _ = calculate_citation_accuracy(
            self.sim_model, self.citation_similarity_threshold,
            self.chroma_collection,
            predicted_answer, predicted_sources, gold_evidence_texts, reranked_chunks
        )
        
        # 计算回答正确率（TP/FP/FN）
        if ragas_metrics is not None:
            # 优先使用传入的 RAGas 评测结果
            tp = ragas_metrics['tp']
            fp = ragas_metrics['fp']
            fn = ragas_metrics['fn']
        elif manual_score is not None:
            # 如果只有评分没有详细指标，使用默认值
            tp, fp, fn = 0, 0, 0
        else:
            # 都没有，重新计算
            correctness_result = calculate_ragas_style_correctness(
                self.scoring_model, question, str(gold_answer), predicted_answer
            )
            tp, fp, fn = correctness_result['tp'], correctness_result['fp'], correctness_result['fn']
        
        return {
            "question": question,
            "question_type": question_type,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "recall_at_1": recall_at_k[1],
            "recall_at_3": recall_at_k[3],
            "recall_at_5": recall_at_k[5],
            "mrr": mrr,
            "citation_accuracy": citation_accuracy,
            "answer_correctness": manual_score,
            "is_unanswerable": False,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
    
    def run_evaluation(self, rag_pipeline_func, output_file: str = "evaluation_results.xlsx",
                       manual_scoring: bool = True):
        """
        运行完整评测
        """
        total = len(self.test_df)
        print(f"\n🚀 开始评测，共 {total} 个问题...")
        print(f"   使用内容匹配模式（相似度阈值={self.similarity_threshold}）\n")
        all_results = []
        
        for idx, row in enumerate(self.test_df.iterrows()):
            idx, row = idx, row[1]
            question = row['问题']
            question_type = row['question_type']
            gold_answer = row['参考答案']
            gold_evidence = row['Gold Evidence']
            
            # 打印进度条
            progress = (idx + 1) / total
            bar_length = 40
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            percent = progress * 100
            
            print(f"\n{'='*80}")
            print(f"📊 进度: [{bar}] {percent:.1f}% ({idx+1}/{total})")
            print(f"{'='*80}")
            print(f"❓ 问题 {idx+1}/{total}: {question[:80]}{'...' if len(question) > 80 else ''}")
            print(f"{'='*80}")
            
            try:
                result = rag_pipeline_func(question)
                
                # 自动评分或人工评分
                # 不可回答问题跳过 LLM 评分，由 evaluate_single_question 内部处理
                is_unanswerable = (question_type == "不可回答" or 
                                  pd.isna(gold_evidence) or str(gold_evidence).strip() == "")
                
                if self.auto_scoring and not is_unanswerable:
                    print(f"\n{'='*80}")
                    print(f"问题 {idx+1}/{len(self.test_df)}: {question}")
                    print(f"{'='*80}")
                    print(f"\n📝 模型回答:")
                    print(result['answer'])
                    print(f"\n📖 参考答案:")
                    print(gold_answer)
                    print(f"\n🤖 LLM 自动评分中...")
                    '''
                    manual_score = self.score_answer_with_llm(
                        question, result['answer'], gold_answer
                    )
                    '''
                    # 替换为 RAGas 风格评测：
                    print(f"\n🧠 正在进行 RAGas 风格知识点拆解评测...")
                    ragas_metrics = calculate_ragas_style_correctness(
                        self.scoring_model, question, gold_answer, result['answer']
                    )
                    manual_score = ragas_metrics["f1_score"]

                    print(f"✅ 知识点统计: 命中(TP)={ragas_metrics['tp']}, 幻觉(FP)={ragas_metrics['fp']}, 遗漏(FN)={ragas_metrics['fn']}")
                    print(f"✅ RAGas F1 正确率: {manual_score}")

                    if manual_score is not None:
                        print(f"✅ 自动评分: {manual_score:.2f}")
                    else:
                        print("⚠️  自动评分失败，跳过评分")
                elif is_unanswerable:
                    # 不可回答问题，跳过评分
                    manual_score = None
                    ragas_metrics = None
                elif manual_scoring:
                    print(f"\n{'='*80}")
                    print(f"问题 {idx+1}/{len(self.test_df)}: {question}")
                    print(f"{'='*80}")
                    print(f"\n📝 模型回答:")
                    print(result['answer'])
                    print(f"\n📖 参考答案:")
                    print(gold_answer)
                    print(f"\n{'='*80}")
                    
                    while True:
                        try:
                            score_input = input("请打分 (0-1, 或 s 跳过): ").strip()
                            if score_input.lower() == 's':
                                manual_score = None
                                break
                            manual_score = float(score_input)
                            if 0 <= manual_score <= 1:
                                break
                            else:
                                print("⚠️  请输入 0-1 之间的数字")
                        except ValueError:
                            print("⚠️  请输入有效的数字")
                else:
                    manual_score = None
                    ragas_metrics = None
                
                metrics = self.evaluate_single_question(
                    question=question,
                    question_type=question_type,
                    gold_answer=gold_answer,
                    gold_evidence_str=gold_evidence,
                    predicted_answer=result['answer'],
                    predicted_sources=result.get('sources', []),
                    reranked_chunks=result.get('context_docs', []),
                    manual_score=manual_score,
                    ragas_metrics=ragas_metrics if (self.auto_scoring and not is_unanswerable) else None
                )
                
                all_results.append(metrics)
                
                # 立即追加写入文件（方便及时查看）
                append_to_excel(metrics, output_file)
                
                # 立即打印当前问题的评测指标
                print(f"\n{'='*80}")
                print(f"📊 问题 {idx+1} 评测指标（重排后）:")
                print(f"{'='*80}")
                print(f"  Recall@1:  {metrics['recall_at_1']:.4f}" if metrics['recall_at_1'] is not None else "  Recall@1:  N/A")
                print(f"  Recall@3:  {metrics['recall_at_3']:.4f}" if metrics['recall_at_3'] is not None else "  Recall@3:  N/A")
                print(f"  Recall@5:  {metrics['recall_at_5']:.4f}" if metrics['recall_at_5'] is not None else "  Recall@5:  N/A")
                print(f"  MRR:       {metrics['mrr']:.4f}" if metrics['mrr'] is not None else "  MRR:       N/A")
                if metrics.get('citation_accuracy') is not None:
                    print(f"  引用准确率: {metrics['citation_accuracy']:.4f}")
                else:
                    print(f"  引用准确率: N/A")
                if metrics['answer_correctness'] is not None:
                    print(f"  回答正确率:   {metrics['answer_correctness']:.2f}")
                print(f"{'='*80}")
                
            except Exception as e:
                print(f"\n❌ 问题 '{question[:50]}...' 评测失败: {e}")
                all_results.append({
                    "question": question,
                    "question_type": question_type,
                    "gold_answer": "",
                    "predicted_answer": "",
                    "error": str(e),
                    "recall_at_1": None, "recall_at_3": None,
                    "recall_at_5": None,
                    "mrr": None, "citation_accuracy": None,
                    "answer_correctness": None,
                    "is_unanswerable": False,
                    "tp": None, "fp": None, "fn": None
                })
        
        # 保存结果（已经逐题追加，这里再保存一份完整的用于备份）
        results_df = pd.DataFrame(all_results)
        results_df.to_excel(output_file.replace('.xlsx', '_final.xlsx'), index=False)
        
        # 在输出文件末尾添加平均值行
        append_averages_to_excel(results_df, output_file, self.question_types)
        
        # 打印汇总
        print_summary(results_df, self.question_types)
        
        return results_df