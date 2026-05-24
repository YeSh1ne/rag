import os
import re
import time
import requests

# 自定义保存路径
save_dir = r"E:\桌面\深圳大学项目\论文"
os.makedirs(save_dir, exist_ok=True)

# 链接-自定义文件名映射，按需修改名称即可
paper_list = [
    ("2025.acl-long.140", "A Survey of Post-Training Scaling in Large Language Models"),
    ("2025.acl-long.1476", "Astute RAG"),
    ("2025.acl-long.366", "How to Train Long-Context Language Models (Effectively)"),
    ("2025.acl-long.1574", "CiteEval"),
    ("2025.acl-long.131", "MAIN-RAG"),
    ("2025.acl-long.176", "Hierarchical Document Refinement for Long-context Retrieval-augmented Generation"),
    ("2025.acl-long.179", "RAG-Critic"),
    ("2025.acl-long.693", "UniRAG"),
    ("2025.acl-long.1062", "FaithfulRAG"),
    ("2025.acl-long.230", "SafeRAG"),
    ("2025.acl-long.250", "Pandora s Box or Aladdin s Lamp"),
    ("2025.acl-long.358", "DRAG"),
    ("2025.acl-long.629", "RankCoT"),
    ("2025.acl-long.861", "Shifting from Ranking to Set Selection for Retrieval Augmented Generation"),
    ("2025.acl-long.929", "KiRAG"),
    ("2025.acl-long.418", "RAGEval"),
    ("2025.acl-long.1101", "MEMERAG"),
    ("2025.acl-long.1162", "Ref-Long"),
    ("2025.acl-long.263", "L-CiteEval"),
    ("2025.acl-long.490", "Think&Cite"),
    ("2025.acl-long.828", "On Synthesizing Data for Context Attribution in Question Answering"),
    ("2025.acl-long.746", "LAQuer"),
    ("2025.acl-long.71", "HALoGEN"),
    ("2025.acl-long.349", "Beyond Facts: Evaluating Intent Hallucination in Large Language Models"),
    ("2025.acl-long.826", "Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization"),
    ("2025.acl-long.183", "LongBench v2"),
    ("2025.acl-long.1538", "Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models"),
    ("2025.acl-long.275", "Self-Taught Agentic Long Context Understanding"),
    ("2025.acl-long.1355", "AgentGym"),
    ("2025.acl-long.1481", "Meta-Tool"),
    ("2025.acl-long.1383", "Agentic Reasoning"),
    ("2025.acl-long.468", "KG-Agent"),
    ("2025.acl-long.438", "CypherBench"),
    ("2025.acl-long.1528", "REAL-MM-RAG"),
    ("2025.acl-long.180", "Progressive Multimodal Reasoning via Active Retrieval"),
    ("2025.acl-long.1178", "Asclepius"),
    ("2025.acl-long.540", "Knowledge-Augmented Multimodal Clinical Rationale Generation for Disease Diagnosis with Small Language Models"),
    ("2025.acl-long.217", "QAEncoder"),
    ("2025.acl-long.127", "Smarter, Better, Faster, Longer"),
    ("2025.acl-long.57", "LongDocURL")
]

headers = {"User-Agent": "Mozilla/5.0"}

def sanitize_filename(name):
    # 移除 Windows 文件名非法字符: < > : " / \ | ? *
    return re.sub(r'[<>:"/\\|?*]', '', name)

for idx, (suffix, title) in enumerate(paper_list, 1):
    url = f"https://aclanthology.org/{suffix}.pdf"
    safe_name = sanitize_filename(title)
    save_path = os.path.join(save_dir, f"{suffix}_{safe_name}.pdf")
    
    # 跳过已存在的文件
    if os.path.exists(save_path):
        print(f"⏭️ 跳过（已存在）：{safe_name} [{idx}/{len(paper_list)}]")
        continue
    
    print(f"📥 正在下载 [{idx}/{len(paper_list)}]：{safe_name}")
    
    # 重试机制（最多3次）
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                # 验证是否为有效PDF（检查文件头）
                if resp.content[:5] == b'%PDF-':
                    with open(save_path, "wb") as f:
                        f.write(resp.content)
                    print(f"✅ 完成：{safe_name}")
                    break
                else:
                    print(f"❌ 非PDF文件：{url}")
                    break
            else:
                print(f"❌ 状态码 {resp.status_code}：{url}")
                break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️ 尝试 {attempt+1}/{max_retries} 失败，2秒后重试：{e}")
                time.sleep(2)
            else:
                print(f"❌ 下载失败 {title}：{e}")
    
    # 礼貌延迟，避免被封IP
    time.sleep(1)