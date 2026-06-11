from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_pipeline import RAGModels, VECTOR_DB_DIR, COLLECTION_NAME
import chromadb
from chromadb.config import Settings
import re

app = Flask(__name__)

models = None
collection = None
paper_title_to_filename = {}  # 原论文名 -> PDF 文件名映射

def init_paper_mapping():
    """从 parsed JSON 文件建立 title -> filename 映射"""
    global paper_title_to_filename
    import glob
    
    json_dir = r"E:\rag_project\code\parsed_output_1024"
    pdf_dir = r"E:\rag_project\code\web_app_flask\static\papers"
    
    # 获取所有 PDF 文件名
    pdf_files = set()
    for f in os.listdir(pdf_dir):
        if f.endswith('.pdf'):
            pdf_files.add(f)
    
    # 读取 JSON 文件建立映射
    for json_file in glob.glob(os.path.join(json_dir, "*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            title = data.get('title', '')
            if title:
                # 从 JSON 文件名提取 PDF 前缀（如 2025.acl-long.230_SafeRAG）
                json_basename = os.path.splitext(os.path.basename(json_file))[0]
                pdf_filename = json_basename + '.pdf'
                if pdf_filename in pdf_files:
                    paper_title_to_filename[title] = pdf_filename
        except:
            pass
    
    print(f"✅ 论文映射加载完成: {len(paper_title_to_filename)} 篇")

def init_models():
    global models, collection
    if models is None:
        print("📦 加载模型...")
        models = RAGModels()
        models.preload()
        
        chroma_client = chromadb.PersistentClient(
            path=VECTOR_DB_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = chroma_client.get_collection(COLLECTION_NAME)
        print(f"✅ 模型加载完成，Collection: {COLLECTION_NAME}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pdf-viewer')
def pdf_viewer():
    """PDF 查看器页面，支持页码跳转和文本高亮"""
    return render_template('pdf-viewer.html')

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': '问题不能为空'}), 400
    
    def generate():
        try:
            from rag_pipeline import retrieve, rerank, build_prompt
            
            # 立即发送初始事件，让前端立即显示助手消息
            yield f"data: {json.dumps({'type': 'start'})}\n\n"
            
            # Step 1: 检索
            t0 = time.time()
            yield f"data: {json.dumps({'type': 'step', 'icon': '📡', 'label': '正在检索论文', 'detail': '', 'time': ''})}\n\n"
            retrieved = retrieve(question, collection, models.embedder)
            t1 = time.time() - t0
            yield f"data: {json.dumps({'type': 'step', 'icon': '📡', 'label': '正在检索论文', 'detail': f'召回 {len(retrieved)} 条', 'time': f'{t1:.2f}秒'})}\n\n"
            
            # Step 2: 重排序
            t0 = time.time()
            yield f"data: {json.dumps({'type': 'step', 'icon': '🔄', 'label': '正在重排序', 'detail': '', 'time': ''})}\n\n"
            reranked = rerank(question, retrieved, models.reranker, embedder=models.embedder)
            t1 = time.time() - t0
            yield f"data: {json.dumps({'type': 'step', 'icon': '🔄', 'label': '正在重排序', 'detail': '重排序完成', 'time': f'{t1:.2f}秒'})}\n\n"
            
            # Step 3: 构建 Prompt
            yield f"data: {json.dumps({'type': 'step', 'icon': '📝', 'label': '正在构建 Prompt', 'detail': '', 'time': ''})}\n\n"
            messages = build_prompt(question, reranked)
            
            # Step 4: 生成
            t0 = time.time()
            yield f"data: {json.dumps({'type': 'step', 'icon': '💬', 'label': '正在生成回答', 'detail': '', 'time': ''})}\n\n"
            answer = models.generate(messages)
            t1 = time.time() - t0
            yield f"data: {json.dumps({'type': 'step', 'icon': '💬', 'label': '正在生成回答', 'detail': '生成完成', 'time': f'{t1:.2f}秒'})}\n\n"
            
            # 后处理引用
            citation_pattern = r'来自[：:]\s*\[([^\]]+)\]'
            citations = re.findall(citation_pattern, answer)
            clean_answer = re.sub(r'\s*来自[：:]\s*\[[^\]]+\]', '', answer)
            clean_answer = re.sub(r'\n{2,}', '\n\n', clean_answer).strip()
            
            # 构建 sources 信息
            sources = []
            if citations:
                clean_answer += '\n\n---\n\n**参考来源：**\n\n'
                for citation in citations:
                    # 解析引用："KiRAG: Knowledge-Driven..., 第4页, ..."
                    parts = [p.strip() for p in citation.split(',')]
                    paper_title = parts[0] if parts else ''
                    page_info = parts[1] if len(parts) > 1 else ''
                    
                    # 通过映射查找对应的 PDF 文件名
                    pdf_filename = paper_title_to_filename.get(paper_title, '')
                    
                    # 提取页码数字
                    page_num = ''
                    page_match = re.search(r'第(\d+)页', page_info)
                    if page_match:
                        page_num = page_match.group(1)
                    
                    # 构建 PDF 直链（浏览器原生查看器）
                    if pdf_filename:
                        pdf_url = f'/static/papers/{pdf_filename}'
                        if page_num:
                            pdf_url += f'#page={page_num}'
                    else:
                        pdf_url = ''
                    
                    sources.append({
                        'paper_title': paper_title,
                        'page': page_info,
                        'pdf_url': pdf_url,
                        'full_citation': citation
                    })
                    
                    clean_answer += f'- 来自: [{citation}]\n'
            
            yield f"data: {json.dumps({'type': 'answer', 'content': clean_answer, 'sources': sources})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

if __name__ == '__main__':
    init_models()
    init_paper_mapping()
    app.run(host='0.0.0.0', port=5000, debug=False)