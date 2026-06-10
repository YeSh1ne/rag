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
            citation_pattern = r'来自[：:]\s*\[[^\]]+\]'
            citations = re.findall(citation_pattern, answer)
            clean_answer = re.sub(r'\s*' + citation_pattern, '', answer)
            clean_answer = re.sub(r'\n{2,}', '\n\n', clean_answer).strip()
            
            if citations:
                clean_answer += '\n\n---\n\n**参考来源：**\n\n'
                for citation in citations:
                    clean_answer += f'- {citation}\n'
            
            yield f"data: {json.dumps({'type': 'answer', 'content': clean_answer})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

if __name__ == '__main__':
    init_models()
    app.run(host='0.0.0.0', port=5000, debug=False)