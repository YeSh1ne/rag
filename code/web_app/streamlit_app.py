"""
RAG 问答 Web Demo（Streamlit）
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from rag_pipeline import ask, RAGModels, VECTOR_DB_DIR, COLLECTION_NAME
import chromadb
from chromadb.config import Settings


@st.cache_resource
def load_models():
    return RAGModels()


@st.cache_resource
def load_collection():
    chroma_client = chromadb.PersistentClient(
        path=VECTOR_DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return chroma_client.get_collection(COLLECTION_NAME)


st.set_page_config(
    page_title="论文 RAG 问答系统",
    page_icon="📚",
    layout="wide"
)

st.title("📚 论文 RAG 问答系统")
st.caption("基于学术论文的智能问答助手")

with st.sidebar:
    st.header("⚙️ 系统配置")
    st.info(f"""
    **向量数据库**: `{VECTOR_DB_DIR}`
    
    **Collection**: `{COLLECTION_NAME}`
    """)

if 'messages' not in st.session_state:
    st.session_state.messages = []

# 页面启动时立即加载模型（带进度提示）
if 'models' not in st.session_state:
    with st.spinner("⏳ 正在加载模型，请稍候..."):
        st.session_state.models = load_models()
        st.session_state.collection = load_collection()
        # 预加载 Embedding 和 Reranker 到 GPU，避免首次查询时延迟
        st.session_state.models.preload()
    st.success("✅ 模型加载完成！")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    with st.chat_message("assistant"):
        status = st.status("🔍 正在处理...", expanded=True)
        
        try:
            models = st.session_state.models
            collection = st.session_state.collection
            
            # Step 1: 向量检索
            status.write("📡 正在检索论文...")
            import time
            t0 = time.time()
            from rag_pipeline import retrieve, RETRIEVE_TOP_K
            retrieved = retrieve(question, collection, models.embedder)
            status.write(f"✅ 召回 {len(retrieved)} 条，耗时 {time.time()-t0:.2f}秒")
            
            # Step 2: 重排序
            status.write("🔄 正在重排序...")
            t0 = time.time()
            from rag_pipeline import rerank, RERANK_TOP_K
            reranked = rerank(question, retrieved, models.reranker, embedder=models.embedder)
            status.write(f"✅ 重排序完成，耗时 {time.time()-t0:.2f}秒")
            
            # Step 3: 构建 Prompt
            status.write("📝 正在构建 Prompt...")
            from rag_pipeline import build_prompt
            messages = build_prompt(question, reranked)
            
            # Step 4: LLM 生成
            status.write("💬 正在生成回答...")
            t0 = time.time()
            answer = models.generate(messages)
            status.write(f"✅ 生成完成，耗时 {time.time()-t0:.2f}秒")
            
            status.update(label="✅ 回答生成完成", state="complete", expanded=False)
            
            # 提取并分离引用，确保回答正文干净
            import re
            
            # 匹配所有引用模式：来自: [xxx] 或 来自：[xxx]
            citation_pattern = r'来自[：:]\s*\[[^\]]+\]'
            citations = re.findall(citation_pattern, answer)
            
            # 从回答中移除所有引用
            clean_answer = re.sub(r'\s*' + citation_pattern, '', answer)
            clean_answer = re.sub(r'\n{2,}', '\n\n', clean_answer)  # 清理多余空行
            clean_answer = clean_answer.strip()
            
            # 如果有引用，统一添加到末尾，确保每个引用单独一行
            if citations:
                clean_answer += '\n\n---\n\n**参考来源：**\n\n'
                # 每个引用单独一行，使用列表格式
                for citation in citations:
                    clean_answer += f'- {citation}\n'
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": clean_answer,
                "sources": reranked[:5]  # 保存来源列表
            })
            
            # 直接渲染回答，避免重新加载页面导致滚动问题
            #st.markdown(clean_answer)
            st.rerun()
        except Exception as e:
            status.update(label="❌ 出错", state="error", expanded=True)
            st.error(f"错误: {str(e)}")
            import traceback
            st.code(traceback.format_exc())