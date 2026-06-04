"""
embedding_utils.py - Embedding 模型工具类

功能：
- 提供 APIEmbedder 类，封装硅基流动 API 调用
- 兼容 SentenceTransformer.encode() 接口
- 支持本地模型和 API 模型的统一调用
"""

import numpy as np
from openai import OpenAI


class APIEmbedder:
    """使用硅基流动 API 生成 embedding，兼容 SentenceTransformer 接口"""
    
    def __init__(self, model_name: str, api_key: str, base_url: str):
        """
        初始化 API Embedder
        
        Args:
            model_name: 模型名称，如 "Qwen/Qwen3-Embedding-8B"
            api_key: 硅基流动 API Key
            base_url: API 基础 URL，如 "https://api.siliconflow.cn/v1"
        """
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=None, **kwargs):
        """
        兼容 SentenceTransformer.encode() 接口
        
        Args:
            texts: 单条文本字符串或文本列表
            normalize_embeddings: 是否归一化（默认 True）
            show_progress_bar: 是否显示进度条（API 调用忽略此参数）
            batch_size: 批量大小（API 调用时可选）
            
        Returns:
            numpy.ndarray: embedding 向量
        """
        # 支持单条或多条文本
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False
        
        # API 调用（逐条或批量）
        all_embeddings = []
        batch_size = batch_size or len(texts)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
                encoding_format="float"
            )
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
        
        # 归一化（如果需要）
        if normalize_embeddings:
            all_embeddings = [e / np.linalg.norm(e) for e in all_embeddings]
        
        result = np.array(all_embeddings)
        
        # 单条文本返回 1D 数组
        if single:
            return result[0]
        return result
    
    def get_sentence_embedding_dimension(self):
        """获取 embedding 维度（Qwen3-Embedding-8B 默认 4096）"""
        return 4096
    
    @property
    def max_seq_length(self):
        """获取最大序列长度（Qwen3-Embedding-8B 默认 32768）"""
        return 32768