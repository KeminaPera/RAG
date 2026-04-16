import os
import sys
from typing import List, Tuple, Any
from logging_config import get_logger

logger = get_logger(__name__)

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    raise RuntimeError("请安装 sentence-transformers: pip install sentence-transformers")

class BGEReranker:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
            cls._instance._load_model()
        return cls._instance
    
    def _load_model(self):
        model_path = os.getenv('RERANKER_MODEL_PATH', './bge-reranker-base')
        if not os.path.exists(model_path):
            raise RuntimeError(f"Reranker 模型路径不存在: {model_path}")
        logger.info(f"正在加载 Reranker 模型: {model_path}")
        self.model = CrossEncoder(model_path)
        logger.info("Reranker 模型加载成功")
    
    def rerank(self, query: str, candidates: List[Any], top_k: int = 5) -> List[Any]:
        if not candidates:
            logger.info("Rerank: 候选文档为空")
            return []
        
        logger.info(f"Rerank: 输入查询长度={len(query)}, 候选文档数={len(candidates)}, top_k={top_k}")
        
        pairs = [(query, doc.page_content) for doc in candidates]
        
        logger.info("Rerank: 开始计算相关性分数...")
        scores = self.model.predict(pairs)
        logger.info(f"Rerank: 分数计算完成，分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
        
        scored_docs = list(zip(candidates, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (doc, score) in enumerate(scored_docs[:top_k], 1):
            meta = getattr(doc, 'metadata', {})
            source = meta.get('source_file', meta.get('source', '未知'))
            content_snippet = doc.page_content[:50] + "..." if len(doc.page_content) > 50 else doc.page_content
            logger.info(f"Rerank: 排序 #{i} | 分数={score:.4f} | 来源={source} | 内容预览={content_snippet}")
        
        result = [doc for doc, score in scored_docs[:top_k]]
        logger.info(f"Rerank: 返回 {len(result)} 条排序后文档")
        
        return result