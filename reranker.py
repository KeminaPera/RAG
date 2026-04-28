"""
Reranker 模块 - 工厂模式实现
支持多种重排序算法，模型启动时加载一次，后续复用
"""
import os
from abc import ABC, abstractmethod
from typing import List, Any
from logging_config import get_logger

# CRITICAL: Disable parallelism to prevent segmentation fault on Windows
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

logger = get_logger(__name__)


# ====================== 1. 抽象基类（所有 Reranker 实现共用）======================
class BaseReranker(ABC):
    """重排序器基类：统一接口、统一调用规范"""
    
    @abstractmethod
    def rerank(self, query: str, candidates: List[Any], top_k: int = 5) -> List[Any]:
        """
        对候选文档进行重排序
        
        Args:
            query: 查询文本
            candidates: 候选文档列表（LangChain Document 对象）
            top_k: 返回前 k 个最相关的文档
            
        Returns:
            重排序后的文档列表
        """
        pass


# ====================== 2. BGE Cross-Encoder 重排序器 ======================
class BGEReranker(BaseReranker):
    """
    BGE Cross-Encoder 重排序器
    使用 BAAI/bge-reranker-base 模型进行精准相关性排序
    特点：精度高，速度中等，需要加载模型
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_path: str = None):
        # Only load model once, even if __init__ is called multiple times
        if not BGEReranker._initialized:
            self.model = None
            self.model_path = model_path or os.getenv('RERANKER_MODEL_PATH', './bge-reranker-base')
            self._load_model()
            BGEReranker._initialized = True
            logger.info("BGEReranker 单例初始化完成")
    
    def _load_model(self):
        """加载 Cross-Encoder 模型（仅调用一次）"""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise RuntimeError("请安装 sentence-transformers: pip install sentence-transformers")
        
        if not os.path.exists(self.model_path):
            raise RuntimeError(f"Reranker 模型路径不存在: {self.model_path}")
        
        logger.info(f"正在加载 BGE Reranker 模型: {self.model_path}")
        self.model = CrossEncoder(self.model_path)
        logger.info("BGE Reranker 模型加载成功")
    
    def rerank(self, query: str, candidates: List[Any], top_k: int = 5) -> List[Any]:
        """使用 Cross-Encoder 进行重排序"""
        if not candidates:
            logger.info("Rerank: 候选文档为空")
            return []
        
        # Ensure model is loaded before using
        if self.model is None:
            logger.warning("Rerank: 模型未加载，正在加载...")
            self._load_model()
        
        logger.info(f"BGE Rerank: 查询长度={len(query)}, 候选文档数={len(candidates)}, top_k={top_k}")
        
        # 构建 (query, document) 对
        pairs = [(query, doc.page_content) for doc in candidates]
        
        # 计算相关性分数（使用 batch_size 优化性能）
        logger.info("BGE Rerank: 开始计算相关性分数...")
        scores = self.model.predict(
            pairs,
            batch_size=8,  # 优化 batch size 提升 CPU 性能
            show_progress_bar=False  # 禁用进度条减少日志噪音
        )
        logger.info(f"BGE Rerank: 分数计算完成，分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # 排序并返回 top_k
        scored_docs = list(zip(candidates, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 记录排序结果
        for i, (doc, score) in enumerate(scored_docs[:top_k], 1):
            meta = getattr(doc, 'metadata', {})
            source = meta.get('source_file', meta.get('source', '未知'))
            content_snippet = doc.page_content[:50] + "..." if len(doc.page_content) > 50 else doc.page_content
            logger.info(f"BGE Rerank: #{i} | 分数={score:.4f} | 来源={source} | 预览={content_snippet}")
        
        result = [doc for doc, score in scored_docs[:top_k]]
        logger.info(f"BGE Rerank: 返回 {len(result)} 条排序后文档")
        
        return result


# ====================== 3. NoOp Reranker（不做重排序）======================
class NoOpReranker(BaseReranker):
    """
    无操作重排序器
    直接返回原始检索结果，不做任何重排序
    特点：速度极快，适用于测试或不需要重排序的场景
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not NoOpReranker._initialized:
            logger.info("NoOpReranker 单例初始化完成（跳过重排序）")
            NoOpReranker._initialized = True
    
    def rerank(self, query: str, candidates: List[Any], top_k: int = 5) -> List[Any]:
        """直接返回 top_k 个候选文档，不做重排序"""
        if not candidates:
            logger.info("NoOp Rerank: 候选文档为空")
            return []
        
        logger.info(f"NoOp Rerank: 直接返回前 {top_k} 个候选文档（共 {len(candidates)} 个）")
        
        result = candidates[:top_k]
        
        # 记录返回结果
        for i, doc in enumerate(result, 1):
            meta = getattr(doc, 'metadata', {})
            source = meta.get('source_file', meta.get('source', '未知'))
            content_snippet = doc.page_content[:50] + "..." if len(doc.page_content) > 50 else doc.page_content
            logger.info(f"NoOp Rerank: #{i} | 来源={source} | 预览={content_snippet}")
        
        return result


# ====================== 4. 工厂类（统一创建 Reranker 实例）======================
class RerankerFactory:
    """
    Reranker 工厂类
    根据配置创建对应的重排序器，内部使用单例模式确保模型只加载一次
    """
    
    # 支持的 Reranker 类型映射
    _reranker_classes = {
        'bge_cross_encoder': BGEReranker,
        'noop': NoOpReranker,
    }
    
    @staticmethod
    def get_reranker(reranker_type: str = None, **kwargs) -> BaseReranker:
        """
        获取 Reranker 实例（单例）
        
        Args:
            reranker_type: Reranker 类型，支持：
                - 'bge_cross_encoder': BGE Cross-Encoder 模型（默认）
                - 'noop': 不做重排序，直接返回
            **kwargs: 额外参数传递给 Reranker 构造函数
            
        Returns:
            BaseReranker 实例
            
        Raises:
            ValueError: 不支持的 Reranker 类型
        """
        # 从环境变量读取类型（如果未指定）
        if reranker_type is None:
            reranker_type = os.getenv('RERANKER_TYPE', 'bge_cross_encoder').strip().lower()
        
        # 验证类型
        if reranker_type not in RerankerFactory._reranker_classes:
            supported = ', '.join(RerankerFactory._reranker_classes.keys())
            raise ValueError(
                f"不支持的 Reranker 类型: {reranker_type}\n"
                f"支持的类型: {supported}"
            )
        
        # 获取对应的类并创建实例（单例）
        reranker_class = RerankerFactory._reranker_classes[reranker_type]
        
        try:
            instance = reranker_class(**kwargs)
            logger.info(f"RerankerFactory: 创建 {reranker_type} 实例成功")
            return instance
        except Exception as e:
            logger.error(f"RerankerFactory: 创建 {reranker_type} 实例失败: {e}")
            raise
    
    @staticmethod
    def get_supported_types() -> List[str]:
        """获取支持的 Reranker 类型列表"""
        return list(RerankerFactory._reranker_classes.keys())
