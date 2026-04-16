import os
from abc import ABC, abstractmethod
from typing import List, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ====================== 1. 统一抽象接口（所有模型共用）======================
class BaseEmbeddings(ABC):
    """向量嵌入基类：统一入口、统一调用规范"""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化"""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """单条查询向量化"""
        pass

    def __call__(self, text: Union[str, List[str]]):
        """统一调用入口，业务层无脑使用"""
        if isinstance(text, str):
            return self.embed_query(text)
        return self.embed_documents(text)

# ====================== 2. 你旧的 Hash 模型（保留兼容，不用删）======================
class SimpleHashEmbeddings(BaseEmbeddings):
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)

    def _embed_text(self, text):
        vector = [0.0] * self.dim
        if not text:
            return vector
        tokens = text.lower().split()
        for token in tokens:
            import hashlib
            digest = hashlib.sha256(token.encode('utf-8')).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vector[idx] += sign
        norm = sum(v * v for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

# ====================== 3. BGE 中文最优模型（本次改造核心）======================
class BGEZhEmbeddings(BaseEmbeddings):
    """
    BAAI/bge-small-zh-v1.5
    中文离线 embedding 最优小模型
    维度：512
    特点：速度快、精度高、无需 GPU、无需联网
    """
    _instance = None  # 单例模式，避免重复加载模型

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._fallback_mode = False
            cls._instance._fallback_reason = None
            cls._instance._fallback_model = None
            cls._instance.model = None
            cls._instance.dim = 512
            try:
                cls._instance._load_model()
                if not hasattr(cls._instance, 'model') or cls._instance.model is None:
                    raise RuntimeError('BGE 模型加载失败：未能正确初始化 model 对象')
                logger.info("BGE 模型加载成功")
            except Exception as e:
                cls._instance._fallback_mode = True
                cls._instance._fallback_reason = str(e)
                cls._instance._fallback_model = SimpleHashEmbeddings(dim=512)
                logger.warning("无法加载 BGE 模型，回退到 Hash embedding: %s", e)
        return cls._instance

    def _load_model(self):
        """模型只加载一次，生产环境必备"""
        try:
            from sentence_transformers import SentenceTransformer
            model_path = os.getenv('BGE_LOCAL_MODEL_PATH', '').strip()
            if model_path:
                if not os.path.exists(model_path):
                    raise RuntimeError(f'本地 BGE 模型路径不存在: {model_path}')
                self.model = SentenceTransformer(model_path)
            else:
                self.model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
            self.dim = 512  # 固定维度 512
        except ImportError as e:
            raise RuntimeError(f"缺少 sentence-transformers 依赖: {e}")
        except Exception as e:
            raise RuntimeError(f"BGE 模型加载失败: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量向量化：自动归一化，适合检索"""
        if not texts:
            return []

        if self._fallback_mode or not hasattr(self, 'model') or self.model is None:
            if self._fallback_model is None:
                self._fallback_model = SimpleHashEmbeddings(dim=512)
            return self._fallback_model.embed_documents(texts)

        texts = [t.strip().replace("\n", " ") for t in texts]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=16
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """单条查询向量化"""
        if self._fallback_mode or not hasattr(self, 'model') or self.model is None:
            if self._fallback_model is None:
                self._fallback_model = SimpleHashEmbeddings(dim=512)
            return self._fallback_model.embed_query(text)

        text = text.strip().replace("\n", " ")
        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )
        return embedding.tolist()

# ====================== 4. 统一工厂（业务代码只调用这里）======================
class EmbeddingFactory:
    @staticmethod
    def get_embedding(embed_type: str = "bge_zh", **kwargs) -> BaseEmbeddings:
        """
        统一获取向量模型
        支持：hash / bge_zh
        """
        if embed_type == "hash":
            return SimpleHashEmbeddings(**kwargs)
        elif embed_type == "bge_zh":
            return BGEZhEmbeddings(**kwargs)
        else:
            raise ValueError(f"不支持的Embedding类型: {embed_type}")