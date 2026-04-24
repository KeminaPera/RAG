"""
Text Splitter 模块 - 工厂模式实现
支持多种文本分割算法，包括 LangChain 内置和第三方实现
"""
import os
from typing import Any, Dict, List, Optional
from logging_config import get_logger

logger = get_logger(__name__)


# ====================== 1. LangChain 内置 TextSplitter ======================

class LangChainTextSplitters:
    """
    LangChain 内置 TextSplitter 集合
    按需导入，避免不必要的依赖
    """
    
    @staticmethod
    def get_character(chunk_size: int = 500, chunk_overlap: int = 50, **kwargs):
        """
        CharacterTextSplitter - 按字符分割
        最简单的分割方式，按固定字符数切分
        """
        from langchain_text_splitters import CharacterTextSplitter
        
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            **kwargs
        )
    
    @staticmethod
    def get_recursive(chunk_size: int = 500, chunk_overlap: int = 50, **kwargs):
        """
        RecursiveCharacterTextSplitter - 递归字符分割（推荐）
        按段落→句子→词的顺序递归分割，保持语义完整性
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        separators = kwargs.pop('separators', ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""])
        
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators,
            **kwargs
        )
    
    @staticmethod
    def get_token(chunk_size: int = 500, chunk_overlap: int = 50, **kwargs):
        """
        TokenTextSplitter - 按 Token 分割
        使用 tiktoken 编码器，适合 LLM token 限制
        """
        from langchain_text_splitters import CharacterTextSplitter
        
        encoding_name = kwargs.pop('encoding_name', 'cl100k_base')
        
        return CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=encoding_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
    
    @staticmethod
    def get_markdown(chunk_size: int = 500, chunk_overlap: int = 50, **kwargs):
        """
        MarkdownTextSplitter - Markdown 结构分割
        根据 Markdown 标题层级分割，保持文档结构
        """
        from langchain_text_splitters import MarkdownTextSplitter
        
        return MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
    
    @staticmethod
    def get_html(headers_to_split_on: List[tuple] = None, **kwargs):
        """
        HTMLHeaderTextSplitter - HTML 结构分割
        根据 HTML 标签分割，适合网页内容
        """
        from langchain_text_splitters import HTMLHeaderTextSplitter
        
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
            ]
        
        return HTMLHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            **kwargs
        )
    
    @staticmethod
    def get_python(chunk_size: int = 500, chunk_overlap: int = 50, **kwargs):
        """
        PythonCodeTextSplitter - Python 代码分割
        根据 Python 语法（类、函数）分割代码
        """
        from langchain_text_splitters import PythonCodeTextSplitter
        
        return PythonCodeTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
    
    @staticmethod
    def get_json(chunk_size: int = 500, chunk_overlap: int = 50, **kwargs):
        """
        JSONHeaderTextSplitter - JSON 结构分割
        根据 JSON 对象/数组分割
        """
        from langchain_text_splitters import JSONHeaderTextSplitter
        
        return JSONHeaderTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )


# ====================== 2. 第三方 TextSplitter ======================

class ThirdPartyTextSplitters:
    """
    第三方 TextSplitter 实现
    需要额外安装依赖
    """
    
    @staticmethod
    def get_semantic(embeddings=None, chunk_size: int = 500, **kwargs):
        """
        SemanticChunker - 语义分割
        使用 embedding 模型计算语义相似度，在语义边界处分割
        需要安装: pip install langchain-experimental
        
        Args:
            embeddings: Embedding 模型实例（如果为 None，尝试从 app 获取）
            chunk_size: 每个 chunk 的大致大小
        """
        try:
            from langchain_experimental.text_splitter import SemanticChunker
        except ImportError:
            raise ImportError(
                "SemanticChunker 需要安装 langchain-experimental: "
                "pip install langchain-experimental"
            )
        
        # 如果没有提供 embeddings，尝试从全局获取
        if embeddings is None:
            try:
                from app import get_embeddings
                embeddings = get_embeddings()
                logger.info("SemanticChunker: 从 app 模块获取 embeddings")
            except ImportError:
                raise ValueError(
                    "SemanticChunker 需要 embeddings 参数，"
                    "请传入 Embedding 实例或在 app 上下文中运行"
                )
        
        # SemanticChunker 使用 breakpoint_threshold_type 控制分割策略
        breakpoint_threshold_type = kwargs.pop('breakpoint_threshold_type', 'percentile')
        breakpoint_threshold_amount = kwargs.pop('breakpoint_threshold_amount', 0.95)
        
        return SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            **kwargs
        )
    
    @staticmethod
    def get_llm_semantic(llm=None, chunk_size: int = 500, **kwargs):
        """
        LLM Semantic Splitter - LLM 驱动语义分割
        使用 LLM 判断分割点，最智能但速度最慢
        需要安装: pip install langchain-experimental
        
        Args:
            llm: LLM 实例（如果为 None，尝试从配置创建）
        """
        try:
            from langchain_experimental.text_splitter import SemanticSplitter
        except ImportError:
            raise ImportError(
                "SemanticSplitter 需要安装 langchain-experimental: "
                "pip install langchain-experimental"
            )
        
        # 如果没有提供 LLM，尝试从配置创建
        if llm is None:
            try:
                from llm_client import load_llm_config
                from langchain_openai import ChatOpenAI
                
                cfg = load_llm_config()
                llm = ChatOpenAI(
                    model=cfg.model,
                    base_url=cfg.base_url,
                    api_key=cfg.api_key,
                    temperature=0,
                )
                logger.info("LLM Semantic Splitter: 从配置创建 LLM")
            except Exception as e:
                raise ValueError(
                    f"LLM Semantic Splitter 需要 llm 参数或有效的 .env 配置: {e}"
                )
        
        return SemanticSplitter(
            llm=llm,
            **kwargs
        )


# ====================== 3. 工厂类（统一创建 TextSplitter）======================

class TextSplitterFactory:
    """
    TextSplitter 工厂类
    根据配置创建对应的文本分割器
    
    支持的类型:
    - character: CharacterTextSplitter（简单字符分割）
    - recursive: RecursiveCharacterTextSplitter（推荐，递归字符分割）
    - token: TokenTextSplitter（Token 分割）
    - markdown: MarkdownTextSplitter（Markdown 结构分割）
    - html: HTMLHeaderTextSplitter（HTML 结构分割）
    - python: PythonCodeTextSplitter（Python 代码分割）
    - json: JSONHeaderTextSplitter（JSON 结构分割）
    - semantic: SemanticChunker（语义分割，需要 embedding）
    - llm_semantic: LLM Semantic Splitter（LLM 驱动语义分割）
    """
    
    # 支持的 TextSplitter 类型映射
    _splitter_types = {
        # LangChain 内置
        'character': 'langchain',
        'recursive': 'langchain',
        'token': 'langchain',
        'markdown': 'langchain',
        'html': 'langchain',
        'python': 'langchain',
        'json': 'langchain',
        # 第三方
        'semantic': 'third_party',
        'llm_semantic': 'third_party',
    }
    
    @staticmethod
    def get_splitter(
        splitter_type: str = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embeddings=None,
        llm=None,
        **kwargs
    ):
        """
        获取 TextSplitter 实例
        
        Args:
            splitter_type: TextSplitter 类型，支持：
                - 'character': 字符分割（默认）
                - 'recursive': 递归字符分割（推荐）
                - 'token': Token 分割
                - 'markdown': Markdown 结构分割
                - 'html': HTML 结构分割
                - 'python': Python 代码分割
                - 'json': JSON 结构分割
                - 'semantic': 语义分割（需要 embedding）
                - 'llm_semantic': LLM 语义分割（需要 LLM）
            chunk_size: 每个 chunk 的大小
            chunk_overlap: chunk 重叠大小
            embeddings: Embedding 实例（semantic 类型需要）
            llm: LLM 实例（llm_semantic 类型需要）
            **kwargs: 额外参数传递给 TextSplitter 构造函数
            
        Returns:
            TextSplitter 实例
            
        Raises:
            ValueError: 不支持的 TextSplitter 类型
            ImportError: 缺少必要的依赖
        """
        # 从环境变量读取类型（如果未指定）
        if splitter_type is None:
            splitter_type = os.getenv('TEXT_SPLITTER_TYPE', 'recursive').strip().lower()
        
        # 验证类型
        if splitter_type not in TextSplitterFactory._splitter_types:
            supported = ', '.join(TextSplitterFactory._splitter_types.keys())
            raise ValueError(
                f"不支持的 TextSplitter 类型: {splitter_type}\n"
                f"支持的类型: {supported}"
            )
        
        category = TextSplitterFactory._splitter_types[splitter_type]
        
        try:
            if category == 'langchain':
                # LangChain 内置分割器
                method_map = {
                    'character': LangChainTextSplitters.get_character,
                    'recursive': LangChainTextSplitters.get_recursive,
                    'token': LangChainTextSplitters.get_token,
                    'markdown': LangChainTextSplitters.get_markdown,
                    'html': LangChainTextSplitters.get_html,
                    'python': LangChainTextSplitters.get_python,
                    'json': LangChainTextSplitters.get_json,
                }
                
                splitter = method_map[splitter_type](
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    **kwargs
                )
                
            elif category == 'third_party':
                # 第三方分割器
                if splitter_type == 'semantic':
                    splitter = ThirdPartyTextSplitters.get_semantic(
                        embeddings=embeddings,
                        chunk_size=chunk_size,
                        **kwargs
                    )
                elif splitter_type == 'llm_semantic':
                    splitter = ThirdPartyTextSplitters.get_llm_semantic(
                        llm=llm,
                        chunk_size=chunk_size,
                        **kwargs
                    )
            else:
                raise ValueError(f"未知的分割器类别: {category}")
            
            logger.info(f"TextSplitterFactory: 创建 {splitter_type} 实例成功 "
                       f"(chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")
            return splitter
            
        except Exception as e:
            logger.error(f"TextSplitterFactory: 创建 {splitter_type} 实例失败: {e}")
            raise
    
    @staticmethod
    def get_supported_types() -> Dict[str, str]:
        """
        获取支持的 TextSplitter 类型列表
        
        Returns:
            Dict[type, category] 类型到类别的映射
        """
        return dict(TextSplitterFactory._splitter_types)
    
    @staticmethod
    def get_type_description(splitter_type: str) -> str:
        """
        获取 TextSplitter 类型的描述信息
        
        Args:
            splitter_type: 分割器类型
            
        Returns:
            描述字符串
        """
        descriptions = {
            'character': '按固定字符数分割，简单快速',
            'recursive': '递归字符分割，保持语义完整性（推荐）',
            'token': '按 Token 分割，适合 LLM token 限制',
            'markdown': '根据 Markdown 标题结构分割',
            'html': '根据 HTML 标签结构分割',
            'python': '根据 Python 语法（类/函数）分割代码',
            'json': '根据 JSON 对象/数组结构分割',
            'semantic': '基于 embedding 语义相似度分割（需要额外依赖）',
            'llm_semantic': '基于 LLM 智能判断分割点（需要额外依赖）',
        }
        
        return descriptions.get(splitter_type, '未知类型')
