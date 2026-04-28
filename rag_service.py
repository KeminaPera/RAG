from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import os
import time
from llm_client import chat_completion
from logging_config import get_logger

logger = get_logger(__name__)

# Configuration constants
VECTOR_SEARCH_TOP_K = int(os.getenv('VECTOR_SEARCH_TOP_K', '10'))

try:
    from reranker import RerankerFactory, BaseReranker
    RERANKER_AVAILABLE = True
    # Preload reranker instance at module level to avoid loading during request
    _reranker_instance = None
    logger.info("Reranker 模块加载成功")
except ImportError as e:
    RERANKER_AVAILABLE = False
    _reranker_instance = None
    logger.warning(f"Reranker 模块加载失败，将跳过重排序: {e}")

def get_reranker() -> BaseReranker:
    """Get or create singleton reranker instance using Factory pattern"""
    global _reranker_instance
    if _reranker_instance is None:
        if not RERANKER_AVAILABLE:
            raise RuntimeError("Reranker 模块不可用")
        _reranker_instance = RerankerFactory.get_reranker()
        logger.info(f"Reranker 实例创建完成: {type(_reranker_instance).__name__}")
    return _reranker_instance

try:
    from memory_manager import MemoryManager
    MEMORY_AVAILABLE = True
    logger.info("Memory 模块加载成功")
except ImportError as e:
    MEMORY_AVAILABLE = False
    logger.warning(f"Memory 模块加载失败: {e}")

@dataclass
class RagAnswer:
    answer: str
    sources: List[Dict[str, Any]]

def _format_sources(docs: List[Any]) -> Tuple[str, List[Dict[str, Any]]]:
    sources: List[Dict[str, Any]] = []
    blocks: List[str] = []
    for idx, doc in enumerate(docs, start=1):
        meta = getattr(doc, "metadata", {}) or {}
        content = getattr(doc, "page_content", "") or ""
        excerpt = content.strip()
        if len(excerpt) > 1200:
            excerpt = excerpt[:1200] + "…"
        src = {
            "id": idx,
            "source_file": meta.get("source_file") or meta.get("source") or "",
            "chunk_id": meta.get("chunk_id"),
            "page": meta.get("page"),
            "metadata": meta,
            "excerpt": excerpt,
        }
        sources.append(src)
        blocks.append(
            f"[{idx}] 来源文件: {src['source_file'] or '未知'}; chunk_id: {src['chunk_id']}; page: {src['page']}\n"
            f"内容:\n{excerpt}\n"
        )
    return "\n\n".join(blocks), sources

def retrieve_and_rerank(query: str, db, top_k: int = 10) -> List[Any]:
    logger.info(f"开始检索: 查询='{query[:50]}...'")
    
    # Phase 1: ChromaDB vector search
    t0 = time.time()
    docs = db.similarity_search(query, k=VECTOR_SEARCH_TOP_K)
    t1 = time.time()
    chroma_duration = t1 - t0
    logger.info(f"[性能] ChromaDB 向量检索完成: {chroma_duration:.2f}秒, 获取到 {len(docs)} 条候选文档")
    
    if not docs:
        logger.info("检索结果为空")
        return []
    
    # Phase 2: Reranker
    if RERANKER_AVAILABLE:
        logger.info("开始重排序...")
        t2 = time.time()
        reranker = get_reranker()
        docs = reranker.rerank(query, docs, top_k=top_k)
        t3 = time.time()
        rerank_duration = t3 - t2
        logger.info(f"[性能] 重排序完成: {rerank_duration:.2f}秒, 返回 {len(docs)} 条文档")
    else:
        logger.info("Reranker 不可用，使用原始检索结果")
        docs = docs[:top_k]
        rerank_duration = 0
    
    # Summary
    total_duration = time.time() - t0
    logger.info(f"[性能总结] 总耗时: {total_duration:.2f}秒 | ChromaDB: {chroma_duration:.2f}秒 ({chroma_duration/total_duration*100:.1f}%) | Reranker: {rerank_duration:.2f}秒 ({rerank_duration/total_duration*100:.1f}%)")
    
    return docs

def answer_with_rag(question: str, docs: List[Any], memory_manager: Optional[MemoryManager] = None, user_id: str = "default_user") -> RagAnswer:
    logger.info(f"开始生成回答: 问题='{question[:50]}...', 参考文档数={len(docs)}, 记忆可用={MEMORY_AVAILABLE}")
    
    t0 = time.time()
    context_text, sources = _format_sources(docs)
    t1 = time.time()
    logger.info(f"[性能] 格式化上下文: {t1-t0:.2f}秒")

    base_system = (
        "你是严谨的企业知识库问答助手。你必须基于给定的【检索上下文】和记忆信息回答。\n"
        "如果上下文不足以支持结论，必须明确回答：无法从已上传文档中确定，并提出需要补充的信息/澄清问题。\n"
        "回答必须给出引用编号，如 [1][3]，引用必须对应上下文条目。\n"
        "禁止编造、禁止引入上下文之外的事实。\n"
        "请结合用户的实体信息、历史对话和长期知识给出个性化回答。"
    )

    memory_prompt = ""
    if memory_manager:
        entity_prompt = memory_manager.get_entity_prompt([user_id])
        long_term_prompt = memory_manager.get_long_term_prompt(question)
        short_term_prompt = memory_manager.get_short_term_prompt()
        
        memory_parts = []
        if entity_prompt:
            memory_parts.append(f"【实体记忆】\n{entity_prompt}")
        if long_term_prompt:
            memory_parts.append(f"【长期记忆】\n{long_term_prompt}")
        if short_term_prompt:
            memory_parts.append(f"【短期记忆】\n{short_term_prompt}")
        
        memory_prompt = "\n\n".join(memory_parts)

    system = base_system
    if memory_prompt:
        system = f"{base_system}\n\n{memory_prompt}"

    user = (
        f"【用户问题】\n{question}\n\n"
        f"【检索上下文】\n{context_text}\n\n"
        "【输出要求】\n"
        "1) 先给出结论性回答（简洁、可执行）。\n"
        "2) 再给出依据与引用（用 [数字] 标注）。\n"
        "3) 若信息不足，按要求拒答并给出澄清问题。\n"
        "4) 参考记忆中的用户信息和历史对话，提供个性化回答。\n"
    )

    t2 = time.time()
    logger.info(f"[性能] 准备 LLM 请求: {t2-t1:.2f}秒")
    
    content = chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=900,
    )
    
    t3 = time.time()
    llm_duration = t3 - t2
    logger.info(f"[性能] LLM 生成回答: {llm_duration:.2f}秒")
    
    logger.info(f"回答生成完成，回答长度={len(content)}")
    return RagAnswer(answer=content, sources=sources)