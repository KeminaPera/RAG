from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from llm_client import chat_completion
from logging_config import get_logger

logger = get_logger(__name__)

try:
    from reranker import BGEReranker
    RERANKER_AVAILABLE = True
    logger.info("Reranker 模块加载成功")
except ImportError as e:
    RERANKER_AVAILABLE = False
    logger.warning(f"Reranker 模块加载失败，将跳过重排序: {e}")

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
    
    docs = db.similarity_search(query, k=15)
    logger.info(f"向量检索完成，获取到 {len(docs)} 条候选文档")
    
    if not docs:
        logger.info("检索结果为空")
        return []
    
    if RERANKER_AVAILABLE:
        logger.info("开始重排序...")
        reranker = BGEReranker()
        docs = reranker.rerank(query, docs, top_k=top_k)
        logger.info(f"重排序完成，返回 {len(docs)} 条文档")
    else:
        logger.info("Reranker 不可用，使用原始检索结果")
        docs = docs[:top_k]
    
    return docs

def answer_with_rag(question: str, docs: List[Any]) -> RagAnswer:
    logger.info(f"开始生成回答: 问题='{question[:50]}...', 参考文档数={len(docs)}")
    
    context_text, sources = _format_sources(docs)

    system = (
        "你是严谨的企业知识库问答助手。你必须只基于给定的【检索上下文】回答。\n"
        "如果上下文不足以支持结论，必须明确回答：无法从已上传文档中确定，并提出需要补充的信息/澄清问题。\n"
        "回答必须给出引用编号，如 [1][3]，引用必须对应上下文条目。\n"
        "禁止编造、禁止引入上下文之外的事实。"
    )
    user = (
        f"【用户问题】\n{question}\n\n"
        f"【检索上下文】\n{context_text}\n\n"
        "【输出要求】\n"
        "1) 先给出结论性回答（简洁、可执行）。\n"
        "2) 再给出依据与引用（用 [数字] 标注）。\n"
        "3) 若信息不足，按要求拒答并给出澄清问题。\n"
    )

    content = chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=900,
    )
    
    logger.info(f"回答生成完成，回答长度={len(content)}")
    return RagAnswer(answer=content, sources=sources)