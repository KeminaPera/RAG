import os
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "llm.log")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def _log_json(event: str, payload: Dict[str, Any]) -> None:
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event": event,
        **payload,
    }
    logger.info(json.dumps(entry, ensure_ascii=False))


@dataclass
class LLMConfig:
    provider: str
    model: str
    base_url: str
    api_key: str
    timeout_s: int = 60


class LLMError(RuntimeError):
    pass


def load_llm_config() -> LLMConfig:
    provider = os.getenv("LLM_PROVIDER", "openai_compatible").strip().lower()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "").strip()
    # 支持 LLM_TIMEOUT_S 或 LLM_CLIENT_TIMEOUT 两个环境变量
    timeout_s = int(os.getenv("LLM_CLIENT_TIMEOUT", os.getenv("LLM_TIMEOUT_S", "120")))
    return LLMConfig(provider=provider, model=model, base_url=base_url, api_key=api_key, timeout_s=timeout_s)


def _format_messages_for_log(messages: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if content is None:
            content = ""
        snippet = content.replace("\n", " ").strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        parts.append(f"{{role={role}, content={snippet}}}")
    return " | ".join(parts)


def _openai_compatible_chat(
    cfg: LLMConfig,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> str:
    if not cfg.api_key:
        raise LLMError("缺少 LLM_API_KEY（OpenAI 兼容接口需要）")
    url = f"{cfg.base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    _log_json(
        event="llm_request_start",
        payload={
            "provider": cfg.provider,
            "model": cfg.model,
            "url": url,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": _format_messages_for_log(messages),
        },
    )
    resp = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout_s)
    logger.info("LLM request finished: status_code=%s", resp.status_code)
    if resp.status_code >= 400:
        raise LLMError(f"LLM 请求失败 {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise LLMError(f"解析 LLM 返回失败: {e}; raw={str(data)[:500]}")


def _ollama_chat(
    cfg: LLMConfig,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> str:
    # Ollama 默认: http://localhost:11434
    base = cfg.base_url or "http://localhost:11434"
    url = f"{base.rstrip('/')}/api/chat"
    payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    _log_json(
        event="llm_request_start",
        payload={
            "provider": cfg.provider,
            "model": cfg.model,
            "url": url,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": _format_messages_for_log(messages),
        },
    )
    resp = requests.post(url, json=payload, timeout=cfg.timeout_s)
    _log_json(
        event="llm_request_finished",
        payload={"status_code": resp.status_code},
    )
    if resp.status_code >= 400:
        raise LLMError(f"Ollama 请求失败 {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    try:
        return data["message"]["content"]
    except Exception as e:
        raise LLMError(f"解析 Ollama 返回失败: {e}; raw={str(data)[:500]}")


def chat_completion(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 800,
    cfg: Optional[LLMConfig] = None,
) -> str:
    cfg = cfg or load_llm_config()
    provider = (cfg.provider or "").lower()
    if provider in {"openai_compatible", "openai", "azure_openai_compatible"}:
        return _openai_compatible_chat(cfg, messages, temperature=temperature, max_tokens=max_tokens)
    if provider in {"ollama"}:
        return _ollama_chat(cfg, messages, temperature=temperature, max_tokens=max_tokens)
    raise LLMError(f"不支持的 LLM_PROVIDER: {cfg.provider}")

