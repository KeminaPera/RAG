from __future__ import annotations
import os
import json
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferMemory
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from embeddings import EmbeddingFactory

# Fix for LangChain 0.3.0 + Pydantic 2.x compatibility
# Try to rebuild model, fallback to ConversationBufferMemory if it fails
MEMORY_CLASS = None
MEMORY_CLASS_ERROR = None

try:
    # Try importing BaseCache to resolve type definitions
    from langchain_core.caches import BaseCache
    ConversationSummaryBufferMemory.model_rebuild(force=True)
    MEMORY_CLASS = ConversationSummaryBufferMemory
    print("Using ConversationSummaryBufferMemory")
except Exception as e:
    MEMORY_CLASS_ERROR = e
    # Fallback to simpler ConversationBufferMemory
    MEMORY_CLASS = ConversationBufferMemory
    print(f"Warning: ConversationSummaryBufferMemory failed ({e}), using ConversationBufferMemory instead")

@dataclass
class MemoryConfig:
    short_term_max_token_limit: int = 3000
    short_term_max_messages: int = 20
    entity_memory_db_path: str = "./data/memory/entity_memory.db"
    long_term_memory_path: str = "./data/memory/long_term"
    long_term_top_k: int = 5
    entity_search_top_k: int = 5

    @classmethod
    def from_env(cls) -> 'MemoryConfig':
        return cls(
            short_term_max_token_limit=int(os.getenv("MEMORY_SHORT_TERM_MAX_TOKENS", "3000")),
            short_term_max_messages=int(os.getenv("MEMORY_SHORT_TERM_MAX_MESSAGES", "20")),
            entity_memory_db_path=os.getenv("MEMORY_ENTITY_DB_PATH", "./data/memory/entity_memory.db"),
            long_term_memory_path=os.getenv("MEMORY_LONG_TERM_PATH", "./data/memory/long_term"),
            long_term_top_k=int(os.getenv("MEMORY_LONG_TERM_TOP_K", "5")),
            entity_search_top_k=int(os.getenv("MEMORY_ENTITY_TOP_K", "5")),
        )

class ShortTermMemory:
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.llm = self._create_llm()
        
        # Use the resolved memory class
        global MEMORY_CLASS
        if MEMORY_CLASS is None:
            raise RuntimeError("No memory class available")
        
        # Build memory with appropriate class
        common_kwargs = {
            "llm": self.llm if MEMORY_CLASS == ConversationSummaryBufferMemory else None,
            "memory_key": "chat_history",
            "return_messages": True,
        }
        
        if MEMORY_CLASS == ConversationSummaryBufferMemory:
            common_kwargs["output_key"] = "answer"
            common_kwargs["max_token_limit"] = self.config.short_term_max_token_limit
        
        # Remove None values
        common_kwargs = {k: v for k, v in common_kwargs.items() if v is not None}
        
        self.memory = MEMORY_CLASS(**common_kwargs)
    
    def _create_llm(self):
        from llm_client import load_llm_config, chat_completion
        cfg = load_llm_config()
        if cfg.provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model=cfg.model, base_url=cfg.base_url)
        else:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=cfg.model,
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                temperature=0,
                max_tokens=500,
            )
    
    def add_message(self, role: str, content: str):
        if role == "user":
            self.memory.chat_memory.add_user_message(content)
        elif role == "assistant":
            self.memory.chat_memory.add_ai_message(content)
        self._trim_history()
    
    def _trim_history(self):
        messages = self.memory.chat_memory.messages
        max_messages = self.config.short_term_max_messages
        if len(messages) > max_messages:
            self.memory.chat_memory.messages = messages[-max_messages:]
    
    def get_history(self) -> str:
        try:
            memory_vars = self.memory.load_memory_variables({})
            chat_history = memory_vars.get("chat_history", [])
            if isinstance(chat_history, list):
                history = []
                for msg in chat_history:
                    role = "user" if msg.type == "human" else "assistant"
                    history.append(f"{role}: {msg.content}")
                return "\n".join(history)
            return str(chat_history)
        except Exception as e:
            history = []
            for msg in self.memory.chat_memory.messages:
                role = "user" if msg.type == "human" else "assistant"
                history.append(f"{role}: {msg.content}")
            return "\n".join(history)
    
    def get_messages(self) -> List[Dict[str, str]]:
        messages = []
        try:
            memory_vars = self.memory.load_memory_variables({})
            chat_history = memory_vars.get("chat_history", [])
            if isinstance(chat_history, list):
                for msg in chat_history:
                    role = "user" if msg.type == "human" else "assistant"
                    messages.append({"role": role, "content": msg.content})
            return messages
        except Exception:
            for msg in self.memory.chat_memory.messages:
                role = "user" if msg.type == "human" else "assistant"
                messages.append({"role": role, "content": msg.content})
            return messages
    
    def clear(self):
        self.memory.chat_memory.clear()

@dataclass
class Entity:
    name: str
    attributes: Dict[str, str]
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

class EntityMemory:
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.db_path = self.config.entity_memory_db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        self.embedding_function = EmbeddingFactory.get_embedding(embed_type='bge_zh')
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                attributes TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
    
    def add_entity(self, name: str, attributes: Dict[str, str]):
        entity = Entity(name=name, attributes=attributes)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO entities 
            (name, attributes, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        ''', (entity.name, json.dumps(entity.attributes), 
              entity.created_at.isoformat(), entity.updated_at.isoformat()))
        conn.commit()
        conn.close()
    
    def get_entity(self, name: str) -> Optional[Entity]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM entities WHERE name = ?', (name,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return Entity(
                name=row[1],
                attributes=json.loads(row[2]),
                created_at=datetime.fromisoformat(row[3]),
                updated_at=datetime.fromisoformat(row[4])
            )
        return None
    
    def update_entity(self, name: str, attributes: Dict[str, str]):
        entity = self.get_entity(name)
        if entity:
            entity.attributes.update(attributes)
            entity.updated_at = datetime.now()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE entities SET attributes = ?, updated_at = ? WHERE name = ?
            ''', (json.dumps(entity.attributes), entity.updated_at.isoformat(), name))
            conn.commit()
            conn.close()
    
    def search_entities(self, query: str, top_k: int = None) -> List[Entity]:
        top_k = top_k or self.config.entity_search_top_k
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM entities')
        rows = cursor.fetchall()
        conn.close()
        
        entities = []
        for row in rows:
            entity = Entity(
                name=row[1],
                attributes=json.loads(row[2]),
                created_at=datetime.fromisoformat(row[3]),
                updated_at=datetime.fromisoformat(row[4])
            )
            entities.append(entity)
        
        if not entities:
            return []
        
        query_vec = self.embedding_function.embed_query(query)
        scored = []
        
        for entity in entities:
            text = f"{entity.name}: {', '.join([f'{k}={v}' for k, v in entity.attributes.items()])}"
            vec = self.embedding_function.embed_query(text)
            score = sum(a * b for a, b in zip(query_vec, vec))
            scored.append((score, entity))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]
    
    def get_all_entities(self) -> List[Entity]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM entities')
        rows = cursor.fetchall()
        conn.close()
        entities = []
        for row in rows:
            entities.append(Entity(
                name=row[1],
                attributes=json.loads(row[2]),
                created_at=datetime.fromisoformat(row[3]),
                updated_at=datetime.fromisoformat(row[4])
            ))
        return entities
    
    def format_entity_prompt(self, entity_names: Optional[List[str]] = None) -> str:
        if entity_names:
            entities = [self.get_entity(name) for name in entity_names if self.get_entity(name)]
        else:
            entities = self.get_all_entities()
        
        if not entities:
            return ""
        
        parts = []
        for entity in entities:
            attr_lines = [f"属性: {k} = {v}" for k, v in entity.attributes.items()]
            parts.append(f"实体: {entity.name}\n" + "\n".join(attr_lines))
        
        return "\n\n".join(parts)

class LongTermMemory:
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.persist_directory = self.config.long_term_memory_path
        os.makedirs(self.persist_directory, exist_ok=True)
        self.embedding_function = EmbeddingFactory.get_embedding(embed_type='bge_zh')
        self.db_file = os.path.join(self.persist_directory, "memory.json")
        self.memory_items = []
        self._load()
    
    def _load(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "r", encoding="utf-8") as f:
                    self.memory_items = json.load(f)
            except json.JSONDecodeError:
                self.memory_items = []
    
    def _save(self):
        with open(self.db_file, "w", encoding="utf-8") as f:
            json.dump(self.memory_items, f, ensure_ascii=False, indent=2)
    
    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        if metadata is None:
            metadata = {}
        
        embedding = self.embedding_function.embed_query(content)
        memory_item = {
            "content": content,
            "embedding": embedding,
            "metadata": metadata,
            "created_at": datetime.now().isoformat()
        }
        self.memory_items.append(memory_item)
        self._save()
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        top_k = top_k or self.config.long_term_top_k
        if not self.memory_items:
            return []
        
        query_vec = self.embedding_function.embed_query(query)
        query_norm = sum(v * v for v in query_vec) ** 0.5 or 1.0
        
        scored = []
        for item in self.memory_items:
            vec = item["embedding"]
            vec_norm = sum(v * v for v in vec) ** 0.5 or 1.0
            score = sum(a * b for a, b in zip(query_vec, vec)) / (query_norm * vec_norm)
            scored.append((score, item))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]
    
    def format_retrieved_memory(self, query: str, top_k: int = None) -> str:
        items = self.retrieve(query, top_k)
        if not items:
            return ""
        
        parts = []
        for idx, item in enumerate(items, 1):
            content = item["content"]
            meta = item.get("metadata", {})
            meta_str = "; ".join([f"{k}={v}" for k, v in meta.items()]) if meta else ""
            parts.append(f"[{idx}] {content}\n元数据: {meta_str}")
        
        return "\n\n".join(parts)
    
    def get_all_memories(self) -> List[Dict[str, Any]]:
        return self.memory_items

class MemoryManager:
    def __init__(self, session_id: str, config: MemoryConfig = None):
        self.session_id = session_id
        self.config = config or MemoryConfig.from_env()
        self.short_term = ShortTermMemory(self.config)
        self.entity_memory = EntityMemory(self.config)
        self.long_term = LongTermMemory(self.config)
    
    def add_short_term(self, role: str, content: str):
        self.short_term.add_message(role, content)
    
    def get_short_term_prompt(self) -> str:
        return self.short_term.get_history()
    
    def update_entity(self, entity_name: str, attributes: Dict[str, str]):
        existing = self.entity_memory.get_entity(entity_name)
        if existing:
            self.entity_memory.update_entity(entity_name, attributes)
        else:
            self.entity_memory.add_entity(entity_name, attributes)
    
    def get_entity_prompt(self, entity_names: Optional[List[str]] = None) -> str:
        return self.entity_memory.format_entity_prompt(entity_names)
    
    def add_long_term(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.long_term.add_memory(content, metadata)
    
    def get_long_term_prompt(self, query: str, top_k: int = None) -> str:
        return self.long_term.format_retrieved_memory(query, top_k)
    
    def build_prompt(self, user_query: str, user_id: str = "default_user") -> Tuple[str, str]:
        entity_prompt = self.get_entity_prompt([user_id])
        long_term_prompt = self.get_long_term_prompt(user_query)
        short_term_prompt = self.get_short_term_prompt()
        
        system_parts = []
        if entity_prompt:
            system_parts.append(f"【实体记忆】\n{entity_prompt}")
        if long_term_prompt:
            system_parts.append(f"【长期记忆】\n{long_term_prompt}")
        if short_term_prompt:
            system_parts.append(f"【短期记忆 - 对话历史】\n{short_term_prompt}")
        
        system_prompt = "\n\n".join(system_parts) if system_parts else ""
        return system_prompt, user_query
    
    def clear_short_term(self):
        self.short_term.clear()