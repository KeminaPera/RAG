import sys
sys.stdout.reconfigure(encoding='utf-8')
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/rag.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

import chromadb
from chromadb.config import Settings
from embeddings import EmbeddingFactory
from rag_service import retrieve_and_rerank, answer_with_rag
from langchain_core.documents import Document

print("测试 RAG 流程...")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(
    path="./data/chromadb",
    settings=Settings(anonymized_telemetry=False)
)

# Get or create collection
try:
    collection = chroma_client.get_collection("documents")
except:
    collection = chroma_client.create_collection(
        name="documents",
        embedding_function=EmbeddingFactory.get_embedding(embed_type="bge_zh")
    )

# Wrapper to provide similarity_search interface
class ChromaDBWrapper:
    def __init__(self, collection):
        self.collection = collection
    
    def similarity_search(self, query, k=5):
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas"]
        )
        
        docs = []
        if results['documents'] and results['documents'][0]:
            for i, content in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                docs.append(Document(page_content=content, metadata=metadata))
        return docs

db = ChromaDBWrapper(collection)

test_query = "Java开发经验"
print(f"\n测试查询: {test_query}")

print("\n1. 开始检索并重排序...")
docs = retrieve_and_rerank(test_query, db, top_k=5)
print(f"检索完成，获取到 {len(docs)} 条文档")

if docs:
    print("\n2. 开始生成回答...")
    result = answer_with_rag(test_query, docs)
    print(f"回答生成完成")
    print("\n📝 回答:")
    print(result.answer[:200] + "..." if len(result.answer) > 200 else result.answer)
    
    print("\n📚 参考来源:")
    for src in result.sources[:3]:
        print(f"  [{src['id']}] {src['source_file'] or '未知'}")
else:
    print("未找到相关文档")

print("\n✅ 测试完成！")