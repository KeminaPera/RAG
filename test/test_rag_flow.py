import sys
sys.stdout.reconfigure(encoding='utf-8')
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/rag.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

from langchain_community.vectorstores import Chroma
from embeddings import EmbeddingFactory
from rag_service import retrieve_and_rerank, answer_with_rag

print("测试 RAG 流程...")

embeddings = EmbeddingFactory.get_embedding(embed_type="bge_zh")
db = Chroma(
    persist_directory="./chromadb",
    embedding_function=embeddings,
    collection_name="documents"
)

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