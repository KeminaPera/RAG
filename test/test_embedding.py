import sys
print("Starting test...")

try:
    from embeddings import BGEZhEmbeddings
    print('Import successful')
    print("Creating instance...")
    embedding = BGEZhEmbeddings()
    print('Instantiation successful')
    print(f'Instance type: {type(embedding).__name__}')
    print(f'Dimension: {embedding.dim}')
    print("Testing embed_query...")
    result = embedding.embed_query('测试文本')
    print(f'Embedding generated successfully, length: {len(result)}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()