from embeddings import EmbeddingFactory
emb = EmbeddingFactory.get_embedding('bge_zh')
print(type(emb).__name__)
print('fallback', getattr(emb, '_fallback_mode', False))
print('reason', getattr(emb, '_fallback_reason', None))
