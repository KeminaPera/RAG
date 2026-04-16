import sys
sys.stdout.reconfigure(encoding='utf-8')

from sentence_transformers import CrossEncoder

print("Loading model...")
model = CrossEncoder('./bge-reranker-base')
print("Model loaded successfully!")

print("\nTesting model...")
scores = model.predict([('问题', '答案'), ('你好', '你好')])
print("Test scores:", scores)
print("\n✅ BAAI/bge-reranker-base model is working correctly!")