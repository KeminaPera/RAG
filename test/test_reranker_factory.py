"""
测试 Reranker 工厂模式 - 验证两种类型切换
"""
import os
import sys

# 设置工作目录
os.chdir(r"e:\idea_workspace\RAG")

print("=" * 60)
print("测试 Reranker 工厂模式")
print("=" * 60)

# 测试 1: BGE Cross-Encoder
print("\n测试 1: BGE Cross-Encoder Reranker")
print("-" * 60)

os.environ['RERANKER_TYPE'] = 'bge_cross_encoder'

from reranker import RerankerFactory, BaseReranker

try:
    reranker_bge = RerankerFactory.get_reranker()
    print(f"✅ 创建成功: {type(reranker_bge).__name__}")
    print(f"✅ 是 BaseReranker 实例: {isinstance(reranker_bge, BaseReranker)}")
    print(f"✅ 支持类型: {RerankerFactory.get_supported_types()}")
except Exception as e:
    print(f"❌ 创建失败: {e}")

# 测试 2: NoOp Reranker
print("\n测试 2: NoOp Reranker")
print("-" * 60)

os.environ['RERANKER_TYPE'] = 'noop'

try:
    reranker_noop = RerankerFactory.get_reranker()
    print(f"✅ 创建成功: {type(reranker_noop).__name__}")
    print(f"✅ 是 BaseReranker 实例: {isinstance(reranker_noop, BaseReranker)}")
except Exception as e:
    print(f"❌ 创建失败: {e}")

# 测试 3: 验证单例模式
print("\n测试 3: 验证单例模式")
print("-" * 60)

reranker_bge_2 = RerankerFactory.get_reranker('bge_cross_encoder')
reranker_noop_2 = RerankerFactory.get_reranker('noop')

print(f"BGE 实例相同: {reranker_bge is reranker_bge_2}")
print(f"NoOp 实例相同: {reranker_noop is reranker_noop_2}")

if reranker_bge is reranker_bge_2 and reranker_noop is reranker_noop_2:
    print("✅ 单例模式验证通过")
else:
    print("❌ 单例模式验证失败")

# 测试 4: 不支持的类型
print("\n测试 4: 不支持的类型")
print("-" * 60)

try:
    invalid_reranker = RerankerFactory.get_reranker('invalid_type')
    print("❌ 应该抛出异常但没有")
except ValueError as e:
    print(f"✅ 正确抛出 ValueError: {str(e)[:80]}...")
except Exception as e:
    print(f"❌ 抛出异常类型错误: {type(e).__name__}")

print("\n" + "=" * 60)
print("✅ 所有测试完成！")
print("=" * 60)
