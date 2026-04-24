"""  
测试 TextSplitter 工厂模式 - 验证多种分割器类型
"""
import os
import sys

# 设置工作目录和 Python 路径
os.chdir(r"e:\idea_workspace\RAG")
sys.path.insert(0, r"e:\idea_workspace\RAG")

print("=" * 70)
print("测试 TextSplitter 工厂模式")
print("=" * 70)

# 测试文档
test_text = """
第一章：人工智能概述

人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个分支，
旨在创建能够执行需要人类智能的任务的系统。这些任务包括学习、推理、
问题解决、感知和语言理解。

1.1 机器学习基础

机器学习是人工智能的核心技术之一。它使计算机能够从数据中学习，
而不需要显式编程。主要类型包括：

- 监督学习：从标记数据中学习
- 无监督学习：从未标记数据中发现模式
- 强化学习：通过与环境交互学习最优策略

1.2 深度学习

深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的
分层表示。它在以下领域取得了突破性进展：

1. 计算机视觉
2. 自然语言处理
3. 语音识别

第二章：自然语言处理

自然语言处理（NLP）是人工智能和语言学的交叉领域，旨在使计算机
能够理解、解释和生成人类语言。

2.1 文本预处理

文本预处理是 NLP 的基础步骤，包括：
- 分词（Tokenization）
- 去除停用词
- 词干提取和词形还原
- 词性标注

2.2 词向量表示

词向量将单词映射到连续的向量空间，使得语义相似的词在空间中距离
较近。常用方法包括：

1. Word2Vec
2. GloVe
3. BERT 等预训练模型

第三章：实际应用

3.1 智能客服

基于 NLP 技术的智能客服系统能够：
- 理解用户问题
- 检索相关知识
- 生成自然语言回答

3.2 文档分析

文档分析系统可以：
- 提取关键信息
- 进行分类和标注
- 生成摘要
"""

# 测试 1: 查看支持的类型
print("\n测试 1: 查看支持的 TextSplitter 类型")
print("-" * 70)

from text_splitter import TextSplitterFactory

supported_types = TextSplitterFactory.get_supported_types()
print(f"✅ 支持的类型数量: {len(supported_types)}")
print(f"\n📋 类型列表:")
for splitter_type, category in supported_types.items():
    description = TextSplitterFactory.get_type_description(splitter_type)
    print(f"   • {splitter_type:15} [{category:10}] - {description}")

# 测试 2: CharacterTextSplitter
print("\n\n测试 2: CharacterTextSplitter (字符分割)")
print("-" * 70)

try:
    splitter = TextSplitterFactory.get_splitter('character', chunk_size=200, chunk_overlap=20)
    from langchain_core.documents import Document
    docs = [Document(page_content=test_text, metadata={"source": "test"})]
    chunks = splitter.split_documents(docs)
    
    print(f"✅ 创建成功: {type(splitter).__name__}")
    print(f"✅ 分割结果: {len(chunks)} 个 chunks")
    print(f"\n📝 示例 chunk 1:")
    print(f"   内容: {chunks[0].page_content[:100]}...")
    print(f"   长度: {len(chunks[0].page_content)} 字符")
    print(f"   元数据: {chunks[0].metadata}")
except Exception as e:
    print(f"❌ 失败: {e}")

# 测试 3: RecursiveCharacterTextSplitter (推荐)
print("\n\n测试 3: RecursiveCharacterTextSplitter (递归字符分割 - 推荐)")
print("-" * 70)

try:
    splitter = TextSplitterFactory.get_splitter('recursive', chunk_size=200, chunk_overlap=20)
    docs = [Document(page_content=test_text, metadata={"source": "test"})]
    chunks = splitter.split_documents(docs)
    
    print(f"✅ 创建成功: {type(splitter).__name__}")
    print(f"✅ 分割结果: {len(chunks)} 个 chunks")
    print(f"\n📝 示例 chunk 1:")
    print(f"   内容: {chunks[0].page_content[:100]}...")
    print(f"   长度: {len(chunks[0].page_content)} 字符")
    print(f"   元数据: {chunks[0].metadata}")
except Exception as e:
    print(f"❌ 失败: {e}")

# 测试 4: MarkdownTextSplitter
print("\n\n测试 4: MarkdownTextSplitter (Markdown 结构分割)")
print("-" * 70)

try:
    splitter = TextSplitterFactory.get_splitter('markdown', chunk_size=300, chunk_overlap=20)
    docs = [Document(page_content=test_text, metadata={"source": "test"})]
    chunks = splitter.split_documents(docs)
    
    print(f"✅ 创建成功: {type(splitter).__name__}")
    print(f"✅ 分割结果: {len(chunks)} 个 chunks")
    print(f"\n📝 示例 chunk 1:")
    print(f"   内容: {chunks[0].page_content[:100]}...")
    print(f"   长度: {len(chunks[0].page_content)} 字符")
except Exception as e:
    print(f"❌ 失败: {e}")

# 测试 5: TokenTextSplitter
print("\n\n测试 5: TokenTextSplitter (Token 分割)")
print("-" * 70)

try:
    splitter = TextSplitterFactory.get_splitter('token', chunk_size=100, chunk_overlap=10)
    docs = [Document(page_content=test_text, metadata={"source": "test"})]
    chunks = splitter.split_documents(docs)
    
    print(f"✅ 创建成功: {type(splitter).__name__}")
    print(f"✅ 分割结果: {len(chunks)} 个 chunks")
    print(f"\n📝 示例 chunk 1:")
    print(f"   内容: {chunks[0].page_content[:100]}...")
    print(f"   长度: {len(chunks[0].page_content)} 字符")
except Exception as e:
    print(f"❌ 失败: {e}")

# 测试 6: 不支持的类型
print("\n\n测试 6: 不支持的类型")
print("-" * 70)

try:
    invalid_splitter = TextSplitterFactory.get_splitter('invalid_type')
    print("❌ 应该抛出异常但没有")
except ValueError as e:
    print(f"✅ 正确抛出 ValueError: {str(e)[:80]}...")
except Exception as e:
    print(f"❌ 抛出异常类型错误: {type(e).__name__}")

# 测试 7: 环境变量配置
print("\n\n测试 7: 从环境变量读取配置")
print("-" * 70)

os.environ['TEXT_SPLITTER_TYPE'] = 'recursive'
try:
    splitter = TextSplitterFactory.get_splitter(chunk_size=200, chunk_overlap=20)
    print(f"✅ 从环境变量读取 TEXT_SPLITTER_TYPE=recursive")
    print(f"✅ 创建成功: {type(splitter).__name__}")
except Exception as e:
    print(f"❌ 失败: {e}")

# 测试 8: 对比不同分割器的效果
print("\n\n测试 8: 不同分割器效果对比")
print("-" * 70)

print(f"\n📊 测试文档总长度: {len(test_text)} 字符")
print(f"📊 配置: chunk_size=200, chunk_overlap=20\n")

splitter_types_to_test = ['character', 'recursive', 'markdown', 'token']
results = []

for stype in splitter_types_to_test:
    try:
        splitter = TextSplitterFactory.get_splitter(stype, chunk_size=200, chunk_overlap=20)
        docs = [Document(page_content=test_text, metadata={"source": "test"})]
        chunks = splitter.split_documents(docs)
        
        avg_length = sum(len(c.page_content) for c in chunks) / len(chunks)
        results.append({
            'type': stype,
            'chunks': len(chunks),
            'avg_length': avg_length,
            'min_length': min(len(c.page_content) for c in chunks),
            'max_length': max(len(c.page_content) for c in chunks),
        })
    except Exception as e:
        results.append({
            'type': stype,
            'error': str(e)[:50]
        })

# 打印对比表格
print(f"{'分割器类型':<15} | {'Chunk数':<8} | {'平均长度':<8} | {'最小':<6} | {'最大':<6}")
print("-" * 70)
for r in results:
    if 'error' in r:
        print(f"{r['type']:<15} | 错误: {r['error']}")
    else:
        print(f"{r['type']:<15} | {r['chunks']:<8} | {r['avg_length']:<8.0f} | {r['min_length']:<6} | {r['max_length']:<6}")

print("\n" + "=" * 70)
print("✅ 所有测试完成！")
print("=" * 70)
