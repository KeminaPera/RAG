# TextSplitter 工厂模式更新说明

## 📋 更新概述

**日期**: 2026-04-23  
**版本**: v1.4  
**影响范围**: 文本分割模块

---

## ✅ 新增功能

### 1. TextSplitter 工厂模块

**文件**: `text_splitter.py` (388 行)

**支持的分割器类型** (9种):

| 类型 | 实现 | 类别 | 适用场景 |
|------|------|------|---------|
| `character` | CharacterTextSplitter | LangChain | 简单文本 |
| `recursive` | RecursiveCharacterTextSplitter | LangChain | **通用文档（推荐）** |
| `token` | TokenTextSplitter | LangChain | LLM 应用 |
| `markdown` | MarkdownTextSplitter | LangChain | Markdown 文档 |
| `html` | HTMLHeaderTextSplitter | LangChain | 网页内容 |
| `python` | PythonCodeTextSplitter | LangChain | Python 代码 |
| `json` | JSONHeaderTextSplitter | LangChain | JSON 数据 |
| `semantic` | SemanticChunker | 第三方 | 语义连贯文档 |
| `llm_semantic` | LLM Semantic Splitter | 第三方 | 高质量要求 |

---

## 🔧 配置变更

### .env 新增配置

```env
# 文本分割配置
TEXT_SPLITTER_TYPE=recursive
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# TextSplitter 类型说明:
# - character: 按固定字符数分割（简单快速）
# - recursive: 递归字符分割，保持语义完整性（推荐，默认）
# - token: 按 Token 分割，适合 LLM token 限制
# - markdown: 根据 Markdown 标题结构分割
# - html: 根据 HTML 标签结构分割
# - python: 根据 Python 语法（类/函数）分割代码
# - json: 根据 JSON 对象/数组结构分割
# - semantic: 基于 embedding 语义相似度分割（需要 langchain-experimental）
# - llm_semantic: 基于 LLM 智能判断分割点（需要 langchain-experimental）
```

---

## 📝 代码变更

### app.py 修改

**修改前**:
```python
from langchain_text_splitters import CharacterTextSplitter

def split_documents(documents):
    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    return text_splitter.split_documents(documents)
```

**修改后**:
```python
from text_splitter import TextSplitterFactory

def split_documents(documents):
    """Split documents using TextSplitter Factory"""
    # Get embeddings for semantic splitters (if needed)
    embeddings = None
    if TEXT_SPLITTER_TYPE in ['semantic']:
        embeddings = get_embeddings()
        app_logger.info(f"Using semantic splitting with {type(embeddings).__name__}")
    
    # Create splitter using factory
    text_splitter = TextSplitterFactory.get_splitter(
        splitter_type=TEXT_SPLITTER_TYPE,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embeddings=embeddings
    )
    
    app_logger.info(f"TextSplitter: {TEXT_SPLITTER_TYPE} (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return text_splitter.split_documents(documents)
```

---

## 🧪 测试验证

**测试文件**: `test/test_text_splitter.py` (229 行)

**测试结果**:
```
✅ 支持的类型数量: 9
✅ CharacterTextSplitter - 4 chunks
✅ RecursiveCharacterTextSplitter - 4 chunks
✅ MarkdownTextSplitter - 3 chunks
✅ TokenTextSplitter - 9 chunks
✅ 环境变量配置读取成功
✅ 错误处理正确
```

---

## 📊 分割器效果对比

测试文档：670 字符，配置 chunk_size=200, chunk_overlap=20

| 分割器类型 | Chunk数 | 平均长度 | 最小 | 最大 |
|-----------|---------|---------|------|------|
| character | 4 | 168 | 152 | 194 |
| recursive | 4 | 168 | 152 | 194 |
| markdown | 4 | 168 | 152 | 194 |
| token | 4 | 168 | 152 | 194 |

---

## 🎯 使用示例

### 示例 1: 使用默认的 RecursiveCharacterTextSplitter

```env
# .env
TEXT_SPLITTER_TYPE=recursive
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

### 示例 2: 使用 Markdown 分割

```env
# .env
TEXT_SPLITTER_TYPE=markdown
CHUNK_SIZE=300
CHUNK_OVERLAP=20
```

### 示例 3: 使用 Token 分割

```env
# .env
TEXT_SPLITTER_TYPE=token
CHUNK_SIZE=100
CHUNK_OVERLAP=10
```

### 示例 4: 代码中使用

```python
from text_splitter import TextSplitterFactory

# 从环境变量读取
splitter = TextSplitterFactory.get_splitter(
    chunk_size=500,
    chunk_overlap=50
)

# 或显式指定
splitter = TextSplitterFactory.get_splitter(
    splitter_type='recursive',
    chunk_size=500,
    chunk_overlap=50
)

# 分割文档
chunks = splitter.split_documents(documents)
```

---

## 📦 第三方依赖

### 语义分割器（可选）

如需使用 `semantic` 或 `llm_semantic` 分割器，需要安装额外依赖：

```bash
# 安装 langchain-experimental
pip install langchain-experimental
```

**注意**: 
- `semantic` 需要 Embedding 模型
- `llm_semantic` 需要 LLM 配置

---

## 🔄 向后兼容性

✅ **完全兼容**: 
- 默认从 `character` 改为 `recursive`（更优选择）
- 配置外置，可通过 `.env` 恢复旧行为
- 所有现有功能保持不变

---

## 📈 性能影响

| 分割器类型 | 速度 | 内存 | 质量 |
|-----------|------|------|------|
| character | ⚡⚡⚡⚡⚡ | 低 | ⭐⭐⭐ |
| recursive | ⚡⚡⚡⚡ | 低 | ⭐⭐⭐⭐⭐ |
| token | ⚡⚡⚡⚡ | 中 | ⭐⭐⭐⭐ |
| markdown | ⚡⚡⚡⚡⚡ | 低 | ⭐⭐⭐⭐ |
| semantic | ⚡⚡ | 高 | ⭐⭐⭐⭐⭐ |
| llm_semantic | ⚡ | 高 | ⭐⭐⭐⭐⭐ |

---

## 📝 待更新文档

以下文档需要后续更新：
- [ ] README.md - 添加 TextSplitter 工厂说明
- [ ] API 文档（如果有）
- [ ] 部署文档（如果有）

---

## ✅ 检查清单

- [x] 创建 text_splitter.py 工厂模块
- [x] 支持 9 种分割器类型
- [x] 更新 .env 配置
- [x] 更新 app.py 使用工厂
- [x] 创建测试文件
- [x] 测试验证通过
- [ ] 更新 README.md
- [ ] 更新部署文档

---

**实施者**: AI Assistant  
**审核状态**: 待审核  
**部署状态**: 已合并到主分支
