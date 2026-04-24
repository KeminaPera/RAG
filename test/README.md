# 测试文件说明

## 📁 测试文件清单

本目录包含所有 RAG 系统的测试脚本和分析文档。

### 🔧 功能测试

| 文件名 | 用途 | 说明 |
|--------|------|------|
| `test_embedding.py` | 测试 Embedding 模块 | 验证 BGE 向量化功能 |
| `test_bge_fallback.py` | 测试 BGE Fallback | 验证 Hash 降级机制 |
| `test_local_bge.py` | 测试本地 BGE 模型 | 验证模型加载 |
| `test_reranker.py` | 测试 Reranker 模块 | 验证重排序功能 |
| `test_reranker_factory.py` | 测试 Reranker 工厂 | 验证工厂模式和类型切换 ⭐ |
| `test_memory.py` | 测试记忆系统 | 验证三层记忆功能 |
| `test_rag_flow.py` | 测试完整 RAG 流程 | 端到端测试 ⭐ |
| `test_import.py` | 测试模块导入 | 验证依赖完整性 |

### 🔍 数据检查

| 文件名 | 用途 | 说明 |
|--------|------|------|
| `inspect_data.py` | 查看已存储数据 | 检查 ChromaDB、Entity Memory、Long-Term Memory ⭐ |

### 📊 综合分析

| 文件名 | 用途 | 说明 |
|--------|------|------|
| `test_all_optimizations.py` | 综合优化测试 | 验证 ChromaDB、LRU 缓存、配置外置等 ⭐ |
| `DATA_STORAGE_ANALYSIS.md` | 数据存储分析 | 详细的数据存储机制文档 📖 |

---

## 🚀 快速使用

### 1. 查看已存储的数据
```bash
.venv\Scripts\python.exe test\inspect_data.py
```

**输出示例**:
```
📊 RAG 系统数据存储检查

1️⃣ ChromaDB 向量数据库
✅ Collection 数量: 1
📁 Collection: documents
   📄 文档总数: 3
   
2️⃣ Entity Memory 实体记忆
✅ 实体数量: 0

3️⃣ Long-Term Memory 长期记忆
✅ 记忆条目数: 21
```

---

### 2. 测试 Reranker 工厂模式
```bash
.venv\Scripts\python.exe test\test_reranker_factory.py
```

**测试内容**:
- ✅ BGE Cross-Encoder 创建
- ✅ NoOp Reranker 创建
- ✅ 单例模式验证
- ✅ 错误处理

---

### 3. 测试完整 RAG 流程
```bash
.venv\Scripts\python.exe test\test_rag_flow.py
```

**测试内容**:
- ✅ ChromaDB 检索
- ✅ Reranker 重排序
- ✅ LLM 回答生成

---

### 4. 综合优化验证
```bash
.venv\Scripts\python.exe test\test_all_optimizations.py
```

**测试内容**:
- ✅ 配置外置
- ✅ 安全配置
- ✅ ChromaDB 集成
- ✅ 查询功能
- ✅ LRU 缓存

---

## 📖 数据存储说明

详细的数据存储机制分析请查看：
👉 [DATA_STORAGE_ANALYSIS.md](./DATA_STORAGE_ANALYSIS.md)

### 核心存储位置

```
data/
├── chromadb/                    # 向量数据库
│   ├── chroma.sqlite3           # 元数据
│   └── {uuid}/                  # HNSW 索引
│       └── *.bin
│
└── memory/
    ├── entity_memory.db          # 实体记忆 (SQLite)
    └── long_term/
        └── memory.json           # 长期记忆 (JSON)
```

### 数据导入流程

```
上传 PDF/DOCX
    ↓
解析 + 分割 (500字符/chunk)
    ↓
BGE 向量化 (512维)
    ↓
保存到 ChromaDB
    ↓
✅ 自动持久化到 data/chromadb/
```

---

## 🧹 清理测试数据

如需清空所有测试数据：

```python
import shutil
import os

# 删除所有数据
if os.path.exists("data"):
    shutil.rmtree("data")

# 重建目录
os.makedirs("data/chromadb", exist_ok=True)
os.makedirs("data/memory/long_term", exist_ok=True)

print("✅ 数据已清空")
```

---

## ⚠️ 注意事项

1. **旧版遗留文件**: `data/chromadb/documents.json` 已不再使用，可安全删除
2. **ChromaDB Telemetry**: 启动时可能出现 telemetry 警告，不影响功能
3. **数据备份**: 重要数据请定期备份 `data/` 目录

---

## 📝 添加新测试

创建新测试文件时，请遵循以下规范：

1. **命名**: `test_{module_name}.py`
2. **工作目录**: 文件开头设置 `os.chdir(r"e:\idea_workspace\RAG")`
3. **输出**: 使用清晰的 emoji 和分隔线
4. **异常处理**: 所有测试都要有 try-except

**模板**:
```python
"""
测试 {模块名称}
"""
import os
os.chdir(r"e:\idea_workspace\RAG")

print("=" * 60)
print("测试 {模块名称}")
print("=" * 60)

try:
    # 测试代码
    print("✅ 测试通过")
except Exception as e:
    print(f"❌ 测试失败: {e}")
```

---

## 🎯 推荐测试顺序

对于新功能验证，建议按以下顺序运行测试：

1. `test_import.py` - 确认依赖正常
2. `test_embedding.py` - 验证向量化
3. `test_reranker_factory.py` - 验证重排序
4. `test_memory.py` - 验证记忆系统
5. `test_rag_flow.py` - 端到端测试
6. `test_all_optimizations.py` - 综合验证
7. `inspect_data.py` - 检查数据存储

---

**最后更新**: 2026-04-23
