# 数据存储分析报告

## 📊 数据导入与保存完整流程

### 1️⃣ 文档上传流程

```
用户上传文件 (PDF/DOCX)
    ↓
保存到临时目录: ./uploads/{filename}
    ↓
解析文档内容
    ↓
文本分割 (chunk_size=500, overlap=50)
    ↓
添加元数据 (source_file, chunk_id)
    ↓
BGE 模型向量化
    ↓
保存到 ChromaDB
    ↓
删除临时文件
```

---

## 📁 数据存储位置详解

### 根目录: `e:\idea_workspace\RAG\data\`

```
data/
├── chromadb/                    # 向量数据库存储
│   │
│   ├── chroma.sqlite3           # ChromaDB 元数据 (SQLite)
│   │   └── 存储: collection信息、配置等
│   │
│   ├── documents.json           # ⚠️ 旧版遗留文件 (已不再使用)
│   │
│   └── 6caed4c5-8d55-4824-8b9b-cc75b73e0a8a/  # HNSW索引数据
│       ├── data_level0.bin      # 向量数据 (层级0)
│       ├── header.bin           # 索引头信息
│       ├── length.bin           # 向量长度信息
│       └── link_lists.bin       # HNSW链接列表
│
└── memory/                      # 记忆系统存储
    │
    ├── entity_memory.db          # 实体记忆 (SQLite)
    │   └── 表: entities
    │       - id, name, attributes, created_at, updated_at
    │
    └── long_term/
        └── memory.json           # 长期记忆 (JSON)
            └── 存储: QA对、知识点等
```

---

## 🔍 各存储系统详解

### 1. ChromaDB 向量数据库

**位置**: `./data/chromadb/`

**存储内容**:
- **文档内容**: 分割后的文本块 (page_content)
- **向量嵌入**: 512维 BGE 向量 (embedding)
- **元数据**: 
  ```json
  {
    "source_file": "【NLP算法实习生_北京】王启飞 24年应届生.pdf",
    "chunk_id": 0,
    "page": 1  // 如果有
  }
  ```
- **唯一ID**: `doc_{uuid4}` (例如: `doc_a1b2c3d4-e5f6-7890-abcd-ef1234567890`)

**实现代码** (`app.py` 第 181-219 行):
```python
def save_to_chroma(chunks, collection_name="documents"):
    """保存文档到 ChromaDB"""
    client = get_chroma_client()
    
    # 准备数据
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata or {} for chunk in chunks]
    ids = [f"doc_{uuid.uuid4()}" for _ in chunks]
    
    # 批量插入
    collection.add(
        documents=texts,      # 文本内容
        metadatas=metadatas,  # 元数据
        ids=ids              # 唯一ID
    )
```

**持久化机制**:
- ✅ **自动持久化**: `chromadb.PersistentClient` 自动保存
- ✅ **无需手动调用**: 每次 `add()` 后立即写入磁盘
- ✅ **重启可用**: 应用重启后数据自动加载

---

### 2. Entity Memory 实体记忆

**位置**: `./data/memory/entity_memory.db`

**数据库类型**: SQLite

**表结构**:
```sql
CREATE TABLE entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,        -- 实体名称 (如: user_id)
    attributes TEXT NOT NULL,          -- JSON 格式的属性
    created_at TEXT NOT NULL,          -- 创建时间
    updated_at TEXT NOT NULL           -- 更新时间
)
```

**示例数据**:
```json
{
    "name": "default_user",
    "attributes": {
        "name": "张三",
        "preference": "AI技术",
        "tool": "Python",
        "interest": "机器学习"
    },
    "created_at": "2026-04-23T16:00:00",
    "updated_at": "2026-04-23T16:30:00"
}
```

**使用场景**:
- 存储用户属性信息
- 跨会话持久化
- 支持向量检索相似实体

---

### 3. Long-Term Memory 长期记忆

**位置**: `./data/memory/long_term/memory.json`

**存储格式**: JSON 数组

**数据结构**:
```json
[
    {
        "content": "Q: NLP算法实习生都有谁?\nA: 根据文档，NLP算法实习生包括...",
        "embedding": [0.123, -0.456, 0.789, ...],  // 512维向量
        "metadata": {
            "type": "qa_pair",
            "session_id": "abc-123-def"
        },
        "created_at": "2026-04-23T16:30:00"
    }
]
```

**特点**:
- 存储历史 QA 对
- 包含向量嵌入支持语义检索
- 增量更新，越用越丰富

---

## 📝 数据导入完整代码流程

### Upload 路由 (`app.py` 第 273-311 行)

```python
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 1. 接收文件
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # 2. 解析文档
        documents = load_document(file_path)  # PDF/DOCX → Documents
        
        # 3. 文本分割
        chunks = split_documents(documents)   # 大文档 → 小块 (500字符)
        
        # 4. 添加元数据
        for i, chunk in enumerate(chunks):
            chunk.metadata["source_file"] = filename
            chunk.metadata["chunk_id"] = i
        
        # 5. 向量化并保存到 ChromaDB
        save_to_chroma(chunks)  # ← 这里触发 BGE 向量化 + 保存
        
        # 6. 清理临时文件
        os.remove(file_path)
```

### save_to_chroma 函数详解

```python
def save_to_chroma(chunks, collection_name="documents"):
    # 1. 获取 ChromaDB 客户端 (单例)
    client = get_chroma_client()
    
    # 2. 获取或创建 collection
    collection = client.get_collection(collection_name)
    
    # 3. 准备数据
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata or {} for chunk in chunks]
    ids = [f"doc_{uuid.uuid4()}" for _ in chunks]
    
    # 4. 批量添加到 ChromaDB
    collection.add(
        documents=texts,      # 文本内容列表
        metadatas=metadatas,  # 元数据列表
        ids=ids              # 唯一ID列表
    )
    # ← 此时 ChromaDB 内部:
    #   a. 调用 embedding_function 将 texts 转为向量
    #   b. 构建 HNSW 索引
    #   c. 持久化到磁盘 (data/chromadb/)
```

---

## 🔎 如何查看已存储的数据

### 1. 查看 ChromaDB 数据

```python
import chromadb

# 连接数据库
client = chromadb.PersistentClient(path="./data/chromadb")

# 获取 collection
collection = client.get_collection("documents")

# 查询所有数据
all_data = collection.get()
print(f"总文档数: {len(all_data['ids'])}")
print(f"示例元数据: {all_data['metadatas'][0]}")

# 相似度查询
results = collection.query(
    query_texts=["NLP算法"],
    n_results=5
)
for i, doc in enumerate(results['documents'][0]):
    print(f"\n文档 {i+1}:")
    print(f"内容: {doc[:100]}...")
    print(f"元数据: {results['metadatas'][0][i]}")
```

### 2. 查看 Entity Memory

```python
import sqlite3

conn = sqlite3.connect("./data/memory/entity_memory.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM entities")
rows = cursor.fetchall()

for row in rows:
    print(f"实体: {row[1]}")
    print(f"属性: {row[2]}")
    print(f"创建时间: {row[3]}")
    print("---")

conn.close()
```

### 3. 查看 Long-Term Memory

```python
import json

with open("./data/memory/long_term/memory.json", "r", encoding="utf-8") as f:
    memories = json.load(f)

print(f"长期记忆条目数: {len(memories)}")

for mem in memories:
    print(f"\n内容: {mem['content'][:100]}...")
    print(f"元数据: {mem['metadata']}")
    print(f"创建时间: {mem['created_at']}")
```

---

## 📊 数据量估算

### 单个文档示例

**假设**: 上传一个 10 页 PDF

| 阶段 | 数据量 | 说明 |
|------|--------|------|
| 原始文档 | ~50KB | PDF 文件大小 |
| 解析后文本 | ~30KB | 纯文本内容 |
| 分割后 chunks | ~60 个 | 每块 500 字符 |
| 向量数据 | ~60 × 512 × 4B = ~120KB | float32 向量 |
| ChromaDB 存储 | ~200-300KB | 包含索引结构 |

### 当前存储状态

运行以下命令查看实际占用：
```bash
# Windows PowerShell
Get-ChildItem -Path data -Recurse | 
    Select-Object FullName, @{Name="Size(KB)";Expression={[math]::Round($_.Length/1KB, 2)}} | 
    Format-Table -AutoSize
```

---

## ⚠️ 注意事项

### 1. 旧版遗留文件
- `data/chromadb/documents.json` 是旧版 `SimpleVectorDB` 的遗留文件
- ✅ 现已不再使用，可以安全删除

### 2. 数据备份
建议定期备份以下文件/目录：
```
data/chromadb/          # 向量数据库
data/memory/            # 记忆系统
```

### 3. 清理数据
如需清空所有数据：
```python
import shutil
import os

# 删除 ChromaDB
if os.path.exists("data/chromadb"):
    shutil.rmtree("data/chromadb")

# 删除记忆系统
if os.path.exists("data/memory"):
    shutil.rmtree("data/memory")

# 重建目录
os.makedirs("data/chromadb", exist_ok=True)
os.makedirs("data/memory/long_term", exist_ok=True)
```

---

## 🎯 总结

### 数据流向一图看懂

```
用户上传 PDF/DOCX
    ↓
临时保存: ./uploads/文件名.pdf
    ↓
解析 + 分割 → N 个 chunks
    ↓
每个 chunk:
    - content: 文本内容 (500字符)
    - metadata: {source_file, chunk_id}
    ↓
BGE 模型向量化 → 512维向量
    ↓
保存到 ChromaDB:
    - data/chromadb/chroma.sqlite3 (元数据)
    - data/chromadb/{uuid}/*.bin (向量+索引)
    ↓
删除临时文件 ./uploads/文件名.pdf
    ↓
✅ 数据持久化完成
```

### 关键特性
- ✅ **自动持久化**: 无需手动保存
- ✅ **重启可用**: 数据不会丢失
- ✅ **高效检索**: HNSW 索引支持快速相似度搜索
- ✅ **增量更新**: 支持多次上传，数据累加
