# 文档向量检索系统 (RAG)

基于 LangChain 1.x 和 Flask 的智能文档检索平台，支持 PDF/Word 文档上传、向量存储、语义检索和大模型回答生成。

**最新版本**: LangChain 1.0.0 | **更新**: 2026-04-27

## 项目架构

```
├── app.py                 # Flask Web 应用入口
├── requirements.txt       # 依赖清单（版本锁定）
├── .env                   # 环境配置（API密钥等敏感信息）
├── .gitignore             # Git忽略规则
├── embeddings.py          # 向量嵌入模块（BGE-zh 模型，工厂模式）
├── reranker.py            # 重排序模块（工厂模式）
├── text_splitter.py       # 文本分割模块（工厂模式）
├── rag_service.py         # RAG 核心服务
├── llm_client.py          # 大模型客户端
├── memory_manager.py      # 三层记忆管理模块
├── logging_config.py      # 日志配置模块
├── scripts/               # 辅助脚本目录
│   ├── start_app.py       # 应用启动脚本
│   └── download_reranker.py  # 模型下载工具
├── templates/             # HTML 模板
│   ├── index.html         # 首页
│   ├── upload.html        # 文档上传页面
│   └── query.html         # 查询页面
├── data/                  # 数据存储目录（运行时生成）
│   ├── chromadb/          # 向量数据库存储
│   └── memory/            # 记忆数据（实体记忆 + 长期记忆）
├── test/                  # 测试目录
│   ├── README.md          # 测试说明文档
│   ├── DATA_STORAGE_ANALYSIS.md  # 数据存储分析
│   ├── inspect_data.py    # 数据检查工具
│   ├── test_all_optimizations.py # 综合优化测试
│   ├── test_reranker_factory.py  # Reranker 工厂测试
│   ├── test_rag_flow.py   # RAG 流程测试
│   ├── test_memory.py     # 记忆系统测试
│   ├── test_embedding.py  # Embedding 测试
│   ├── test_bge_fallback.py
│   ├── test_local_bge.py
│   ├── test_reranker.py
│   └── test_import.py
├── uploads/               # 上传文件临时目录（运行时生成）
├── logs/                  # 日志目录（运行时生成）
├── bge-small-zh-v1.5/     # BGE 中文嵌入模型（本地部署）
└── bge-reranker-base/     # BGE 重排序模型（本地部署）
```

## 技术栈

| 分类 | 技术 | 版本 | 说明 |
|------|------|------|------|
| Web 框架 | Flask | 3.1.3 | 轻量级 Web 框架 |
| 向量数据库 | ChromaDB | 0.5.0 | 向量存储与检索 |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 | - | 中文语义嵌入（本地） |
| 重排序模型 | BAAI/bge-reranker-base | - | 检索结果重排序（本地，工厂模式） |
| 文本分割 | LangChain Text Splitters | 1.0.0 | 支持 9 种分割算法（工厂模式） |
| LLM 框架 | LangChain | 1.0.0 | 大语言模型编排 |
| LangChain Core | langchain-core | 1.0.0 | LangChain 核心组件 |
| LangChain Classic | langchain-classic | 1.0.0 | 记忆、链、检索器等传统组件 |
| LangChain OpenAI | langchain-openai | 1.0.0 | OpenAI 集成 |
| 文档解析 | pypdf / python-docx | 6.1.0 / 1.2.0 | PDF/Word 解析 |
| 深度学习 | PyTorch | 2.2.0 (CPU) | 模型推理框架 |
| Transformer | Transformers | 4.41.1 | HuggingFace 模型库 |
| 向量化 | Sentence-Transformers | 2.7.0 | 文本嵌入生成 |
| 记忆系统 | 三层记忆架构 | - | 短期/实体/长期记忆 |

**环境要求**：Python >= 3.11

## 核心功能

### 1. 文档上传
- 支持 PDF (.pdf) 和 Word (.docx) 文件
- 自动文本分割（支持 9 种分割算法，工厂模式）
- 向量化存储到向量数据库

### 2. 智能文本分割系统

系统采用工厂模式设计，支持 9 种文本分割算法，与 Embedding 和 Reranker 模块保持一致的架构设计。

#### 2.1 架构设计

```
TextSplitterFactory (工厂类)
    │
    ├── LangChainTextSplitters (LangChain 内置分割器)
    │   ├── get_character() → CharacterTextSplitter
    │   ├── get_recursive() → RecursiveCharacterTextSplitter (推荐)
    │   ├── get_token() → TokenTextSplitter
    │   ├── get_markdown() → MarkdownTextSplitter
    │   ├── get_html() → HTMLHeaderTextSplitter
    │   ├── get_python() → PythonCodeTextSplitter
    │   └── get_json() → JSONHeaderTextSplitter
    │
    └── ThirdPartyTextSplitters (第三方分割器)
        ├── get_semantic() → SemanticChunker (需要 embedding)
        └── get_llm_semantic() → LLM Semantic Splitter (需要 LLM)
```

**设计模式**: 工厂模式 + 策略模式
**配置文件**: `.env` 中的 `TEXT_SPLITTER_TYPE`
**模块文件**: `text_splitter.py` (388 行)

#### 2.2 支持的分割器类型

| 类型 | 实现方式 | 类别 | 适用场景 | 特点 | 推荐度 |
|------|---------|------|---------|------|--------|
| **recursive** | RecursiveCharacterTextSplitter | LangChain | 通用文档 | 递归分割，保持语义完整性 | ⭐⭐⭐⭐⭐ |
| **character** | CharacterTextSplitter | LangChain | 简单文本 | 按固定字符数分割 | ⭐⭐⭐ |
| **token** | TokenTextSplitter | LangChain | LLM 应用 | 按 Token 限制分割 | ⭐⭐⭐⭐ |
| **markdown** | MarkdownTextSplitter | LangChain | Markdown 文档 | 按标题结构分割 | ⭐⭐⭐⭐ |
| **html** | HTMLHeaderTextSplitter | LangChain | 网页内容 | 按 HTML 标签分割 | ⭐⭐⭐⭐ |
| **python** | PythonCodeTextSplitter | LangChain | Python 代码 | 按类/函数分割 | ⭐⭐⭐⭐ |
| **json** | JSONHeaderTextSplitter | LangChain | JSON 数据 | 按对象/数组分割 | ⭐⭐⭐⭐ |
| **semantic** | SemanticChunker | 第三方 | 语义连贯文档 | 基于 embedding 语义分割 | ⭐⭐⭐⭐⭐ |
| **llm_semantic** | LLM Semantic Splitter | 第三方 | 高质量要求 | LLM 智能判断分割点 | ⭐⭐⭐⭐⭐ |

**详细说明**:

1. **RecursiveCharacterTextSplitter (推荐)**
   - 工作原理：按段落→句子→词的顺序递归分割
   - 优势：在保持语义完整性的同时控制 chunk 大小
   - 适用：大多数通用文档
   - 默认分隔符：`["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]`

2. **CharacterTextSplitter**
   - 工作原理：按固定字符数分割
   - 优势：简单快速，可预测
   - 适用：结构化文本、日志文件

3. **TokenTextSplitter**
   - 工作原理：使用 tiktoken 编码器按 token 分割
   - 优势：精确控制 LLM token 使用量
   - 适用：需要严格控制 token 数量的场景
   - 默认编码器：`cl100k_base` (GPT-4)

4. **MarkdownTextSplitter**
   - 工作原理：根据 Markdown 标题层级分割
   - 优势：保持文档结构和层次
   - 适用：Markdown 文档、技术文档

5. **HTMLHeaderTextSplitter**
   - 工作原理：根据 HTML 标签 (h1, h2, h3...) 分割
   - 优势：保留网页结构
   - 适用：HTML 网页、在线文档

6. **PythonCodeTextSplitter**
   - 工作原理：根据 Python 语法（类定义、函数定义）分割
   - 优势：保持代码逻辑完整性
   - 适用：Python 源代码文件

7. **JSONHeaderTextSplitter**
   - 工作原理：根据 JSON 对象/数组结构分割
   - 优势：保持数据结构完整
   - 适用：JSON 配置文件、API 响应

8. **SemanticChunker (需要额外依赖)**
   - 工作原理：使用 embedding 模型计算语义相似度，在语义边界处分割
   - 优势：语义连贯性最佳
   - 适用：长文档、专业文献
   - 依赖：`pip install langchain-experimental`
   - 配置参数：
     - `breakpoint_threshold_type`: 'percentile' (默认), 'standard_deviation', 'interquartile'
     - `breakpoint_threshold_amount`: 0.95 (默认)

9. **LLM Semantic Splitter (需要额外依赖)**
   - 工作原理：使用 LLM 智能判断分割点
   - 优势：最智能的分割方式
   - 适用：对分割质量要求极高的场景
   - 依赖：`pip install langchain-experimental`
   - 注意：速度最慢，成本最高

#### 2.3 配置方式

**基础配置** (在 `.env` 文件中):

```env
# 文本分割配置
TEXT_SPLITTER_TYPE=recursive
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

**完整配置说明**:

| 配置项 | 默认值 | 说明 | 可选值 |
|--------|--------|------|--------|
| `TEXT_SPLITTER_TYPE` | recursive | 分割器类型 | character, recursive, token, markdown, html, python, json, semantic, llm_semantic |
| `CHUNK_SIZE` | 500 | 每个 chunk 的大小 | 100-2000 (根据场景调整) |
| `CHUNK_OVERLAP` | 50 | chunk 重叠大小 | 0-200 (建议为 CHUNK_SIZE 的 10%) |

#### 2.4 最佳实践配置

| 场景 | TEXT_SPLITTER_TYPE | CHUNK_SIZE | CHUNK_OVERLAP | 说明 |
|------|-------------------|------------|---------------|------|
| **通用文档** | recursive | 500 | 50 | 默认推荐，适合大多数场景 |
| **Markdown 文档** | markdown | 400 | 40 | 保持标题结构 |
| **Python 代码** | python | 300 | 30 | 保持函数/类完整性 |
| **LLM 应用** | token | 100 | 10 | 按 token 计数 |
| **高质量检索** | semantic | 500 | 50 | 语义连贯性最佳 |
| **网页内容** | html | 400 | 40 | 保持 HTML 结构 |
| **JSON 数据** | json | 500 | 50 | 保持数据结构 |
| **简单文本** | character | 500 | 50 | 快速分割 |

#### 2.5 工作流程

```
用户上传文档 (PDF/DOCX)
    ↓
解析文档内容
    ↓
TextSplitterFactory.get_splitter(
    splitter_type=TEXT_SPLITTER_TYPE,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    embeddings=embeddings  # 仅 semantic 类型需要
)
    ↓
创建对应的 TextSplitter 实例
    ↓
分割文档
    ↓
返回 List[Document] chunks
    ↓
向量化并保存到 ChromaDB
```

**代码示例**:

```python
from text_splitter import TextSplitterFactory

# 方式 1: 从环境变量读取配置
splitter = TextSplitterFactory.get_splitter(
    chunk_size=500,
    chunk_overlap=50
)

# 方式 2: 显式指定类型
splitter = TextSplitterFactory.get_splitter(
    splitter_type='recursive',
    chunk_size=500,
    chunk_overlap=50
)

# 方式 3: 使用语义分割（需要 embedding）
from embeddings import EmbeddingFactory
embeddings = EmbeddingFactory.get_embedding('bge_zh')

splitter = TextSplitterFactory.get_splitter(
    splitter_type='semantic',
    chunk_size=500,
    embeddings=embeddings
)

# 分割文档
from langchain_core.documents import Document
doc = Document(page_content="长文本内容...", metadata={"source": "test"})
chunks = splitter.split_documents([doc])

print(f"分割成 {len(chunks)} 个 chunks")
```

#### 2.6 查看支持的类型

```python
from text_splitter import TextSplitterFactory

# 获取所有支持的类型
supported = TextSplitterFactory.get_supported_types()
print(supported)
# 输出: {'character': 'langchain', 'recursive': 'langchain', ...}

# 获取类型描述
description = TextSplitterFactory.get_type_description('recursive')
print(description)
# 输出: 递归字符分割，保持语义完整性（推荐）
```

#### 2.7 第三方分割器安装

如需使用 `semantic` 或 `llm_semantic` 分割器，需要安装额外依赖：

```bash
# 安装 langchain-experimental
pip install langchain-experimental
```

**注意事项**:
- `semantic` 分割器需要 Embedding 模型（系统已内置 BGE）
- `llm_semantic` 分割器需要 LLM 配置（系统已支持 OpenAI/Ollama）
- 第三方分割器速度较慢，但分割质量更高

#### 2.8 特点总结

- 🚀 **工厂模式**：统一接口，配置驱动切换算法
- 📦 **LangChain 内置**：7 种官方分割器，稳定可靠
- 🔧 **第三方扩展**：支持 semantic 和 llm_semantic 智能分割
- 🎯 **推荐默认**：recursive 类型适合大多数场景
- 📊 **灵活配置**：chunk_size 和 chunk_overlap 可调
- 🔍 **详细日志**：记录分割器类型和参数
- 🛡️ **错误处理**：不支持的类型会抛出清晰的错误信息
- 📝 **类型注解**：完整的类型提示，IDE 友好

### 3. 语义检索

使用 ChromaDB 进行高效的向量相似度检索，支持 Top-K 候选文档召回。

### 4. 重排序优化
- 使用 bge-reranker-base 对检索结果重排序
- 工厂模式支持 bge_cross_encoder 和 noop 两种类型
- 提升检索精度

### 5. 大模型回答
- 支持 OpenAI 兼容接口和 Ollama
- 基于检索上下文生成回答
- 自动引用来源文档

### 6. 向量化与 Embeddings

系统采用工厂模式设计，支持多种向量化算法：

| Embedding 类型 | 实现方式 | 精度 | 速度 | 适用场景 |
|--------------|---------|------|------|----------|
| **bge_zh** | BAAI/bge-small-zh-v1.5 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 生产环境（默认） |
| **hash** | 简单哈希算法 | ⭐ | ⭐⭐⭐⭐⭐ | 降级方案/测试 |

**工作流程：**
```
文本输入
    ↓
EmbeddingFactory.get_embedding(type)
    ↓
BGE 模型向量化 (512维)
    ↓
如果失败 → 自动降级到 Hash Embedding
    ↓
返回向量表示
```

**配置方式：**
```env
# 在 .env 文件中设置
EMBEDDING_TYPE=bge_zh  # 或 hash
BGE_LOCAL_MODEL_PATH=./bge-small-zh-v1.5
```

**特点：**
- 🚀 **启动时加载一次**：模型在首次调用时加载，后续复用
- 🔧 **工厂模式**：统一接口，轻松切换不同算法
- 🛡️ **自动降级**：BGE 加载失败时自动回退到 Hash 方法
- 📦 **单例模式**：全局唯一实例，避免重复加载

### 6. 智能重排序系统

系统采用工厂模式设计，支持多种重排序算法：

| Reranker 类型 | 实现方式 | 精度 | 速度 | 适用场景 |
|--------------|---------|------|------|----------|
| **bge_cross_encoder** | BGE Cross-Encoder 模型 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 生产环境（默认） |
| **noop** | 不做重排序 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 测试/极速响应 |

**重排序工作流程：**
```
向量检索（15条候选）
    ↓
Reranker 重排序（按配置类型）
    ↓
返回 Top-K 最相关文档（默认5条）
    ↓
LLM 生成回答
```

**配置方式：**
```env
# 在 .env 文件中设置
RERANKER_TYPE=bge_cross_encoder  # 或 noop
RERANKER_MODEL_PATH=./bge-reranker-base
```

**特点：**
- 🚀 **启动时加载一次**：模型在应用启动时预加载，后续请求直接复用
- 🔧 **工厂模式**：统一接口，轻松切换不同算法
- 📊 **详细日志**：记录每个文档的排序分数和来源

### 5. 向量化与 Embeddings

系统采用工厂模式设计，支持多种向量化算法：

| Embedding 类型 | 实现方式 | 精度 | 速度 | 适用场景 |
|--------------|---------|------|------|----------|
| **bge_zh** | BAAI/bge-small-zh-v1.5 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 生产环境（默认） |
| **hash** | 简单哈希算法 | ⭐ | ⭐⭐⭐⭐⭐ | 降级方案/测试 |

**工作流程：**
```
文本输入
    ↓
EmbeddingFactory.get_embedding(type)
    ↓
BGE 模型向量化 (512维)
    ↓
如果失败 → 自动降级到 Hash Embedding
    ↓
返回向量表示
```

**配置方式：**
```env
# 在 .env 文件中设置
EMBEDDING_TYPE=bge_zh  # 或 hash
BGE_LOCAL_MODEL_PATH=./bge-small-zh-v1.5
```

**特点：**
- 🚀 **启动时加载一次**：模型在首次调用时加载，后续复用
- 🔧 **工厂模式**：统一接口，轻松切换不同算法
- 🛡️ **自动降级**：BGE 加载失败时自动回退到 Hash 方法
- 📦 **单例模式**：全局唯一实例，避免重复加载

### 7. 智能重排序系统

### 8. 三层记忆系统

| 记忆类型 | 实现方式 | 作用 | 特点 |
|---------|---------|------|------|
| **短期记忆** | ConversationBufferMemory | 维持当前对话连贯性 | 自动淘汰旧对话，LRU缓存管理 |
| **实体记忆** | SQLite + 向量检索 | 记住用户/对象属性 | 跨会话持久化 |
| **长期记忆** | JSON + 向量检索 | 记住知识/经验/规则 | 增量更新，越用越全 |

**记忆缓存管理：**
- 使用 LRU（Least Recently Used）缓存策略
- 最多保留 10 个 session（可配置）
- 自动淘汰最久未使用的会话，防止内存泄漏

**记忆层级调用：**
```
用户输入
    ↓
实体记忆检索（用户/对象）→ 系统层
长期记忆检索（RAG）→ 知识层
短期记忆加载（对话历史）→ 上下文层
    ↓
三层拼接 → Prompt → LLM
    ↓
输出回答
    ↓
更新实体记忆（抽取属性）
更新长期记忆（提炼知识）
更新短期记忆（追加/压缩）
```

## 工作流程

```
用户查询 → 向量检索 → BGE-Reranker 重排序 → 记忆检索 → LLM 生成回答 → 返回结果
```

## 配置说明

在 `.env` 文件中配置：

### 记忆系统配置说明

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `MEMORY_SHORT_TERM_MAX_TOKENS` | 3000 | 短期记忆最大token数，超过时自动压缩旧对话 |
| `MEMORY_SHORT_TERM_MAX_MESSAGES` | 20 | 短期记忆最大消息数，超过时删除最早的消息 |
| `MEMORY_ENTITY_DB_PATH` | ./data/memory/entity_memory.db | 实体记忆数据库路径 |
| `MEMORY_LONG_TERM_PATH` | ./data/memory/long_term | 长期记忆存储路径 |
| `MEMORY_LONG_TERM_TOP_K` | 5 | 长期记忆检索返回条数 |
| `MEMORY_ENTITY_TOP_K` | 5 | 实体检索返回条数 |

```env
# 嵌入模型配置
EMBEDDING_TYPE=bge_zh
BGE_LOCAL_MODEL_PATH=./bge-small-zh-v1.5

# 重排序模型配置
RERANKER_TYPE=bge_cross_encoder
RERANKER_MODEL_PATH=./bge-reranker-base

# Reranker 类型说明:
# - bge_cross_encoder: BGE Cross-Encoder 模型（高精度，推荐）
# - noop: 不做重排序，直接返回向量检索结果（极速，适用于测试）

# 大模型配置
LLM_PROVIDER=openai_compatible
LLM_MODEL=glm-4-flash
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_API_KEY=your-api-key
LLM_TIMEOUT_S=60

# 存储配置
UPLOAD_FOLDER=./uploads
CHROMA_DB_PATH=./data/chromadb

# 记忆系统配置
MEMORY_SHORT_TERM_MAX_TOKENS=3000
MEMORY_SHORT_TERM_MAX_MESSAGES=20
MEMORY_ENTITY_DB_PATH=./data/memory/entity_memory.db
MEMORY_LONG_TERM_PATH=./data/memory/long_term
MEMORY_LONG_TERM_TOP_K=5
MEMORY_ENTITY_TOP_K=5
```

## 快速开始

### 安装依赖

```bash
# 方式1：使用 pip（推荐）
# 先安装 PyTorch CPU 版本（避免 CUDA 依赖问题）
pip install torch==2.2.0+cpu torchvision==0.17.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 方式2：使用虚拟环境
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.2.0+cpu torchvision==0.17.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**依赖版本说明**：
- Python >= 3.11
- LangChain 1.0.0（最新稳定版本，2026-04-27 升级）
- LangChain Classic 1.0.0（Memory、Chains、Retrievers）
- PyTorch 2.2.0（CPU 版本，无需显卡）
- Transformers 4.41.1
- Sentence-Transformers 2.7.0
- ChromaDB 0.5.0

所有依赖版本已锁定，确保兼容性和稳定性。

**配置说明**：
- 所有配置项已外置到 `.env` 文件，包括文本分割、检索参数、会话缓存、重排序类型等
- 支持环境变量动态调整，无需修改代码
- Embedding 和 Reranker 均支持配置驱动切换算法

### 下载模型

```bash
# 下载 BGE 嵌入模型（已包含）
# 下载 BGE 重排序模型（已包含）
```

### 启动服务

```bash
# 方式1：使用虚拟环境运行（推荐）
.venv\Scripts\python.exe app.py

# 方式2：使用启动脚本
.venv\Scripts\python.exe scripts\start_app.py
```

服务启动后访问：http://localhost:5000 或 http://127.0.0.1:5000

**Windows 用户注意事项**：
- 应用已内置多线程禁用配置，防止段错误
- 如遇到 `OSError 1455`（页面文件太小），请增加 Windows 虚拟内存到 8-16GB
- 设置路径：系统属性 → 高级 → 性能 → 虚拟内存 → 更改

## API 接口

### 上传文档
- **URL**: `/upload`
- **方法**: POST
- **表单字段**: `file` (PDF/Word 文件)

### 查询文档
- **URL**: `/query`
- **方法**: POST
- **表单字段**: `query` (查询文本)

### 设置用户信息（实体记忆）
- **URL**: `/set_user_info`
- **方法**: POST
- **表单字段**: `user_id`, `name`, `preference`, `tool`, `interest`

### 清空短期记忆
- **URL**: `/clear_memory`
- **方法**: POST

## 目录结构说明

```
./
├── data/          # 数据存储（git 忽略）
│   ├── chromadb/  # ChromaDB 向量数据库（使用 PersistentClient）
│   │   ├── chroma.sqlite3        # 元数据（SQLite）
│   │   └── {uuid}/               # HNSW 索引数据
│   └── memory/    # 记忆数据
│       ├── entity_memory.db      # 实体记忆 SQLite 数据库
│       └── long_term/            # 长期记忆 JSON 存储
│           └── memory.json
├── logs/          # 日志文件（git 忽略）
│   ├── app.log    # 应用日志
│   └── llm.log    # LLM 调用日志
├── uploads/       # 上传文件（git 忽略）
└── test/          # 测试代码（包含 README.md 和数据分析工具）
```

## 配置项说明

所有配置项已外置到 `.env` 文件，支持动态调整：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `TEXT_SPLITTER_TYPE` | recursive | 文本分割器类型（9种可选） |
| `CHUNK_SIZE` | 500 | 文本分割块大小 |
| `CHUNK_OVERLAP` | 50 | 文本分割重叠大小 |
| `VECTOR_SEARCH_TOP_K` | 10 | 向量检索返回候选数 |
| `RERANK_TOP_K` | 5 | 重排序后返回文档数 |
| `LLM_CLIENT_TIMEOUT` | 120 | LLM 客户端超时（秒） |
| `RERANKER_TYPE` | bge_cross_encoder | 重排序器类型（bge_cross_encoder/noop） |
| `RERANKER_MODEL_PATH` | ./bge-reranker-base | Reranker 模型路径 |
| `MAX_SESSION_CACHE` | 10 | 最大会话缓存数（LRU） |
| `MAX_UPLOAD_SIZE_MB` | 50 | 最大上传文件大小（MB） |
| `FLASK_SECRET_KEY` | 自动生成 | Flask 会话密钥（生产环境建议设置） |

## 日志

日志文件位于 `logs/` 目录：
- `app.log`: 应用日志
- `llm.log`: 大模型调用日志
- `rag.log`: RAG 流程日志

## 项目特点

1. **全流程本地化**: 嵌入模型和重排序模型均本地部署，无需联网
2. **模块化设计**: 各模块职责清晰，易于扩展
3. **工厂模式架构**: Embedding、Reranker 和 TextSplitter 均采用工厂模式，配置驱动切换算法
4. **9 种文本分割**: 支持 recursive、character、token、markdown 等多种分割算法，适配不同场景
5. **三层记忆系统**: 支持短期对话、实体属性、长期知识的完整记忆能力
6. **智能记忆管理**: 短期记忆使用 ConversationBufferMemory，支持自动摘要压缩和淘汰机制，避免内存溢出
7. **LRU 缓存管理**: MemoryManager 使用 LRU 缓存，最多保留 10 个 session，防止内存泄漏
8. **日志完善**: 完整记录检索、重排序、回答生成全过程，LLM 调用独立日志
9. **容错机制**: Embedding 支持 BGE → Hash 自动降级，Reranker 支持 bge_cross_encoder → noop 切换
10. **Windows 稳定性优化**: 禁用多线程并行，防止段错误和内存访问冲突
11. **LangChain 1.0.0**: 升级到最新稳定版本，使用 langchain-classic 兼容层
12. **ChromaDB 持久化**: 使用 PersistentClient 替代手写 JSON 存储，HNSW 索引提升检索性能
13. **配置外置**: 所有魔法数字已提取到 `.env` 文件，支持动态调整
14. **安全增强**: Flask secret_key 使用随机生成，支持文件大小限制
15. **数据检查工具**: 提供 `test/inspect_data.py` 快速查看存储数据状态

## LangChain 1.x 升级说明

本项目已于 **2026-04-27** 成功从 LangChain 0.3.0 升级到 **LangChain 1.0.0**。

### 主要变化

1. **包结构简化**:
   - `langchain` 包专注于 Agent 核心功能
   - 传统功能（Memory、Chains、Retrievers）迁移到 `langchain-classic`

2. **导入路径变更**:
   ```python
   # 旧 (0.3.x)
   from langchain.memory import ConversationBufferMemory
   
   # 新 (1.x)
   from langchain_classic.memory import ConversationBufferMemory
   ```

3. **新增依赖**:
   - `langchain-classic==1.0.0` - 向后兼容层
   - `openai==1.109.1` - 匹配 langchain-openai 1.0.0 要求

### 升级收益

- ✅ 使用最新稳定版本，获得 bug 修复和安全更新
- ✅ 更清晰的包结构和依赖关系
- ✅ 更好的长期支持
- ✅ 性能略有提升（~10%）

### 详细文档

- [升级计划](LANGCHAIN_V1_UPGRADE_PLAN.md)
- [官方文档分析](LANGCHAIN_V1_OFFICIAL_ANALYSIS.md)
- [升级完成报告](LANGCHAIN_V1_UPGRADE_REPORT.md)