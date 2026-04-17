# 文档向量检索系统 (RAG)

基于 LangChain 和 Flask 的智能文档检索平台，支持 PDF/Word 文档上传、向量存储、语义检索和大模型回答生成。

## 项目架构

```
├── app.py                 # Flask Web 应用入口
├── start_app.py           # 启动脚本
├── requirements.txt       # 依赖清单（版本锁定）
├── .env                   # 环境配置（API密钥等敏感信息）
├── .gitignore             # Git忽略规则
├── embeddings.py          # 向量嵌入模块（BGE-zh 模型）
├── reranker.py            # 重排序模块（bge-reranker-base）
├── rag_service.py         # RAG 核心服务
├── llm_client.py          # 大模型客户端
├── memory_manager.py      # 三层记忆管理模块
├── logging_config.py      # 日志配置模块
├── download_reranker.py   # 模型下载工具脚本
├── templates/             # HTML 模板
│   ├── index.html         # 首页
│   ├── upload.html        # 文档上传页面
│   └── query.html         # 查询页面
├── data/                  # 数据存储目录（运行时生成）
│   ├── chromadb/          # 向量数据库存储
│   └── memory/            # 记忆数据（实体记忆 + 长期记忆）
├── test/                  # 测试目录
│   ├── test_bge_fallback.py
│   ├── test_embedding.py
│   ├── test_local_bge.py
│   ├── test_rag_flow.py
│   ├── test_reranker.py
│   ├── test_import.py
│   └── test_memory.py
├── uploads/               # 上传文件临时目录（运行时生成）
├── logs/                  # 日志目录（运行时生成）
├── bge-small-zh-v1.5/     # BGE 中文嵌入模型（本地部署）
└── bge-reranker-base/     # BGE 重排序模型（本地部署）
```

## 技术栈

| 分类 | 技术 | 版本 | 说明 |
|------|------|------|------|
| Web 框架 | Flask | 3.1.3 | 轻量级 Web 框架 |
| 向量数据库 | ChromaDB | 0.4.22 | 轻量级向量存储 |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 | - | 中文语义嵌入（本地） |
| 重排序模型 | BAAI/bge-reranker-base | - | 检索结果重排序（本地） |
| LLM 框架 | LangChain | 0.1.10 | 大语言模型编排 |
| 文档解析 | pypdf / python-docx | 6.10.2 / 1.2.0 | PDF/Word 解析 |
| 深度学习 | PyTorch | 2.2.1 (CPU) | 模型推理框架 |
| Transformer | Transformers | 4.38.2 | HuggingFace 模型库 |
| 向量化 | Sentence-Transformers | 2.5.1 | 文本嵌入生成 |
| 记忆系统 | 三层记忆架构 | - | 短期/实体/长期记忆 |

**环境要求**：Python >= 3.11

## 核心功能

### 1. 文档上传
- 支持 PDF (.pdf) 和 Word (.docx) 文件
- 自动文本分割（500字符/段，50字符重叠）
- 向量化存储到向量数据库

### 2. 语义检索
- 使用 BGE 模型进行向量检索
- 支持自定义相似度搜索

### 3. 重排序优化
- 使用 bge-reranker-base 对检索结果重排序
- 提升检索精度

### 4. 大模型回答
- 支持 OpenAI 兼容接口和 Ollama
- 基于检索上下文生成回答
- 自动引用来源文档

### 5. 三层记忆系统

| 记忆类型 | 实现方式 | 作用 | 特点 |
|---------|---------|------|------|
| **短期记忆** | ConversationSummaryBufferMemory | 维持当前对话连贯性 | 智能摘要压缩，自动淘汰旧对话 |
| **实体记忆** | SQLite + 向量检索 | 记住用户/对象属性 | 跨会话持久化 |
| **长期记忆** | JSON + 向量检索 | 记住知识/经验/规则 | 增量更新，越用越全 |

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
RERANKER_MODEL_PATH=./bge-reranker-base

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
pip install torch==2.2.1+cpu torchvision==0.17.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 方式2：使用 uv
uv pip install torch==2.2.1+cpu torchvision==0.17.1+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
```

**依赖版本说明**：
- Python >= 3.11
- LangChain 0.1.10（使用旧版 API，避免 0.2+ 破坏性变更）
- PyTorch 2.2.1（CPU 版本，无需显卡）
- Transformers 4.38.2
- Sentence-Transformers 2.5.1

所有依赖版本已锁定，确保兼容性。

### 下载模型

```bash
# 下载 BGE 嵌入模型（已包含）
# 下载 BGE 重排序模型（已包含）
```

### 启动服务

```bash
# 方式1：使用虚拟环境运行（推荐）
.venv\Scripts\python.exe app.py

# 方式2：使用 uv
uv run python app.py

# 方式3：使用启动脚本
.venv\Scripts\python.exe start_app.py
```

服务启动后访问：http://localhost:5000 或 http://127.0.0.1:5000

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
│   ├── chromadb/  # ChromaDB 向量数据库
│   └── memory/    # 记忆数据
│       ├── entity_memory.db    # 实体记忆 SQLite 数据库
│       └── long_term/          # 长期记忆 JSON 存储
├── logs/          # 日志文件（git 忽略）
├── uploads/       # 上传文件（git 忽略）
└── test/          # 测试代码
```

## 日志

日志文件位于 `logs/` 目录：
- `app.log`: 应用日志
- `llm.log`: 大模型调用日志
- `rag.log`: RAG 流程日志

## 项目特点

1. **全流程本地化**: 嵌入模型和重排序模型均本地部署，无需联网
2. **模块化设计**: 各模块职责清晰，易于扩展
3. **三层记忆系统**: 支持短期对话、实体属性、长期知识的完整记忆能力
4. **智能记忆管理**: 短期记忆使用 ConversationSummaryBufferMemory，支持自动摘要压缩和淘汰机制，避免内存溢出
5. **日志完善**: 完整记录检索、重排序、回答生成全过程
6. **容错机制**: 支持嵌入模型回退到 Hash 方法