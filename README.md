# 文档向量检索系统 (RAG)

基于 LangChain 和 Flask 的智能文档检索平台，支持 PDF/Word 文档上传、向量存储、语义检索和大模型回答生成。

## 项目架构

```
├── app.py                 # Flask Web 应用入口
├── start_app.py           # 启动脚本
├── requirements.txt       # 依赖清单
├── .env                   # 环境配置
├── embeddings.py          # 向量嵌入模块（BGE-zh 模型）
├── reranker.py            # 重排序模块（bge-reranker-base）
├── rag_service.py         # RAG 核心服务
├── llm_client.py          # 大模型客户端
├── templates/             # HTML 模板
│   ├── index.html
│   ├── upload.html
│   └── query.html
├── chromadb/              # 向量数据库存储
├── uploads/               # 上传文件临时目录
├── logs/                  # 日志目录
├── bge-small-zh-v1.5/     # BGE 中文嵌入模型
└── bge-reranker-base/     # BGE 重排序模型
```

## 技术栈

| 分类 | 技术 | 版本 | 说明 |
|------|------|------|------|
| Web 框架 | Flask | 2.x | 轻量级 Web 框架 |
| 向量数据库 | ChromaDB | 0.4.x | 轻量级向量存储 |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 | - | 中文语义嵌入 |
| 重排序模型 | BAAI/bge-reranker-base | - | 检索结果重排序 |
| LLM 框架 | LangChain | 0.1.x | 大语言模型编排 |
| 文档解析 | pypdf / python-docx | - | PDF/Word 解析 |

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

## 工作流程

```
用户查询 → 向量检索 → BGE-Reranker 重排序 → 大模型生成回答 → 返回结果
```

## 配置说明

在 `.env` 文件中配置：

```env
# 嵌入模型配置
EMBEDDING_TYPE=bge_zh
BGE_LOCAL_MODEL_PATH=./bge-small-zh-v1.5

# 重排序模型配置
RERANKER_MODEL_PATH=./bge-reranker-base

# 大模型配置
LLM_PROVIDER=openai_compatible
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your-api-key

# 存储配置
UPLOAD_FOLDER=./uploads
CHROMA_DB_PATH=./chromadb
```

## 快速开始

### 安装依赖

```bash
# 使用 uv 安装依赖
uv pip install flask langchain langchain-community langchain-text-splitters chromadb pypdf python-docx sentence-transformers python-dotenv requests
```

### 下载模型

```bash
# 下载 BGE 嵌入模型（已包含）
# 下载 BGE 重排序模型（已包含）
```

### 启动服务

```bash
# 方式1：直接运行
uv run python app.py

# 方式2：使用启动脚本
uv run python start_app.py
```

服务启动后访问：`http://localhost:5000`

## API 接口

### 上传文档
- **URL**: `/upload`
- **方法**: POST
- **表单字段**: `file` (PDF/Word 文件)

### 查询文档
- **URL**: `/query`
- **方法**: POST
- **表单字段**: `query` (查询文本)

## 日志

日志文件位于 `logs/` 目录：
- `app.log`: 应用日志
- `llm.log`: 大模型调用日志
- `rag.log`: RAG 流程日志

## 项目特点

1. **全流程本地化**: 嵌入模型和重排序模型均本地部署，无需联网
2. **模块化设计**: 各模块职责清晰，易于扩展
3. **日志完善**: 完整记录检索、重排序、回答生成全过程
4. **容错机制**: 支持嵌入模型回退到 Hash 方法