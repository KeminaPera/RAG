from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask import jsonify
import os
import hashlib
import json
import math
import uuid
from pathlib import Path
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from rag_service import answer_with_rag, retrieve_and_rerank
from embeddings import EmbeddingFactory
from logging_config import setup_logging, get_logger

load_dotenv()

setup_logging(log_level="INFO", log_file="app.log")

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', './uploads')
app.config['CHROMA_DB_PATH'] = os.getenv('CHROMA_DB_PATH', './chromadb')
app.config['PERMANENT_SESSION_LIFETIME'] = 3600
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CHROMA_DB_PATH'], exist_ok=True)

app_logger = get_logger('app')

try:
    from memory_manager import MemoryManager
    MEMORY_ENABLED = True
    app_logger.info("Memory Manager 模块加载成功")
except ImportError as e:
    MEMORY_ENABLED = False
    app_logger.warning(f"Memory Manager 模块加载失败: {e}")

# Preload reranker at startup to avoid first-request delay
try:
    from rag_service import get_reranker, RERANKER_AVAILABLE
    if RERANKER_AVAILABLE:
        app_logger.info("正在预加载 Reranker 模型...")
        _ = get_reranker()
        app_logger.info("Reranker 模型预加载完成")
except Exception as e:
    app_logger.warning(f"Reranker 预加载失败: {e}")

memory_managers = {}

def get_memory_manager(session_id: str) -> MemoryManager:
    if session_id not in memory_managers:
        memory_managers[session_id] = MemoryManager(session_id)
    return memory_managers[session_id]

def get_or_create_session_id():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session.permanent = True
    return session['session_id']

class SimpleVectorDB:
    def __init__(self, persist_directory, collection_name, embedding_function):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.db_file = os.path.join(self.persist_directory, f"{self.collection_name}.json")
        self.items = []
        self._load()

    def _load(self):
        if os.path.exists(self.db_file):
            with open(self.db_file, "r", encoding="utf-8") as f:
                self.items = json.load(f)

    def persist(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        with open(self.db_file, "w", encoding="utf-8") as f:
            json.dump(self.items, f, ensure_ascii=False)

    def add_documents(self, documents):
        texts = [doc.page_content for doc in documents]
        vectors = self.embedding_function.embed_documents(texts)
        for doc, vector in zip(documents, vectors):
            self.items.append({
                "content": doc.page_content,
                "metadata": doc.metadata or {},
                "embedding": vector
            })

    def similarity_search(self, query_text, k=5):
        query_vec = self.embedding_function.embed_query(query_text)
        query_norm = math.sqrt(sum(v * v for v in query_vec)) or 1.0
        scored = []
        for item in self.items:
            vec = item["embedding"]
            vec_norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            score = sum(a * b for a, b in zip(query_vec, vec)) / (query_norm * vec_norm)
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_items = scored[:k]
        return [Document(page_content=i["content"], metadata=i["metadata"]) for _, i in top_items]

embeddings = None

def get_embeddings():
    global embeddings
    if embeddings is None:
        embed_type = os.getenv('EMBEDDING_TYPE', 'bge_zh')
        embeddings = EmbeddingFactory.get_embedding(embed_type=embed_type)
        actual_type = type(embeddings).__name__
        fallback = getattr(embeddings, '_fallback_mode', False)
        fallback_reason = getattr(embeddings, '_fallback_reason', None)
        if not hasattr(embeddings, 'model') or getattr(embeddings, 'model', None) is None:
            fallback = True
            fallback_reason = fallback_reason or 'model attribute missing'
        if fallback:
            actual_type = f'{actual_type}(fallback)'
        app_logger.info(
            'Embedding initialized; requested=%s actual=%s fallback=%s reason=%s',
            embed_type,
            actual_type,
            fallback,
            fallback_reason,
        )
    return embeddings

def load_document(file_path):
    if file_path.endswith('.pdf'):
        try:
            from langchain.document_loaders import PyPDFLoader
        except ImportError as exc:
            raise RuntimeError("缺少 pypdf 依赖。请运行 `pip install pypdf`") from exc
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        try:
            from langchain.document_loaders import Docx2txtLoader
        except ImportError as exc:
            raise RuntimeError("缺少 python-docx 依赖。请运行 `pip install python-docx`") from exc
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    return loader.load()

def split_documents(documents):
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    return text_splitter.split_documents(documents)

def save_to_chroma(chunks, collection_name="documents"):
    db = SimpleVectorDB(
        persist_directory=app.config['CHROMA_DB_PATH'],
        collection_name=collection_name,
        embedding_function=get_embeddings(),
    )
    db.add_documents(chunks)
    db.persist()
    return db

def get_chroma_db(collection_name="documents"):
    return SimpleVectorDB(
        persist_directory=app.config['CHROMA_DB_PATH'],
        collection_name=collection_name,
        embedding_function=get_embeddings(),
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            app_logger.info('Upload received; file=%s path=%s', filename, file_path)

            try:
                documents = load_document(file_path)
                chunks = split_documents(documents)
                for i, chunk in enumerate(chunks):
                    chunk.metadata = chunk.metadata or {}
                    chunk.metadata["source_file"] = filename
                    chunk.metadata["chunk_id"] = i
                embedding_type = type(get_embeddings()).__name__
                app_logger.info('Begin embedding conversion; file=%s chunk_count=%s embedding=%s', filename, len(chunks), embedding_type)
                save_to_chroma(chunks)
                app_logger.info('Upload processed successfully; file=%s chunk_count=%s', filename, len(chunks))
                flash(f'Successfully processed {len(chunks)} chunks from {filename}')
            except Exception as e:
                app_logger.exception('Upload processing failed; file=%s error=%s', filename, e)
                flash(f'Error processing file: {str(e)}')
            finally:
                try:
                    os.remove(file_path)
                except Exception:
                    pass
            return redirect(url_for('upload_file'))

    return render_template('upload.html')

@app.route('/query', methods=['GET', 'POST'])
def query():
    results = []
    answer = ""
    sources = []
    if request.method == 'POST':
        query_text = request.form.get('query', '')
        if query_text:
            try:
                session_id = get_or_create_session_id()
                db = get_chroma_db()
                app_logger.info(f'Query received: {query_text[:50]}... session_id={session_id}')
                
                docs = retrieve_and_rerank(query_text, db, top_k=5)
                app_logger.info(f'Retrieved {len(docs)} documents after reranking')
                
                memory_manager = None
                if MEMORY_ENABLED:
                    memory_manager = get_memory_manager(session_id)
                    memory_manager.add_short_term("user", query_text)
                    app_logger.info("Short-term memory updated")
                
                results = [{'content': doc.page_content, 'metadata': doc.metadata} for doc in docs]
                
                user_id = session.get('user_id', 'default_user')
                rag = answer_with_rag(query_text, docs, memory_manager, user_id)
                answer = rag.answer
                sources = rag.sources
                
                if MEMORY_ENABLED and memory_manager:
                    memory_manager.add_short_term("assistant", answer)
                    memory_manager.add_long_term(
                        content=f"Q: {query_text}\nA: {answer[:200]}",
                        metadata={"type": "qa_pair", "session_id": session_id}
                    )
                    app_logger.info("Memory updated after response")
                
                app_logger.info('Query completed successfully')
            except Exception as e:
                app_logger.exception(f'Query failed: {e}')
                flash(f'Error searching: {str(e)}')
    
    return render_template('query.html', results=results, answer=answer, sources=sources)

@app.route('/set_user_info', methods=['POST'])
def set_user_info():
    if not MEMORY_ENABLED:
        return jsonify({"status": "error", "message": "Memory module not enabled"}), 500
    
    try:
        session_id = get_or_create_session_id()
        user_id = request.form.get('user_id', 'default_user')
        attributes = {
            "name": request.form.get('name', ''),
            "preference": request.form.get('preference', ''),
            "tool": request.form.get('tool', ''),
            "interest": request.form.get('interest', '')
        }
        
        memory_manager = get_memory_manager(session_id)
        memory_manager.update_entity(user_id, attributes)
        session['user_id'] = user_id
        
        app_logger.info(f"User info updated: {user_id}")
        return jsonify({"status": "success", "message": "User info updated"})
    except Exception as e:
        app_logger.exception(f"Set user info failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    if not MEMORY_ENABLED:
        return jsonify({"status": "error", "message": "Memory module not enabled"}), 500
    
    try:
        session_id = get_or_create_session_id()
        memory_manager = get_memory_manager(session_id)
        memory_manager.clear_short_term()
        app_logger.info("Short-term memory cleared")
        return jsonify({"status": "success", "message": "Memory cleared"})
    except Exception as e:
        app_logger.exception(f"Clear memory failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'docx'}

def secure_filename(filename):
    sanitized = os.path.basename(filename).replace('\x00', '')
    if not sanitized:
        return "uploaded_file"
    return sanitized

if __name__ == '__main__':
    # Disable debug mode to prevent dual-process memory issue
    # Debug mode creates a reloader process that doubles memory usage
    app.run(debug=False, host='0.0.0.0', port=5000)