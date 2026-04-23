from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask import jsonify
import os
import secrets
import json
import uuid
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

# CRITICAL: Disable parallelism to prevent segmentation fault on Windows
# Must be set before importing torch, transformers, or sentence-transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism
os.environ["OMP_NUM_THREADS"] = "1"             # Force single thread for OpenMP
os.environ["MKL_NUM_THREADS"] = "1"             # Disable MKL threading
os.environ["OPENBLAS_NUM_THREADS"] = "1"        # Disable OpenBLAS threading
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"      # Disable vecLib threading
os.environ["NUMEXPR_NUM_THREADS"] = "1"         # Disable numexpr threading

from langchain_core.documents import Document
from rag_service import answer_with_rag, retrieve_and_rerank
from embeddings import EmbeddingFactory
from text_splitter import TextSplitterFactory
from logging_config import setup_logging, get_logger

load_dotenv()

setup_logging(log_level="INFO", log_file="app.log")

app = Flask(__name__)
# Generate secret key: use env var if set, otherwise generate random key
secret_key = os.getenv('FLASK_SECRET_KEY', '').strip()
if not secret_key:
    secret_key = secrets.token_hex(32)
app.secret_key = secret_key
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', './uploads')
app.config['CHROMA_DB_PATH'] = os.getenv('CHROMA_DB_PATH', './data/chromadb')
app.config['PERMANENT_SESSION_LIFETIME'] = 3600
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_UPLOAD_SIZE_MB', '50')) * 1024 * 1024  # 50MB default
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

# Configuration constants
TEXT_SPLITTER_TYPE = os.getenv('TEXT_SPLITTER_TYPE', 'recursive').strip().lower()
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))
VECTOR_SEARCH_TOP_K = int(os.getenv('VECTOR_SEARCH_TOP_K', '15'))
RERANK_TOP_K = int(os.getenv('RERANK_TOP_K', '5'))
MAX_SESSION_CACHE = int(os.getenv('MAX_SESSION_CACHE', '10'))

def get_or_create_session_id():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session.permanent = True
    return session['session_id']

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

# LRU Cache for MemoryManager to prevent memory leak
from collections import OrderedDict

class MemoryManagerCache:
    """LRU Cache for MemoryManager with max size limit"""
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.cache = OrderedDict()
    
    def get(self, session_id: str) -> MemoryManager:
        if session_id in self.cache:
            self.cache.move_to_end(session_id)  # Mark as recently used
            return self.cache[session_id]
        return None
    
    def put(self, session_id: str, manager: MemoryManager):
        if session_id in self.cache:
            self.cache.move_to_end(session_id)
        else:
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                oldest_session = next(iter(self.cache))
                del self.cache[oldest_session]
                app_logger.info(f"MemoryManager cache evicted: {oldest_session}")
            self.cache[session_id] = manager
    
    def clear(self, session_id: str = None):
        if session_id:
            self.cache.pop(session_id, None)
        else:
            self.cache.clear()

memory_cache = MemoryManagerCache(max_size=MAX_SESSION_CACHE)

def get_memory_manager(session_id: str) -> MemoryManager:
    manager = memory_cache.get(session_id)
    if manager is None:
        manager = MemoryManager(session_id)
        memory_cache.put(session_id, manager)
        app_logger.info(f"MemoryManager created for session: {session_id}, cache size: {len(memory_cache.cache)}")
    return manager

def load_document(file_path):
    if file_path.endswith('.pdf'):
        try:
            from langchain_community.document_loaders import PyPDFLoader
        except ImportError as exc:
            raise RuntimeError("缺少 pypdf 依赖。请运行 `pip install pypdf`") from exc
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        try:
            from langchain_community.document_loaders import Docx2txtLoader
        except ImportError as exc:
            raise RuntimeError("缺少 python-docx 依赖。请运行 `pip install python-docx`") from exc
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    return loader.load()

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

def get_chroma_client():
    """Get or create ChromaDB client singleton"""
    if not hasattr(get_chroma_client, '_client'):
        get_chroma_client._client = chromadb.PersistentClient(
            path=app.config['CHROMA_DB_PATH'],
            settings=Settings(
                anonymized_telemetry=False,
                # Disable ChromaDB's default embedding function
                # We'll use our own BGE embedding
            )
        )
    return get_chroma_client._client

def save_to_chroma(chunks, collection_name="documents"):
    """Save documents to ChromaDB"""
    client = get_chroma_client()
    
    # Create ChromaDB-compatible embedding function adapter
    class EmbeddingFunctionAdapter:
        def __init__(self, embedding_instance):
            self.embedding = embedding_instance
        
        def __call__(self, input):
            if isinstance(input, str):
                return self.embedding.embed_query(input)
            else:
                return self.embedding.embed_documents(input)
    
    chroma_embedding = EmbeddingFunctionAdapter(get_embeddings())
    
    # Get or create collection
    try:
        collection = client.get_collection(collection_name)
    except:
        collection = client.create_collection(
            name=collection_name,
            embedding_function=chroma_embedding
        )
    
    # Prepare data for batch insertion
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata or {} for chunk in chunks]
    ids = [f"doc_{uuid.uuid4()}" for _ in chunks]
    
    # Add to ChromaDB
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    app_logger.info(f"Saved {len(chunks)} chunks to ChromaDB collection: {collection_name}")
    return collection

def get_chroma_db(collection_name="documents"):
    """Get ChromaDB collection for querying"""
    client = get_chroma_client()
    
    # Create ChromaDB-compatible embedding function adapter
    class EmbeddingFunctionAdapter:
        def __init__(self, embedding_instance):
            self.embedding = embedding_instance
        
        def __call__(self, input):
            # ChromaDB expects 'input' parameter
            if isinstance(input, str):
                return self.embedding.embed_query(input)
            else:
                return self.embedding.embed_documents(input)
    
    chroma_embedding = EmbeddingFunctionAdapter(get_embeddings())
    
    try:
        collection = client.get_collection(collection_name)
    except:
        # Return empty collection if not exists
        collection = client.create_collection(
            name=collection_name,
            embedding_function=chroma_embedding
        )
    
    # Wrap collection to provide similarity_search interface
    class ChromaDBWrapper:
        def __init__(self, collection):
            self.collection = collection
        
        def similarity_search(self, query, k=5):
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas"]
            )
            
            docs = []
            if results['documents'] and results['documents'][0]:
                for i, content in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    docs.append(Document(page_content=content, metadata=metadata))
            return docs
    
    return ChromaDBWrapper(collection)

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
                
                docs = retrieve_and_rerank(query_text, db, top_k=RERANK_TOP_K)
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