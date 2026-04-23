from app import app
import os

if __name__ == '__main__':
    print("重新启动 Flask RAG 应用...")
    print("EMBEDDING_TYPE:", os.getenv('EMBEDDING_TYPE'))
    print("BGE_LOCAL_MODEL_PATH:", os.getenv('BGE_LOCAL_MODEL_PATH'))
    app.run(host='0.0.0.0', port=5000, debug=False)