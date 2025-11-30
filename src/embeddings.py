import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.config import OPENAI_API_KEY, CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL


def get_embeddings():
    """OpenAI embedding modelini döndürür"""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )
    return embeddings


def create_vector_store(chunks):
    """Chunk'lardan vektör veritabanı oluşturur"""
    embeddings = get_embeddings()
    
    # Mevcut veritabanını sil (varsa)
    if os.path.exists(CHROMA_PERSIST_DIR):
        import shutil
        shutil.rmtree(CHROMA_PERSIST_DIR)
    
    # Yeni vektör veritabanı oluştur
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )
    
    print(f"✓ Vektör veritabanı oluşturuldu: {CHROMA_PERSIST_DIR}")
    return vector_store


def load_vector_store():
    """Mevcut vektör veritabanını yükler"""
    embeddings = get_embeddings()
    
    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    print("✓ Vektör veritabanı yüklendi")
    return vector_store