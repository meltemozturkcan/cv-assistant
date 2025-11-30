import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")
# Model ayarları
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# ChromaDB ayarları
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "cv_collection"

# Chunk ayarları
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# CV dosya yolu
CV_FILE_PATH = "./cv.md"