from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CV_FILE_PATH, CHUNK_SIZE, CHUNK_OVERLAP


def load_cv_document():
    """CV dosyasını yükler"""
    loader = TextLoader(CV_FILE_PATH, encoding="utf-8")
    documents = loader.load()
    return documents


def split_documents(documents):
    """Dokümanları chunk'lara ayırır"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n---\n", "\n## ", "\n# ", "\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def load_and_split_cv():
    """CV'yi yükler ve chunk'lara ayırır"""
    documents = load_cv_document()
    chunks = split_documents(documents)
    print(f"✓ CV yüklendi: {len(chunks)} chunk oluşturuldu")
    return chunks