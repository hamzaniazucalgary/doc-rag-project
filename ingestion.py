"""PDF ingestion pipeline: load, chunk, embed."""
import os
from typing import Callable, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS,
    EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, MIN_EXTRACTED_CHARS
)
from utils import compute_file_hash
from storage import VectorStore


def load_pdf(file_path: str) -> tuple[list[dict], int]:
    """
    Load PDF and extract text with page metadata.
    
    Returns: (pages, total_chars)
        pages: list of {"content": str, "page": int}
        total_chars: total characters extracted
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    pages = []
    total_chars = 0
    
    for doc in documents:
        content = doc.page_content
        page_num = doc.metadata.get("page", 0) + 1  # 1-indexed
        pages.append({"content": content, "page": page_num})
        total_chars += len(content)
    
    return pages, total_chars


def chunk_pages(
    pages: list[dict],
    doc_id: str,
    doc_name: str
) -> list[dict]:
    """
    Split pages into chunks with metadata.
    
    Returns list of:
        {
            "id": unique chunk id,
            "content": chunk text,
            "metadata": {doc_id, doc_name, page, chunk_index}
        }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS
    )
    
    chunks = []
    chunk_index = 0
    
    for page in pages:
        page_chunks = splitter.split_text(page["content"])
        
        for chunk_text in page_chunks:
            chunks.append({
                "id": f"{doc_id}-{chunk_index}",
                "content": chunk_text,
                "metadata": {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page": page["page"],
                    "chunk_index": chunk_index
                }
            })
            chunk_index += 1
    
    return chunks


def embed_chunks(
    chunks: list[dict],
    progress_callback: Optional[Callable[[float], None]] = None
) -> list[list[float]]:
    """
    Generate embeddings for chunks in batches.
    
    Args:
        chunks: list of chunk dicts with "content" key
        progress_callback: optional function(progress: float) for UI updates
    
    Returns: list of embedding vectors
    """
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    
    embeddings = []
    total_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
    
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        batch_embeddings = embedder.embed_documents(batch)
        embeddings.extend(batch_embeddings)
        
        if progress_callback:
            progress = min(1.0, (i + len(batch)) / len(texts))
            progress_callback(progress)
    
    return embeddings


def ingest_pdf(
    file_path: str,
    file_name: str,
    file_bytes: bytes,
    store: VectorStore,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> dict:
    """
    Full ingestion pipeline.
    
    Args:
        file_path: path to PDF file
        file_name: original filename for display
        file_bytes: raw file bytes for hashing
        store: VectorStore instance
        progress_callback: optional function(stage: str, progress: float)
    
    Returns:
        {
            "status": "ingested" | "skipped" | "error",
            "doc_id": str,
            "doc_name": str,
            "pages": int,
            "chunks": int,
            "error": optional str
        }
    """
    doc_id = compute_file_hash(file_bytes)
    
    # Skip if already exists
    if store.doc_exists(doc_id):
        return {
            "status": "skipped",
            "doc_id": doc_id,
            "doc_name": file_name,
            "pages": 0,
            "chunks": 0
        }
    
    try:
        # Stage 1: Load PDF
        if progress_callback:
            progress_callback("Loading PDF", 0.1)
        
        pages, total_chars = load_pdf(file_path)
        
        # Validate content
        if total_chars < MIN_EXTRACTED_CHARS:
            return {
                "status": "error",
                "doc_id": doc_id,
                "doc_name": file_name,
                "error": "PDF appears empty or scanned (no extractable text)"
            }
        
        # Stage 2: Chunk
        if progress_callback:
            progress_callback("Chunking", 0.2)
        
        chunks = chunk_pages(pages, doc_id, file_name)
        
        # Stage 3: Embed
        def embed_progress(p):
            if progress_callback:
                progress_callback("Embedding", 0.2 + p * 0.7)
        
        embeddings = embed_chunks(chunks, embed_progress)
        
        # Stage 4: Store
        if progress_callback:
            progress_callback("Storing", 0.95)
        
        store.add_chunks(
            ids=[c["id"] for c in chunks],
            embeddings=embeddings,
            documents=[c["content"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks]
        )
        
        if progress_callback:
            progress_callback("Complete", 1.0)
        
        return {
            "status": "ingested",
            "doc_id": doc_id,
            "doc_name": file_name,
            "pages": len(pages),
            "chunks": len(chunks)
        }
    
    except Exception as e:
        return {
            "status": "error",
            "doc_id": doc_id,
            "doc_name": file_name,
            "error": str(e)
        }
