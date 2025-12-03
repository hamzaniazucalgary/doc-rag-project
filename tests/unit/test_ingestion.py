import pytest
from unittest.mock import MagicMock, patch
from ingestion import load_pdf, chunk_pages, embed_chunks, ingest_pdf
from storage import VectorStore

def test_load_pdf():
    with patch("ingestion.PyPDFLoader") as mock_loader:
        instance = mock_loader.return_value
        doc_mock = MagicMock()
        doc_mock.page_content = "Test content"
        doc_mock.metadata = {"page": 0}
        instance.load.return_value = [doc_mock, doc_mock]
        
        pages, total_chars = load_pdf("dummy.pdf")
        
        assert len(pages) == 2
        assert pages[0]["page"] == 1 # 1-indexed
        assert total_chars == len("Test content") * 2

def test_chunk_pages():
    # Use text with spaces to ensure splitter can split on separators
    text = "word " * 500  # approx 2500 chars
    pages = [{"content": text, "page": 1}]
    chunks = chunk_pages(pages, "doc1", "test.pdf")
    
    # CHUNK_SIZE is 1500, so 2500 chars should be split
    assert len(chunks) >= 2
    assert chunks[0]["metadata"]["doc_id"] == "doc1"
    assert chunks[0]["metadata"]["page"] == 1

def test_embed_chunks(mock_openai_embeddings):
    chunks = [{"content": "text1"}, {"content": "text2"}]
    embeddings = embed_chunks(chunks)
    
    assert len(embeddings) == 2
    # Mock returns list of 1536 floats
    assert len(embeddings[0]) == 1536

def test_ingest_pdf_success(temp_persist_dir, mock_chroma_client, mock_openai_embeddings):
    store = VectorStore(temp_persist_dir)
    store.doc_exists = MagicMock(return_value=False)
    
    with patch("ingestion.load_pdf") as mock_load:
        mock_load.return_value = ([{"content": "test", "page": 1}], 100)
        
        result = ingest_pdf(
            file_path="test.pdf",
            file_name="test.pdf",
            file_bytes=b"bytes",
            store=store
        )
        
        assert result["status"] == "ingested"
        assert result["chunks"] > 0
        store.collection.add.assert_called()

def test_ingest_pdf_already_exists(temp_persist_dir, mock_chroma_client):
    store = VectorStore(temp_persist_dir)
    store.doc_exists = MagicMock(return_value=True)
    
    result = ingest_pdf("path", "name", b"bytes", store)
    assert result["status"] == "skipped"

def test_ingest_pdf_empty(temp_persist_dir, mock_chroma_client):
    store = VectorStore(temp_persist_dir)
    store.doc_exists = MagicMock(return_value=False)
    
    with patch("ingestion.load_pdf") as mock_load:
        # Return very little text
        mock_load.return_value = ([{"content": "", "page": 1}], 5)
        
        result = ingest_pdf("path", "name", b"bytes", store)
        assert result["status"] == "error"
        assert "empty" in result["error"].lower()
