import pytest
from unittest.mock import patch, MagicMock
from ingestion import ingest_pdf
from retrieval import retrieve_context
from storage import VectorStore

def test_ingest_and_retrieve(temp_persist_dir, mock_openai_embeddings, mock_openai_chat):
    # 1. Setup Store
    store = VectorStore(temp_persist_dir)
    
    # 2. Ingest Document
    with patch("ingestion.load_pdf") as mock_load:
        # Mock PDF content
        mock_load.return_value = ([
            {"content": "The secret code is 12345.", "page": 1}
        ], 300)
        
        ingest_pdf(
            file_path="dummy.pdf",
            file_name="dummy.pdf",
            file_bytes=b"dummy",
            store=store
        )
    
    # DEBUG: Ensure ingestion happened
    assert store.collection.count() > 0, "Store should not be empty after ingestion"
    
    # 3. Retrieve
    # We need to ensure query embedding matches the "dummy" embedding from conftest
    # mock_openai_embeddings returns [0.1, ...]
    # ChromaDB will calculate cosine distance.
    # Since all embeddings are identical in the mock ([0.1]*1536), distance is 0 (perfect match).
    
    # Mock rewrite_query to return original
    with patch("retrieval.rewrite_query", side_effect=lambda q, h: q):
        result = retrieve_context("What is the code?", store, [])
    
    assert len(result["chunks"]) > 0
    assert "12345" in result["chunks"][0]["content"]