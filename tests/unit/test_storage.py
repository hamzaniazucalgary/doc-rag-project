import pytest
from unittest.mock import MagicMock
from storage import VectorStore

def test_init(temp_persist_dir, mock_chroma_client):
    store = VectorStore(temp_persist_dir)
    mock_chroma_client.assert_called_once()
    assert store.persist_dir == temp_persist_dir

def test_add_chunks(temp_persist_dir, mock_chroma_client):
    store = VectorStore(temp_persist_dir)
    
    ids = ["1", "2"]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    documents = ["doc1", "doc2"]
    metadatas = [{"a": 1}, {"b": 2}]
    
    store.add_chunks(ids, embeddings, documents, metadatas)
    store.collection.add.assert_called_once_with(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

def test_query(temp_persist_dir, mock_chroma_client):
    store = VectorStore(temp_persist_dir)
    
    # Mock return value is set in conftest, but we can override or assert on it
    result = store.query([0.1] * 1536)
    
    assert "ids" in result
    assert "documents" in result
    assert "metadatas" in result
    assert "distances" in result
    store.collection.query.assert_called_once()

def test_filter_by_threshold(temp_persist_dir, mock_chroma_client):
    store = VectorStore(temp_persist_dir)
    
    # Case 1: Good matches
    results = {
        "ids": ["1", "2"],
        "distances": [0.1, 0.2], # Threshold is 0.25
        "documents": ["d1", "d2"],
        "metadatas": [{}, {}]
    }
    filtered, is_low = store.filter_by_threshold(results)
    assert len(filtered["ids"]) == 2
    assert is_low is False
    
    # Case 2: Poor matches
    results_bad = {
        "ids": ["1", "2"],
        "distances": [0.5, 0.6],
        "documents": ["d1", "d2"],
        "metadatas": [{}, {}]
    }
    filtered_bad, is_low_bad = store.filter_by_threshold(results_bad)
    assert len(filtered_bad["ids"]) == 1 # Fallback to top 1
    assert is_low_bad is True

def test_doc_exists(temp_persist_dir, mock_chroma_client):
    store = VectorStore(temp_persist_dir)
    
    # Mock exists
    store.collection.get.return_value = {"ids": ["1"]}
    assert store.doc_exists("doc123") is True
    
    # Mock not exists
    store.collection.get.return_value = {"ids": []}
    assert store.doc_exists("doc456") is False

def test_delete_doc(temp_persist_dir, mock_chroma_client):
    store = VectorStore(temp_persist_dir)
    
    store.collection.get.return_value = {"ids": ["chunk1", "chunk2"]}
    count = store.delete_doc("doc1")
    
    store.collection.delete.assert_called_once_with(ids=["chunk1", "chunk2"])
    assert count == 2
