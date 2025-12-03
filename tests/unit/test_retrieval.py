import pytest
from unittest.mock import MagicMock, patch
from retrieval import rewrite_query, get_query_embedding, retrieve_context
from storage import VectorStore

def test_rewrite_query_no_history(mock_openai_chat):
    q = "What is X?"
    rewritten = rewrite_query(q, [])
    assert rewritten == q
    mock_openai_chat.assert_not_called()

def test_rewrite_query_with_history(mock_openai_chat):
    q = "it?"
    history = [{"role": "user", "content": "What is X?"}]
    
    # Mock LLM response
    mock_openai_chat.return_value.invoke.return_value.content = "What is X?"
    
    rewritten = rewrite_query(q, history)
    assert rewritten == "What is X?"
    mock_openai_chat.assert_called()

def test_get_query_embedding(mock_openai_embeddings):
    emb = get_query_embedding("test")
    assert len(emb) == 1536

def test_retrieve_context(temp_persist_dir, mock_chroma_client, mock_openai_embeddings, mock_openai_chat):
    store = VectorStore(temp_persist_dir)
    
    # Setup store query return (mocked in conftest but ensuring here)
    store.collection.query.return_value = {
        "ids": [["1"]],
        "documents": [["content"]],
        "metadatas": [[{"doc_name": "doc", "page": 1}]],
        "distances": [[0.1]]
    }
    
    result = retrieve_context("query", store, [])
    
    assert len(result["chunks"]) == 1
    assert result["chunks"][0]["content"] == "content"
    assert result["is_low_confidence"] is False
