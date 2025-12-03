import pytest
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

@pytest.fixture(autouse=True)
def mock_env_setup():
    """Set up environment variables for testing."""
    original_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "sk-test-key-12345"
    yield
    if original_key:
        os.environ["OPENAI_API_KEY"] = original_key
    else:
        del os.environ["OPENAI_API_KEY"]

@pytest.fixture
def temp_persist_dir():
    """Create a temporary directory for testing persistence."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(autouse=True)
def mock_openai_embeddings():
    """Mock OpenAI Embeddings in all usage contexts."""
    mock_class = MagicMock()
    instance = mock_class.return_value
    # Mock embeddings to return a valid unit vector of length 1536
    # [1.0, 0.0, ... 0.0] has magnitude 1.0
    dummy_vec = [1.0] + [0.0] * 1535
    instance.embed_documents.side_effect = lambda texts: [dummy_vec for _ in texts]
    instance.embed_query.side_effect = lambda text: dummy_vec
    
    # Patch in all modules where it is imported
    with patch("ingestion.OpenAIEmbeddings", new=mock_class), \
         patch("retrieval.OpenAIEmbeddings", new=mock_class), \
         patch("agent.OpenAIEmbeddings", new=mock_class):
        yield mock_class

@pytest.fixture(autouse=True)
def mock_openai_chat():
    """Mock OpenAI Chat in all usage contexts."""
    mock_class = MagicMock()
    instance = mock_class.return_value
    
    # Default simple response
    instance.invoke.return_value.content = "Mocked response"
    
    # Mock stream
    chunk = MagicMock()
    chunk.content = "Mocked chunk"
    instance.stream.return_value = [chunk, chunk]
    
    # Patch in all modules where it is imported
    with patch("agent.ChatOpenAI", new=mock_class), \
         patch("retrieval.ChatOpenAI", new=mock_class), \
         patch("generation.ChatOpenAI", new=mock_class):
        yield mock_class

@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client."""
    with patch("storage.chromadb.PersistentClient") as mock:
        client_instance = mock.return_value
        collection_mock = MagicMock()
        client_instance.get_or_create_collection.return_value = collection_mock
        
        # Setup default behaviors for collection
        collection_mock.query.return_value = {
            "ids": [["id1"]],
            "documents": [["doc1"]],
            "metadatas": [[{"doc_id": "123"}]],
            "distances": [[0.1]]
        }
        collection_mock.get.return_value = {"ids": []}
        
        yield mock
