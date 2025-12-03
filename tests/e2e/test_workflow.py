import pytest
from unittest.mock import patch, MagicMock
from storage import VectorStore
from ingestion import ingest_pdf
from agent import RAGAgent
from hybrid_retrieval import create_retrieval_pipeline

def test_full_workflow(temp_persist_dir, mock_openai_embeddings, mock_openai_chat):
    """
    Simulate a full user session:
    1. Upload PDF
    2. Initialize Retriever/Agent
    3. Ask Question
    """
    # 1. Setup
    store = VectorStore(temp_persist_dir)
    
    # 2. Ingest
    with patch("ingestion.load_pdf") as mock_load:
        mock_load.return_value = (
            [
                {"content": "Python is a programming language." * 5, "page": 1},
                {"content": "It was created by Guido van Rossum." * 5, "page": 1}
            ], 300)

        ingest_pdf("python.pdf", "python.pdf", b"data", store)
    
    # 3. Initialize Agent
    retriever = create_retrieval_pipeline(store, enable_reranking=False)
    agent = RAGAgent(retriever)
    
    # 4. Ask Question (Mocking LLM to "read" the context and answer)
    # We need the mock_openai_chat to behave intelligently or just return a fixed string
    # verifying that it *received* the context.
    
    # The agent calls llm.invoke multiple times.
    # We want to ensure it calls with the retrieved context.
    
    # Mock sequence: 
    # 1. Thought: Search "Who created Python?" -> Action: search
    # 2. Thought: Answer -> Answer: Guido
    agent.llm.invoke.side_effect = [
        MagicMock(content='Thought: Search creator.\nAction: search("creator")'),
        MagicMock(content='Thought: Found it.\nAnswer: Guido van Rossum')
    ]
    
    result = agent.run("Who created Python?")
    
    assert result.answer == "Guido van Rossum"
    # Verify context was retrieved
    # The retriever uses the store, which uses the mock embeddings.
    # Since all embeddings are [0.1...], it retrieves everything.
    assert len(result.sources) > 0