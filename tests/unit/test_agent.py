import pytest
from unittest.mock import MagicMock, patch
from agent import RAGAgent, AgentResult, AgentStep

@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    # Setup a dummy result object
    class Result:
        content = "Retrieved content"
        doc_name = "doc1"
        page = 1
        def to_dict(self):
            return {"content": self.content, "doc_name": self.doc_name, "page": self.page}
            
    retriever.retrieve.return_value = ([Result()], False)
    return retriever

def test_agent_init(mock_retriever):
    with patch("agent.ChatOpenAI"), patch("agent.OpenAIEmbeddings"):
        agent = RAGAgent(mock_retriever)
        assert agent.retriever == mock_retriever

def test_agent_run_answer_immediately(mock_retriever):
    with patch("agent.ChatOpenAI") as mock_chat, patch("agent.OpenAIEmbeddings"):
        # Mock LLM to return immediate answer
        mock_chat.return_value.invoke.return_value.content = "Thought: Simple question.\nAnswer: The answer is 42."
        
        agent = RAGAgent(mock_retriever)
        result = agent.run("What is 6*7?")
        
        assert result.answer == "The answer is 42."
        assert len(result.steps) == 1
        assert result.steps[0].action == "answer"

def test_agent_run_search_then_answer(mock_retriever):
    with patch("agent.ChatOpenAI") as mock_chat, patch("agent.OpenAIEmbeddings"):
        # Mock LLM responses:
        # 1. Search action
        # 2. Answer action
        mock_chat.return_value.invoke.side_effect = [
            MagicMock(content='Thought: I need to search.\nAction: search("query")'),
            MagicMock(content='Thought: I have info.\nAnswer: The answer is found.')
        ]
        
        agent = RAGAgent(mock_retriever)
        result = agent.run("Complex question")
        
        # Verify retrieval was called
        mock_retriever.retrieve.assert_called()
        
        assert result.answer == "The answer is found."
        assert len(result.steps) == 2
        assert result.steps[0].action == "search"
        assert result.steps[1].action == "answer"

def test_agent_run_max_iterations(mock_retriever):
    with patch("agent.ChatOpenAI") as mock_chat, patch("agent.OpenAIEmbeddings"):
        # Always search
        mock_chat.return_value.invoke.return_value.content = 'Thought: Searching...\nAction: search("query")'
        
        agent = RAGAgent(mock_retriever, max_iterations=2)
        result = agent.run("Hard question")
        
        # Should have stopped and forced answer (which uses LLM one last time)
        # The last call is _force_answer
        assert "Unable" in result.answer or "Search" in result.answer or len(result.steps) == 2

def test_parse_response():
    agent = RAGAgent(MagicMock()) # No need to mock internal libs if we just test static method-ish logic
    
    # Test Search
    resp1 = 'Thought: Thinking.\nAction: search("foo")'
    parsed1 = agent._parse_response(resp1)
    assert parsed1["type"] == "search"
    assert parsed1["query"] == "foo"
    
    # Test Answer
    resp2 = 'Thought: Done.\nAnswer: Bar'
    parsed2 = agent._parse_response(resp2)
    assert parsed2["type"] == "answer"
    assert parsed2["answer"] == "Bar"
