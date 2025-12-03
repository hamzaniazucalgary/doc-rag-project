import pytest
from unittest.mock import patch, MagicMock
from generation import generate_response, stream_response

def test_generate_response(mock_openai_chat):
    mock_openai_chat.return_value.invoke.return_value.content = "Answer"
    
    response = generate_response("Q", "Context", "Concise")
    assert response == "Answer"
    mock_openai_chat.assert_called()

def test_stream_response(mock_openai_chat):
    # Mock streaming
    chunk = MagicMock()
    chunk.content = "chunk"
    mock_openai_chat.return_value.stream.return_value = [chunk, chunk]
    
    gen = stream_response("Q", "Context", "Concise")
    result = "".join(list(gen))
    
    assert result == "chunkchunk"
