import pytest
from io import BytesIO
from utils import compute_file_hash, is_valid_pdf, format_chat_history

def test_compute_file_hash():
    content = b"test content"
    # echo -n "test content" | md5sum -> 94739dd040970550613237cb2d6ddf4
    # Wait, md5 of "test content" is 1e269774c404a953326252c9a047db70 (oops, calculated manually)
    # Let's just check consistency
    hash1 = compute_file_hash(content)
    hash2 = compute_file_hash(content)
    assert hash1 == hash2
    assert len(hash1) == 32

def test_is_valid_pdf_valid():
    # Create a dummy valid PDF (header only)
    content = b"%PDF-1.4\n..."
    f = BytesIO(content)
    f.name = "test.pdf"
    
    is_valid, error = is_valid_pdf(f)
    assert is_valid is True
    assert error is None

def test_is_valid_pdf_invalid_extension():
    f = BytesIO(b"content")
    f.name = "test.txt"
    is_valid, error = is_valid_pdf(f)
    assert is_valid is False
    assert "must be a PDF" in error

def test_is_valid_pdf_invalid_header():
    content = b"Not a PDF"
    f = BytesIO(content)
    f.name = "test.pdf"
    is_valid, error = is_valid_pdf(f)
    assert is_valid is False
    assert "Invalid PDF" in error

def test_is_valid_pdf_too_large():
    # Mock size check by patching file.tell or creating large buffer (expensive)
    # Instead, we can rely on the logic. 
    # Let's create a mock file object to control size reporting
    class MockFile:
        name = "large.pdf"
        def seek(self, *args): pass
        def tell(self): return 51 * 1024 * 1024 # 51 MB
        def read(self, *args): return b"%PDF" # Valid header
        
    f = MockFile()
    is_valid, error = is_valid_pdf(f)
    assert is_valid is False
    assert "too large" in error

def test_format_chat_history():
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Question"}
    ]
    formatted = format_chat_history(messages)
    expected = "User: Hi\nAssistant: Hello\nUser: Question"
    assert formatted == expected
