"""Utility functions for file handling and validation."""
import hashlib
from typing import Optional


def compute_file_hash(file_bytes: bytes) -> str:
    """Compute MD5 hash of file content for deduplication."""
    return hashlib.md5(file_bytes).hexdigest()


def is_valid_pdf(file) -> tuple[bool, Optional[str]]:
    """
    Validate PDF file.
    
    Returns: (is_valid, error_message)
    """
    # Check file extension
    if not file.name.lower().endswith('.pdf'):
        return False, "File must be a PDF"
    
    # Check file size
    file.seek(0, 2)  # Seek to end
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)  # Reset position
    
    if size_mb > 50:
        return False, f"File too large ({size_mb:.1f}MB). Maximum is 50MB."
    
    # Check PDF magic bytes
    header = file.read(4)
    file.seek(0)
    
    if header != b'%PDF':
        return False, "Invalid PDF file format"
    
    return True, None


def format_chat_history(messages: list[dict], max_turns: int = 5) -> str:
    """Format recent chat history for query rewriting."""
    recent = messages[-(max_turns * 2):]  # Last N turns (user + assistant pairs)
    
    formatted = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content'][:500]}")  # Truncate long messages
    
    return "\n".join(formatted)
