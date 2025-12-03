import hashlib
from typing import Optional


def compute_file_hash(file_bytes: bytes) -> str:
    """Compute MD5 hash of file bytes for deduplication."""
    return hashlib.md5(file_bytes).hexdigest()


def is_valid_pdf(file) -> tuple[bool, Optional[str]]:
    """Validate that a file is a valid PDF."""
    if not file.name.lower().endswith('.pdf'):
        return False, "File must be a PDF"
    
    file.seek(0, 2)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)
    
    if size_mb > 50:
        return False, f"File too large ({size_mb:.1f}MB). Maximum is 50MB."
    
    header = file.read(4)
    file.seek(0)
    
    if header != b'%PDF':
        return False, "Invalid PDF file format"
    
    return True, None


def format_chat_history(messages: list[dict], max_turns: int = 5) -> str:
    """Format chat history for context in prompts."""
    recent = messages[-(max_turns * 2):]
    formatted = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content'][:500]}")
    return "\n".join(formatted)


def validate_api_key(api_key: str) -> tuple[bool, Optional[str]]:
    """Validate API key format (does not test connectivity)."""
    if not api_key:
        return False, "API key cannot be empty"
    
    api_key = api_key.strip()
    
    if not api_key.startswith("sk-"):
        return False, "API key must start with 'sk-'"
    
    if len(api_key) < 20:
        return False, "API key is too short"
    
    # Check for common copy-paste errors
    if " " in api_key:
        return False, "API key contains spaces"
    
    if api_key.endswith("..."):
        return False, "API key appears to be truncated"
    
    return True, None


def test_api_key(api_key: str) -> tuple[bool, Optional[str]]:
    """Actually test the API key by making a minimal API call."""
    import os
    
    # Temporarily set the key
    old_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = api_key
    
    try:
        from langchain_openai import OpenAIEmbeddings
        
        # Try to create an embedding - minimal API call
        embedder = OpenAIEmbeddings(model="text-embedding-3-small")
        _ = embedder.embed_query("test")
        
        return True, None
        
    except Exception as e:
        error_msg = str(e).lower()
        
        if "invalid api key" in error_msg or "incorrect api key" in error_msg:
            return False, "Invalid API key. Please check your key and try again."
        elif "rate limit" in error_msg:
            # Key is valid but rate limited - that's actually fine
            return True, None
        elif "quota" in error_msg:
            return False, "API key has exceeded its quota. Please check your billing."
        elif "authentication" in error_msg:
            return False, "Authentication failed. Please verify your API key."
        else:
            return False, f"API error: {str(e)[:100]}"
            
    finally:
        # Restore original key
        if old_key:
            os.environ["OPENAI_API_KEY"] = old_key
        elif "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]


def get_custom_css() -> str:
    """Return custom CSS for the Streamlit app."""
    return """
    <style>
        /* === FONTS === */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
        
        /* === GLOBAL === */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            -webkit-font-smoothing: antialiased;
        }
        
        /* Hide Streamlit branding but KEEP header for sidebar toggle */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Ensure sidebar toggle (hamburger) is always visible */
        [data-testid="collapsedControl"] {
            display: flex !important;
            visibility: visible !important;
        }
        
        /* Style the sidebar toggle button */
        button[kind="header"] {
            visibility: visible !important;
        }
        
        /* === TYPOGRAPHY === */
        h1 {
            font-weight: 700 !important;
            letter-spacing: -0.03em !important;
        }
        
        h2, h3, h4 {
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
        }
        
        /* === SIDEBAR === */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(0,0,0,0.02) 0%, rgba(0,0,0,0.05) 100%);
            border-right: 1px solid rgba(128, 128, 128, 0.1);
        }
        
        [data-testid="stSidebar"] h2 {
            color: inherit;
        }
        
        /* === BUTTONS === */
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
            font-size: 0.875rem;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease;
            border: 1px solid rgba(128, 128, 128, 0.2);
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            border: none;
            color: white !important;
        }
        
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #5558e3 0%, #7c4de8 100%);
        }
        
        /* === INPUTS === */
        .stTextInput > div > div > input {
            border-radius: 8px;
            border: 1px solid rgba(128, 128, 128, 0.2);
            padding: 0.75rem 1rem;
            font-size: 0.95rem;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }
        
        /* === CHAT === */
        .stChatMessage {
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 0.75rem;
        }
        
        [data-testid="stChatInput"] > div {
            border-radius: 12px;
            border: 1px solid rgba(128, 128, 128, 0.2);
        }
        
        [data-testid="stChatInput"] textarea {
            font-size: 0.95rem;
        }
        
        /* === FILE UPLOADER === */
        [data-testid="stFileUploader"] {
            border-radius: 12px;
        }
        
        [data-testid="stFileUploader"] > div {
            border-radius: 12px;
            border: 2px dashed rgba(128, 128, 128, 0.3);
            transition: all 0.2s ease;
        }
        
        [data-testid="stFileUploader"] > div:hover {
            border-color: #6366f1;
            background: rgba(99, 102, 241, 0.05);
        }
        
        /* === EXPANDERS === */
        .streamlit-expanderHeader {
            font-weight: 500;
            font-size: 0.9rem;
            border-radius: 8px;
            background-color: rgba(128, 128, 128, 0.05);
            border: 1px solid rgba(128, 128, 128, 0.1);
        }
        
        .streamlit-expanderContent {
            border: 1px solid rgba(128, 128, 128, 0.1);
            border-top: none;
            border-radius: 0 0 8px 8px;
            padding: 1rem;
            background: rgba(128, 128, 128, 0.02);
        }
        
        /* === ALERTS === */
        .stAlert {
            border-radius: 8px;
            border: none;
        }
        
        [data-testid="stNotification"] {
            border-radius: 8px;
        }
        
        /* === DIVIDERS === */
        hr {
            margin: 1rem 0;
            border: none;
            border-top: 1px solid rgba(128, 128, 128, 0.1);
        }
        
        /* === METRICS === */
        [data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.05);
            padding: 1rem;
            border-radius: 8px;
        }
        
        /* === PROGRESS BAR === */
        .stProgress > div > div {
            background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
            border-radius: 4px;
        }
        
        /* === TOGGLES === */
        [data-testid="stCheckbox"] label span {
            font-weight: 500;
        }
        
        /* === FEATURE BADGES === */
        .feature-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.35rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 0.5rem;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
            color: #6366f1;
            border: 1px solid rgba(99, 102, 241, 0.2);
        }
        
        /* === SCROLLBAR === */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(128, 128, 128, 0.3);
            border-radius: 3px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(128, 128, 128, 0.5);
        }
        
        /* === SELECTBOX === */
        [data-testid="stSelectbox"] > div > div {
            border-radius: 8px;
        }
        
        /* === TOOLTIPS === */
        [data-testid="stTooltipIcon"] {
            opacity: 0.5;
        }
        
        /* === LINK BUTTONS === */
        .stLinkButton > a {
            border-radius: 8px;
            font-weight: 500;
            border: 1px solid rgba(128, 128, 128, 0.2);
        }
        
        /* === SPINNER === */
        .stSpinner > div {
            border-top-color: #6366f1 !important;
        }
        
        /* === CAPTION === */
        .stCaption {
            opacity: 0.7;
        }
        
        /* === CODE BLOCKS === */
        code {
            background: rgba(128, 128, 128, 0.1);
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-size: 0.85em;
        }
        
        /* === EMPTY STATE === */
        .empty-state {
            text-align: center;
            padding: 3rem;
            border: 2px dashed rgba(128, 128, 128, 0.2);
            border-radius: 12px;
            margin: 2rem 0;
        }
    </style>
    """
