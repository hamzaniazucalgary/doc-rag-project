"""Document-based question suggestions."""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config import LLM_MODEL, SUGGESTION_PROMPT


def generate_suggestions(document_text: str, max_chars: int = 3000) -> list[str]:
    """
    Generate 3 suggested questions based on document content.
    
    Args:
        document_text: full document text
        max_chars: max characters to send to LLM (default 3000)
    
    Returns:
        list of 3 question strings
    """
    excerpt = document_text[:max_chars]
    
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.7)
    prompt = SUGGESTION_PROMPT.format(excerpt=excerpt)
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Parse response into questions
    lines = response.content.strip().split("\n")
    questions = [line.strip() for line in lines if line.strip()]
    
    # Ensure exactly 3 questions
    questions = questions[:3]
    while len(questions) < 3:
        questions.append("What are the main topics covered in this document?")
    
    return questions


def get_combined_text(pages: list[dict]) -> str:
    """Combine page contents into single string."""
    return "\n\n".join(p["content"] for p in pages)
