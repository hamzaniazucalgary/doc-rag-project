"""LLM response generation with streaming."""
from typing import Generator

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import LLM_MODEL, MAX_TOKENS, TEMPERATURE, SYSTEM_PROMPT, STYLE_MAP


def build_messages(
    question: str,
    context: str,
    style: str = "Concise"
) -> list:
    """Build message list for LLM."""
    style_instruction = STYLE_MAP.get(style, STYLE_MAP["Concise"])
    
    system_content = SYSTEM_PROMPT.format(
        style_instruction=style_instruction,
        context=context
    )
    
    return [
        SystemMessage(content=system_content),
        HumanMessage(content=question)
    ]


def generate_response(
    question: str,
    context: str,
    style: str = "Concise"
) -> str:
    """Generate complete response (non-streaming)."""
    llm = ChatOpenAI(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE
    )
    
    messages = build_messages(question, context, style)
    response = llm.invoke(messages)
    
    return response.content


def stream_response(
    question: str,
    context: str,
    style: str = "Concise"
) -> Generator[str, None, None]:
    """
    Generate streaming response.
    
    Yields: tokens as they're generated
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        streaming=True
    )
    
    messages = build_messages(question, context, style)
    
    for chunk in llm.stream(messages):
        # Handle both string content and None
        # Use getattr for safety in case chunk structure varies
        content = getattr(chunk, 'content', None)
        if content is not None and content != "":
            yield content
