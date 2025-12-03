from typing import Generator
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import LLM_MODEL, MAX_TOKENS, TEMPERATURE, SYSTEM_PROMPT, STYLE_MAP


def build_messages(
    question: str,
    context: str,
    style: str = "Concise"
) -> list:
    """Build the message list for the LLM."""
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
    """Generate a response using the LLM."""
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
    """Stream a response from the LLM."""
    llm = ChatOpenAI(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        streaming=True
    )
    
    messages = build_messages(question, context, style)
    
    for chunk in llm.stream(messages):
        content = getattr(chunk, 'content', None)
        if content is not None and content != "":
            yield content
