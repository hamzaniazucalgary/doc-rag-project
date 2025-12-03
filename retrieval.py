from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage

from config import (
    EMBEDDING_MODEL, LLM_MODEL, TEMPERATURE,
    QUERY_REWRITE_PROMPT, TOP_K
)
from storage import VectorStore
from utils import format_chat_history


def rewrite_query(question: str, chat_history: list[dict]) -> str:
    """Rewrite follow-up questions to be standalone."""
    if not chat_history:
        return question
    
    standalone_indicators = ["what is", "who is", "how does", "explain", "describe"]
    if any(question.lower().startswith(ind) for ind in standalone_indicators):
        pass
    
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    formatted_history = format_chat_history(chat_history)
    
    prompt = QUERY_REWRITE_PROMPT.format(
        chat_history=formatted_history,
        question=question
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def get_query_embedding(query: str) -> list[float]:
    """Generate embedding for a query."""
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return embedder.embed_query(query)


def retrieve_context(
    query: str,
    store: VectorStore,
    chat_history: list[dict] = None
) -> dict:
    """Retrieve relevant context for a query."""
    rewritten = rewrite_query(query, chat_history or [])
    query_embedding = get_query_embedding(rewritten)
    
    results = store.query(query_embedding, n_results=TOP_K)
    filtered, is_low_confidence = store.filter_by_threshold(results)
    
    chunks = []
    for i, doc in enumerate(filtered["documents"]):
        meta = filtered["metadatas"][i] if filtered["metadatas"] else {}
        chunks.append({
            "content": doc,
            "doc_name": meta.get("doc_name", "Unknown"),
            "doc_id": meta.get("doc_id", ""),
            "page": meta.get("page", 0),
            "chunk_index": meta.get("chunk_index", 0),
            "distance": filtered["distances"][i] if filtered["distances"] else 1.0
        })
    
    return {
        "original_query": query,
        "rewritten_query": rewritten,
        "chunks": chunks,
        "is_low_confidence": is_low_confidence
    }


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string."""
    if not chunks:
        return "No relevant context found."
    
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        doc_name = chunk.get("doc_name", "Unknown")
        page = chunk.get("page", 0)
        content = chunk.get("content", "")
        
        formatted.append(
            f"[Source {i}: {doc_name}, Page {page}]\n"
            f"{content}"
        )
    
    return "\n\n---\n\n".join(formatted)
