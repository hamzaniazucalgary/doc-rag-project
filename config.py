"""
Configuration constants for Ask Your Docs.
All magic numbers and prompts live here.
"""

# === CHUNKING ===
CHUNK_SIZE = 1500          # characters (not tokens)
CHUNK_OVERLAP = 200        # characters
SEPARATORS = ["\n\n", "\n", ". ", " "]

# === RETRIEVAL ===
TOP_K = 5                  # candidates to retrieve
TOP_K_CANDIDATES = 20      # candidates before reranking
SIMILARITY_THRESHOLD = 0.25  # ChromaDB distance (1 - cosine_sim)
                             # 0.25 distance ≈ 0.75 similarity

# === HYBRID SEARCH ===
HYBRID_ALPHA = 0.5         # Weight for semantic vs BM25 (1.0 = pure semantic)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# === EMBEDDING ===
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 50

# === LLM ===
LLM_MODEL = "gpt-4o-mini"
MAX_TOKENS = 1024
TEMPERATURE = 0.1

# === AGENT ===
AGENT_MAX_ITERATIONS = 3
AGENT_MODEL = "gpt-4o-mini"

# === VALIDATION ===
MIN_EXTRACTED_CHARS = 100  # below this = likely scanned/empty PDF
MAX_FILE_SIZE_MB = 50
MAX_DOCS_PER_SESSION = 10

# === PROMPTS ===
SYSTEM_PROMPT = """You are a document assistant. Answer based ONLY on the provided context.

CRITICAL RULES:
1. If the context doesn't contain the answer, say: "I don't have information about that in the uploaded documents."
2. Always cite your sources by mentioning the document name and page number.
3. Never make up information not present in the context.

Style: {style_instruction}

Context:
{context}
"""

STYLE_MAP = {
    "Concise": "Be brief. Max 2-3 sentences. No unnecessary elaboration.",
    "ELI5": "Explain like I'm 5 years old. Use simple words, analogies, and examples a child would understand.",
    "Detailed": "Be thorough. Include all relevant details from the context. Use structured formatting if helpful."
}

QUERY_REWRITE_PROMPT = """Given the conversation history and a follow-up question, rewrite the question to be standalone.

The rewritten question must include all necessary context from the history to be understood independently.

Conversation History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""

SUGGESTION_PROMPT = """Based on the following document excerpt, generate exactly 3 specific questions a user might ask about this document.

Requirements:
- Questions should be answerable from the document
- Questions should be specific, not generic
- Questions should cover different aspects/topics
- Format: Return ONLY the 3 questions, one per line, no numbering

Document Excerpt:
{excerpt}

Questions:"""

# === AGENT PROMPTS ===
AGENT_SYSTEM_PROMPT = """You are a document analysis agent. Your job is to answer questions by searching through uploaded documents.

You have access to ONE tool:
- search("query"): Search documents for relevant information. Returns relevant text chunks with source info.

PROCESS:
1. Think: What specific information do I need to answer this question?
2. Act: Use search("specific search query") to find relevant information
3. Observe: Review what was found
4. Repeat if needed (max {max_iterations} searches)
5. Answer: When you have enough information, provide your final answer

FORMAT YOUR RESPONSE EXACTLY AS:

Thought: [your reasoning about what to search for]
Action: search("[your specific search query]")

After receiving search results, continue with:

Thought: [your analysis of the results - do you have enough info?]
Action: search("[another query if needed]")

OR if you have enough information:

Thought: [your reasoning about how to answer]
Answer: [your final answer citing sources as (DocumentName, Page X)]

RULES:
- Be specific in your search queries
- If first search doesn't find relevant info, try different keywords
- Always cite sources in your final answer
- If you cannot find the answer after {max_iterations} searches, say so honestly"""

# === EVALUATION PROMPTS ===
FAITHFULNESS_PROMPT = """You are evaluating if an answer is supported by the given context.

Context:
{context}

Answer:
{answer}

Rate from 0.0 to 1.0:
- 0.0 = Answer contains claims not supported by context (hallucination)
- 0.5 = Answer is partially supported but includes unsupported claims
- 1.0 = Answer is fully grounded in the context

Respond with ONLY a decimal number between 0.0 and 1.0."""

RELEVANCY_PROMPT = """You are evaluating if an answer addresses the question asked.

Question:
{question}

Answer:
{answer}

Rate from 0.0 to 1.0:
- 0.0 = Answer does not address the question at all
- 0.5 = Answer partially addresses the question
- 1.0 = Answer directly and completely addresses the question

Respond with ONLY a decimal number between 0.0 and 1.0."""
