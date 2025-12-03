# === CHUNKING SETTINGS ===
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ". ", " "]

# === RETRIEVAL SETTINGS ===
TOP_K = 5
TOP_K_CANDIDATES = 20
SIMILARITY_THRESHOLD = 0.25
HYBRID_ALPHA = 0.5
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# === EMBEDDING SETTINGS ===
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 50

# === LLM SETTINGS ===
LLM_MODEL = "gpt-4o-mini"
MAX_TOKENS = 1024
TEMPERATURE = 0.1

# === AGENT SETTINGS ===
AGENT_MAX_ITERATIONS = 3
AGENT_MODEL = "gpt-4o-mini"

# === FILE LIMITS ===
MIN_EXTRACTED_CHARS = 100
MAX_FILE_SIZE_MB = 50
MAX_DOCS_PER_SESSION = 10

# === SYSTEM PROMPTS ===
SYSTEM_PROMPT = """You are a helpful document assistant. Answer questions based ONLY on the provided context from the uploaded documents.

IMPORTANT RULES:
1. If the context doesn't contain enough information to answer the question, clearly say: "I don't have enough information in the uploaded documents to answer this question."
2. Always cite your sources by mentioning the document name and page number when providing information.
3. Never make up or infer information that isn't explicitly stated in the context.
4. Be concise but thorough in your responses.

Response Style: {style_instruction}

Context from documents:
{context}
"""

STYLE_MAP = {
    "Concise": "Be brief and direct. Provide the key information in 2-3 sentences. No unnecessary elaboration.",
    "Detailed": "Be thorough and comprehensive. Include all relevant details, examples, and context from the documents. Use structured formatting if it helps clarity.",
    "ELI5": "Explain like I'm 5 years old. Use simple words, relatable analogies, and break down complex concepts into easy-to-understand pieces."
}

QUERY_REWRITE_PROMPT = """Given the conversation history and a follow-up question, rewrite the question to be a standalone question that includes all necessary context.

The rewritten question should be clear and self-contained, incorporating relevant context from the conversation history.

Conversation History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""

SUGGESTION_PROMPT = """Based on the following document excerpt, generate exactly 3 specific questions that a user might want to ask about this document.

Requirements:
- Questions must be answerable from the document content
- Questions should be specific and interesting, not generic
- Questions should cover different aspects or topics from the document
- Keep questions concise (under 80 characters if possible)

Format: Return ONLY the 3 questions, one per line, without numbering or bullet points.

Document Excerpt:
{excerpt}

Questions:"""

AGENT_SYSTEM_PROMPT = """You are an intelligent document analysis agent. Your task is to answer questions by strategically searching through uploaded documents.

You have ONE tool available:
- search("query"): Search the documents for relevant information. Returns text chunks with source citations.

WORKFLOW:
1. THINK: What specific information do I need to answer this question?
2. ACT: Use search("your specific query") to find relevant information
3. OBSERVE: Analyze the search results
4. ITERATE: If needed, search again with different terms (max {max_iterations} searches)
5. ANSWER: When you have sufficient information, provide your final answer

RESPONSE FORMAT (follow exactly):

For searching:
Thought: [your reasoning about what to search for]
Action: search("[your specific search query]")

For answering (when you have enough information):
Thought: [your analysis and how you'll construct the answer]
Answer: [your final answer with citations like (DocumentName, Page X)]

IMPORTANT GUIDELINES:
- Use specific, targeted search queries
- If initial search doesn't find what you need, try alternative keywords
- Always cite sources in your final answer
- If you cannot find the answer after {max_iterations} searches, acknowledge this honestly
- Synthesize information from multiple sources when relevant"""

# === EVALUATION PROMPTS ===
FAITHFULNESS_PROMPT = """You are evaluating whether an answer is factually supported by the given context.

Context:
{context}

Answer:
{answer}

Rate the faithfulness from 0.0 to 1.0:
- 0.0 = Answer contains claims not supported by context (hallucination)
- 0.5 = Answer is partially supported but includes some unsupported claims
- 1.0 = Answer is fully grounded in and supported by the context

Respond with ONLY a decimal number between 0.0 and 1.0."""

RELEVANCY_PROMPT = """You are evaluating whether an answer properly addresses the question asked.

Question:
{question}

Answer:
{answer}

Rate the relevancy from 0.0 to 1.0:
- 0.0 = Answer does not address the question at all
- 0.5 = Answer partially addresses the question but misses key aspects
- 1.0 = Answer directly and completely addresses the question

Respond with ONLY a decimal number between 0.0 and 1.0."""
