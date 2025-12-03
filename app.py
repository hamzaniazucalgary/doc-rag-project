"""Streamlit application for Ask Your Docs - Enhanced with Agentic RAG."""
import os
import tempfile
import logging
import json

import streamlit as st
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

from config import MAX_DOCS_PER_SESSION, STYLE_MAP
from storage import VectorStore
from ingestion import ingest_pdf, load_pdf
from retrieval import retrieve_context, format_context, get_query_embedding
from generation import stream_response
from suggestions import generate_suggestions, get_combined_text
from utils import is_valid_pdf
from hybrid_retrieval import HybridRetriever, create_retrieval_pipeline
from agent import RAGAgent, AgentStep
from evaluation import RAGEvaluator, EvalCase


# === PAGE CONFIG ===
st.set_page_config(
    page_title="Ask Your Docs",
    page_icon="📄",
    layout="wide"
)


# === ERROR HANDLING ===
def safe_api_call(func, *args, **kwargs):
    """Wrapper for API calls with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_msg = str(e)
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            st.error("⏳ Rate limited. Please wait a moment and try again.")
        elif "api_key" in error_msg.lower():
            st.error("🔑 API key error. Check your OPENAI_API_KEY in .env")
        else:
            st.error(f"❌ Error: {error_msg}")
            logger.exception("API call failed")
        return None


def handle_api_error(e: Exception) -> None:
    """Handle API errors with user-friendly messages."""
    error_msg = str(e)
    if "rate_limit" in error_msg.lower() or "429" in error_msg:
        st.error("⏳ Rate limited. Please wait a moment and try again.")
    elif "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
        st.error("🔑 API key error. Check your OPENAI_API_KEY in .env")
    elif "model" in error_msg.lower():
        st.error(f"❌ Model error: {error_msg}")
    else:
        st.error(f"❌ Error: {error_msg}")
        logger.exception("API call failed")


# === SESSION STATE INITIALIZATION ===
def init_session_state():
    """Initialize all session state variables."""
    if "initialized" not in st.session_state:
        # Create unique persist dir for this session
        st.session_state.persist_dir = tempfile.mkdtemp(prefix="chroma_")
        st.session_state.store = VectorStore(st.session_state.persist_dir)
        st.session_state.documents = {}  # doc_id -> {name, pages, chunks}
        st.session_state.messages = []
        st.session_state.suggested_questions = []
        st.session_state.processed_files = set()  # Track processed file names
        st.session_state.hybrid_retriever = None  # Will be created after first doc
        st.session_state.agent = None  # Will be created after first doc
        st.session_state.initialized = True


init_session_state()


# === HELPER FUNCTIONS ===
def rebuild_retriever():
    """Rebuild hybrid retriever after document changes."""
    if st.session_state.documents:
        st.session_state.hybrid_retriever = create_retrieval_pipeline(
            st.session_state.store,
            enable_reranking=True
        )
        st.session_state.agent = RAGAgent(
            st.session_state.hybrid_retriever,
            verbose=False
        )
    else:
        st.session_state.hybrid_retriever = None
        st.session_state.agent = None


def create_pipeline_func():
    """Create pipeline function for evaluation."""
    def pipeline(question: str) -> dict:
        # Use hybrid retrieval
        if st.session_state.hybrid_retriever:
            query_embedding = get_query_embedding(question)
            results, is_low_conf = st.session_state.hybrid_retriever.retrieve(
                query=question,
                query_embedding=query_embedding
            )
            chunks = [r.to_dict() for r in results]
            context = "\n\n".join([c["content"] for c in chunks])
        else:
            # Fallback to basic retrieval
            retrieval_result = retrieve_context(
                query=question,
                store=st.session_state.store,
                chat_history=[]
            )
            chunks = retrieval_result["chunks"]
            context = format_context(chunks)
        
        # Generate answer
        from generation import generate_response
        answer = generate_response(question, context, "Concise")
        
        return {
            "answer": answer,
            "context": context,
            "chunks": chunks
        }
    
    return pipeline


# === SIDEBAR ===
with st.sidebar:
    st.header("📁 Document Management")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Max {MAX_DOCS_PER_SESSION} documents",
        key="pdf_uploader"
    )
    
    st.divider()
    
    # === MODE & STYLE SELECTION ===
    st.subheader("⚙️ Settings")
    
    # Agent mode toggle
    agent_mode = st.toggle(
        "🤖 Agent Mode",
        value=False,
        help="Enable multi-step reasoning with iterative retrieval. Better for complex questions."
    )
    
    # Hybrid search toggle
    use_hybrid = st.toggle(
        "🔀 Hybrid Search",
        value=True,
        help="Combine keyword (BM25) + semantic search. Better for exact matches."
    )
    
    # Reranking toggle
    use_reranking = st.toggle(
        "📊 Cross-Encoder Reranking",
        value=True,
        help="Use cross-encoder to rerank results. More accurate but slower."
    )
    
    # Style selector
    style = st.selectbox(
        "Response Style",
        options=list(STYLE_MAP.keys()),
        index=0,
        help="How should responses be formatted?"
    )
    
    st.divider()
    
    # === DOCUMENT LIST ===
    if st.session_state.documents:
        st.subheader("📚 Loaded Documents")
        for doc_id, doc_info in st.session_state.documents.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"📄 {doc_info['name'][:20]}...")
            with col2:
                if st.button("🗑️", key=f"del_{doc_id}"):
                    st.session_state.store.delete_doc(doc_id)
                    del st.session_state.documents[doc_id]
                    st.session_state.processed_files.discard(doc_info['name'])
                    rebuild_retriever()
                    st.rerun()
    
    st.divider()
    
    # === EVALUATION SECTION ===
    st.subheader("🧪 Evaluation")
    
    eval_file = st.file_uploader(
        "Upload test cases (JSON)",
        type=["json"],
        help="JSON array with input/question, expected_output/expected_answer fields",
        key="eval_uploader"
    )
    
    if eval_file and st.session_state.documents:
        if st.button("▶️ Run Evaluation", use_container_width=True):
            try:
                # Load test cases
                test_data = json.load(eval_file)
                cases = [EvalCase.from_dict(c) for c in test_data]
                
                evaluator = RAGEvaluator(cases, verbose=False)
                pipeline_func = create_pipeline_func()
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total):
                    progress_bar.progress(current / total)
                    status_text.text(f"Evaluating {current}/{total}...")
                
                # Run evaluation
                report = evaluator.run_all(pipeline_func, progress_callback=update_progress)
                
                # Clear progress
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                st.success("Evaluation complete!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Faithfulness", f"{report.avg_faithfulness:.2f}")
                col2.metric("Relevancy", f"{report.avg_relevancy:.2f}")
                col3.metric("Retrieval Acc", f"{report.retrieval_accuracy:.0%}")
                
                # Detailed results expander
                with st.expander("📋 Detailed Results"):
                    for r in report.results:
                        st.markdown(f"**Q:** {r.question[:80]}...")
                        st.markdown(f"Faith: {r.faithfulness_score:.2f} | Rel: {r.relevancy_score:.2f} | Hit: {'✅' if r.retrieval_hit else '❌'}")
                        if r.error:
                            st.error(f"Error: {r.error}")
                        st.divider()
                
            except Exception as e:
                st.error(f"Evaluation error: {e}")
    
    st.divider()
    
    # Clear chat button
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.suggested_questions = []
        st.rerun()


# === FILE PROCESSING ===
if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    
    if new_files:
        total_docs = len(st.session_state.documents) + len(new_files)
        if total_docs > MAX_DOCS_PER_SESSION:
            st.error(f"Maximum {MAX_DOCS_PER_SESSION} documents allowed")
        else:
            for uploaded_file in new_files:
                file_bytes = uploaded_file.read()
                uploaded_file.seek(0)
                
                is_valid, error = is_valid_pdf(uploaded_file)
                if not is_valid:
                    st.error(f"{uploaded_file.name}: {error}")
                    st.session_state.processed_files.add(uploaded_file.name)
                    continue
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name
                
                try:
                    progress_container = st.empty()
                    progress_bar = st.progress(0)
                    
                    def update_progress(stage, progress):
                        progress_container.text(f"{uploaded_file.name}: {stage}")
                        progress_bar.progress(progress)
                    
                    result = safe_api_call(
                        ingest_pdf,
                        file_path=tmp_path,
                        file_name=uploaded_file.name,
                        file_bytes=file_bytes,
                        store=st.session_state.store,
                        progress_callback=update_progress
                    )
                    
                    if result and result["status"] == "ingested":
                        st.session_state.documents[result["doc_id"]] = {
                            "name": result["doc_name"],
                            "pages": result["pages"],
                            "chunks": result["chunks"]
                        }
                        st.success(f"✅ {uploaded_file.name}: {result['pages']} pages, {result['chunks']} chunks")
                        
                        # Rebuild retriever with new documents
                        rebuild_retriever()
                        
                        # Generate suggestions if first doc
                        if len(st.session_state.documents) == 1 and not st.session_state.messages:
                            pages, _ = load_pdf(tmp_path)
                            combined_text = get_combined_text(pages)
                            suggestions = safe_api_call(generate_suggestions, combined_text)
                            if suggestions:
                                st.session_state.suggested_questions = suggestions
                    
                    elif result and result["status"] == "skipped":
                        st.info(f"ℹ️ {uploaded_file.name}: Already loaded")
                    elif result and result["status"] == "error":
                        st.error(f"❌ {uploaded_file.name}: {result.get('error', 'Unknown error')}")
                    
                    st.session_state.processed_files.add(uploaded_file.name)
                    progress_container.empty()
                    progress_bar.empty()
                    
                finally:
                    os.unlink(tmp_path)


# === MAIN CHAT AREA ===
st.header("💬 Ask Your Docs")

# Mode indicator
if agent_mode:
    st.caption("🤖 Agent Mode: Multi-step reasoning enabled")
elif use_hybrid:
    st.caption("🔀 Hybrid Search: BM25 + Semantic")
else:
    st.caption("🔍 Standard Mode: Semantic search only")


def process_question_standard(question: str) -> None:
    """Process question using standard RAG pipeline."""
    with st.chat_message("assistant"):
        # Retrieve context
        if use_hybrid and st.session_state.hybrid_retriever:
            query_embedding = get_query_embedding(question)
            results, is_low_confidence = st.session_state.hybrid_retriever.retrieve(
                query=question,
                query_embedding=query_embedding,
                use_reranking=use_reranking
            )
            chunks = [r.to_dict() for r in results]
            context = format_context(chunks)
        else:
            retrieval_result = safe_api_call(
                retrieve_context,
                query=question,
                store=st.session_state.store,
                chat_history=st.session_state.messages[:-1]
            )
            
            if not retrieval_result:
                st.error("Failed to retrieve context. Please try again.")
                st.session_state.messages.pop()
                return
            
            chunks = retrieval_result["chunks"]
            context = format_context(chunks)
            is_low_confidence = retrieval_result["is_low_confidence"]
        
        if is_low_confidence:
            st.warning("⚠️ Low confidence match - answer may be less reliable")
        
        # Stream response
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            stream_gen = stream_response(question, context, style)
            for token in stream_gen:
                full_response += token
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        except Exception as e:
            handle_api_error(e)
            st.session_state.messages.pop()
            return
        
        if full_response:
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": chunks
            })
            
            with st.expander("📚 View Sources"):
                for source in chunks:
                    st.markdown(f"**{source['doc_name']}** - Page {source['page']}")
                    content = source["content"]
                    st.text(content[:300] + "..." if len(content) > 300 else content)
                    st.divider()


def process_question_agent(question: str) -> None:
    """Process question using agentic RAG."""
    with st.chat_message("assistant"):
        if not st.session_state.agent:
            st.error("Agent not initialized. Please upload documents first.")
            st.session_state.messages.pop()
            return
        
        # Create containers for reasoning trace and answer
        reasoning_container = st.container()
        answer_placeholder = st.empty()
        
        try:
            # Run agent
            with reasoning_container:
                with st.expander("🧠 Agent Reasoning", expanded=True):
                    step_placeholder = st.empty()
                    steps_display = []
                    
                    def display_step(step: AgentStep):
                        steps_display.append(step)
                        step_text = ""
                        for s in steps_display:
                            step_text += f"**Step {s.step_num}**\n"
                            step_text += f"💭 *Thought:* {s.thought}\n\n"
                            if s.action == "search":
                                step_text += f"🔍 *Search:* `{s.action_input}`\n\n"
                                step_text += f"📄 *Found:* {s.observation[:200]}...\n\n"
                            step_text += "---\n"
                        step_placeholder.markdown(step_text)
                    
                    # Execute agent
                    result = st.session_state.agent.run(question)
                    
                    # Display all steps
                    for step in result.steps:
                        display_step(step)
            
            # Display answer
            answer_placeholder.markdown(result.answer)
            
            # Store message
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.answer,
                "sources": result.sources,
                "agent_steps": [s.to_dict() for s in result.steps]
            })
            
            # Show sources
            if result.sources:
                with st.expander("📚 View Sources"):
                    seen = set()
                    for source in result.sources:
                        key = (source.get("doc_name"), source.get("page"))
                        if key not in seen:
                            seen.add(key)
                            st.markdown(f"**{source['doc_name']}** - Page {source['page']}")
                            content = source["content"]
                            st.text(content[:300] + "..." if len(content) > 300 else content)
                            st.divider()
            
            # Show stats
            st.caption(f"📊 {result.total_retrievals} searches performed")
            
        except Exception as e:
            handle_api_error(e)
            st.session_state.messages.pop()
            return


def process_question(question: str) -> None:
    """Route to appropriate processor based on mode."""
    if agent_mode:
        process_question_agent(question)
    else:
        process_question_standard(question)


# Show suggestions if no messages
if not st.session_state.messages and st.session_state.suggested_questions:
    st.write("**Suggested questions:**")
    cols = st.columns(3)
    for i, question in enumerate(st.session_state.suggested_questions):
        with cols[i]:
            if st.button(question, key=f"suggest_{i}", use_container_width=True):
                st.session_state.suggested_questions = []
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.write(question)
                process_question(question)
                st.rerun()


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        if message["role"] == "assistant":
            # Show agent reasoning if available
            if "agent_steps" in message and message["agent_steps"]:
                with st.expander("🧠 View Reasoning"):
                    for step in message["agent_steps"]:
                        st.markdown(f"**Step {step['step']}:** {step['thought'][:100]}...")
                        if step["action"] == "search":
                            st.caption(f"🔍 Searched: {step['action_input']}")
            
            # Show sources
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    seen = set()
                    for source in message["sources"]:
                        doc_name = source.get("doc_name", "Unknown")
                        page = source.get("page", 0)
                        key = (doc_name, page)
                        if key not in seen:
                            seen.add(key)
                            st.markdown(f"**{doc_name}** - Page {page}")
                            content = source.get("content", "")
                            st.text(content[:300] + "..." if len(content) > 300 else content)
                            st.divider()


# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    if not st.session_state.documents:
        st.warning("Please upload at least one PDF first")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        process_question(prompt)
        st.rerun()


# === FOOTER ===
if not st.session_state.documents:
    st.info("👆 Upload PDF documents using the sidebar to get started")
else:
    # Show current capabilities
    capabilities = []
    if agent_mode:
        capabilities.append("🤖 Agentic RAG")
    if use_hybrid:
        capabilities.append("🔀 Hybrid Search")
    if use_reranking:
        capabilities.append("📊 Reranking")
    
    if capabilities:
        st.caption(f"Active: {' | '.join(capabilities)}")
