import os
import tempfile
import logging
import json
import streamlit as st
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

from config import MAX_DOCS_PER_SESSION, STYLE_MAP
from storage import VectorStore
from ingestion import ingest_pdf, load_pdf
from retrieval import retrieve_context, format_context, get_query_embedding
from generation import stream_response
from suggestions import generate_suggestions, get_combined_text
from utils import is_valid_pdf, get_custom_css, validate_api_key, test_api_key
from hybrid_retrieval import HybridRetriever, create_retrieval_pipeline
from agent import RAGAgent, AgentStep
from evaluation import RAGEvaluator, EvalCase

# Constants
GITHUB_URL = "https://github.com/hamzaniazucalgary/doc-rag-project"
OPENAI_API_URL = "https://platform.openai.com/api-keys"

# Page config
st.set_page_config(
    page_title="Ask Your Docs",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)


def check_api_key() -> bool:
    """Check if a valid API key is configured."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return False
    is_valid, _ = validate_api_key(api_key)
    return is_valid


def show_api_key_error():
    """Show error dialog when API key is missing."""
    st.error(
        "⚠️ **OpenAI API Key Required**\n\n"
        "Please enter your API key in the sidebar to use this feature.\n\n"
        f"[🔗 Get an API key here]({OPENAI_API_URL})"
    )


def handle_api_error(e: Exception) -> None:
    """Handle API errors with user-friendly messages."""
    error_msg = str(e).lower()
    if "rate_limit" in error_msg or "429" in error_msg:
        st.error("⏳ Rate limited. Please wait a moment and try again.")
    elif "api_key" in error_msg or "authentication" in error_msg or "invalid" in error_msg:
        st.error(
            "🔑 **Invalid API Key**\n\n"
            f"Please check your API key in the sidebar. [Get a new key]({OPENAI_API_URL})"
        )
    elif "model" in error_msg:
        st.error(f"❌ Model error: {str(e)}")
    else:
        st.error(f"❌ Error: {str(e)}")
        logger.exception("API call failed")


def safe_api_call(func, *args, **kwargs):
    """Wrapper for API calls with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handle_api_error(e)
        return None


def init_session_state():
    """Initialize all session state variables."""
    if "initialized" not in st.session_state:
        st.session_state.persist_dir = tempfile.mkdtemp(prefix="chroma_")
        st.session_state.store = VectorStore(st.session_state.persist_dir)
        st.session_state.documents = {}
        st.session_state.messages = []
        st.session_state.suggested_questions = []
        st.session_state.processed_files = set()
        st.session_state.hybrid_retriever = None
        st.session_state.agent = None
        st.session_state.initialized = True


def rebuild_retriever():
    """Rebuild the retrieval pipeline after document changes."""
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
    """Create evaluation pipeline function."""
    def pipeline(question: str) -> dict:
        if st.session_state.hybrid_retriever:
            query_embedding = get_query_embedding(question)
            results, is_low_conf = st.session_state.hybrid_retriever.retrieve(
                query=question,
                query_embedding=query_embedding
            )
            chunks = [r.to_dict() for r in results]
            context = "\n\n".join([c["content"] for c in chunks])
        else:
            retrieval_result = retrieve_context(
                query=question,
                store=st.session_state.store,
                chat_history=[]
            )
            chunks = retrieval_result["chunks"]
            context = format_context(chunks)

        from generation import generate_response
        answer = generate_response(question, context, "Concise")
        return {
            "answer": answer,
            "context": context,
            "chunks": chunks
        }
    return pipeline


def render_sidebar():
    """Render the sidebar with all controls."""
    with st.sidebar:
        # Header with GitHub link

        
        # API Key Section
        st.markdown("### 🔑 API Key")
        
        current_key = os.environ.get("OPENAI_API_KEY", "")
        has_key = check_api_key()
        
        if has_key:
            st.success("✓ API Key configured", icon="✅")
            masked_key = current_key[:7] + "..." + current_key[-4:] if len(current_key) > 15 else "***"
            st.caption(f"Key: `{masked_key}`")
            if st.button("Change Key", use_container_width=True):
                os.environ["OPENAI_API_KEY"] = ""
                st.rerun()
        else:
            api_key_input = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Required to process documents",
                label_visibility="collapsed"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Key", type="primary", use_container_width=True):
                    if api_key_input:
                        is_valid, error = validate_api_key(api_key_input)
                        if is_valid:
                            os.environ["OPENAI_API_KEY"] = api_key_input
                            st.success("✓ Key saved!")
                            st.rerun()
                        else:
                            st.error(error)
                    else:
                        st.warning("Enter a key first")
            with col2:
                st.link_button("Get Key", OPENAI_API_URL, use_container_width=True)
        
        st.divider()
        
        # Document Upload Section
        st.markdown("### 📁 Documents")
        
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Maximum {MAX_DOCS_PER_SESSION} documents, 50MB each",
            label_visibility="collapsed"
        )
        
        # Display uploaded documents
        if st.session_state.documents:
            st.markdown("**Uploaded Files:**")
            for doc_id, doc_info in list(st.session_state.documents.items()):
                col1, col2 = st.columns([4, 1])
                with col1:
                    display_name = doc_info['name'][:22] + "..." if len(doc_info['name']) > 22 else doc_info['name']
                    st.caption(f"📄 {display_name}")
                with col2:
                    if st.button("✕", key=f"del_{doc_id}", help=f"Remove {doc_info['name']}"):
                        st.session_state.store.delete_doc(doc_id)
                        del st.session_state.documents[doc_id]
                        st.session_state.processed_files.discard(doc_info['name'])
                        rebuild_retriever()
                        st.rerun()
        else:
            st.info("No documents uploaded", icon="📭")
        
        st.divider()
        
        # Settings Section
        st.markdown("### ⚙️ Settings")
        
        agent_mode = st.toggle(
            "🤖 Agent Mode",
            value=False,
            help="Enable multi-step reasoning for complex questions"
        )
        
        use_hybrid = st.toggle(
            "🔀 Hybrid Search",
            value=True,
            help="Combine keyword and semantic search"
        )
        
        use_reranking = st.toggle(
            "📊 Reranking",
            value=True,
            help="Use cross-encoder to improve result quality"
        )
        
        style = st.selectbox(
            "Response Style",
            options=list(STYLE_MAP.keys()),
            index=0,
            help="How detailed should responses be?"
        )
        
        st.divider()
        
        # Evaluation Section
        with st.expander("🧪 Evaluation", expanded=False):
            eval_file = st.file_uploader(
                "Upload test cases (JSON)",
                type=["json"],
                help="Upload a JSON file with test cases"
            )
            
            if eval_file and st.session_state.documents:
                if st.button("▶ Run Evaluation", type="primary", use_container_width=True):
                    if not check_api_key():
                        show_api_key_error()
                    else:
                        try:
                            test_data = json.load(eval_file)
                            cases = [EvalCase.from_dict(c) for c in test_data]
                            evaluator = RAGEvaluator(cases, verbose=False)
                            pipeline_func = create_pipeline_func()
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            def update_progress(current, total):
                                progress_bar.progress(current / total)
                                status_text.text(f"Testing {current}/{total}...")
                            
                            report = evaluator.run_all(pipeline_func, progress_callback=update_progress)
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.success("Evaluation Complete!")
                            st.metric("Faithfulness", f"{report.avg_faithfulness:.2f}")
                            st.metric("Relevancy", f"{report.avg_relevancy:.2f}")
                            st.metric("Accuracy", f"{report.retrieval_accuracy:.1%}")
                            
                        except json.JSONDecodeError:
                            st.error("Invalid JSON file")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            elif eval_file and not st.session_state.documents:
                st.warning("Upload documents first")
        
        st.divider()
        
        # Actions
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.suggested_questions = []
            st.rerun()
        
    
    return agent_mode, use_hybrid, use_reranking, style, uploaded_files


def process_uploads(uploaded_files):
    """Process uploaded PDF files."""
    if not uploaded_files:
        return
    
    # Check API key before processing
    if not check_api_key():
        show_api_key_error()
        return
    
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    
    if not new_files:
        return
    
    total_docs = len(st.session_state.documents) + len(new_files)
    if total_docs > MAX_DOCS_PER_SESSION:
        st.error(f"Maximum {MAX_DOCS_PER_SESSION} documents allowed")
        return
    
    for uploaded_file in new_files:
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        
        is_valid, error = is_valid_pdf(uploaded_file)
        if not is_valid:
            st.error(f"❌ {uploaded_file.name}: {error}")
            st.session_state.processed_files.add(uploaded_file.name)
            continue
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        try:
            progress_container = st.empty()
            progress_bar = st.progress(0)
            
            def update_progress(stage, progress):
                progress_container.caption(f"📄 {uploaded_file.name}: {stage}")
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
                st.success(f"✓ {uploaded_file.name}: {result['pages']} pages indexed")
                rebuild_retriever()
                
                # Generate suggestions for first document
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


def process_question_standard(question: str, use_hybrid: bool, use_reranking: bool, style: str):
    """Process question using standard RAG pipeline."""
    with st.chat_message("assistant", avatar="🤖"):
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
            
            # Show sources
            if chunks:
                with st.expander("📚 View Sources", expanded=False):
                    for i, source in enumerate(chunks):
                        st.markdown(f"**{source['doc_name']}** — Page {source['page']}")
                        content = source["content"]
                        st.text(content[:400] + "..." if len(content) > 400 else content)
                        if i < len(chunks) - 1:
                            st.divider()


def process_question_agent(question: str):
    """Process question using agent mode."""
    with st.chat_message("assistant", avatar="🤖"):
        if not st.session_state.agent:
            st.error("Agent not initialized. Please upload documents first.")
            st.session_state.messages.pop()
            return
        
        reasoning_container = st.container()
        answer_placeholder = st.empty()
        
        try:
            with reasoning_container:
                with st.expander("🧠 Agent Reasoning", expanded=True):
                    step_placeholder = st.empty()
                    steps_display = []
                    
                    def display_step(step: AgentStep):
                        steps_display.append(step)
                        step_text = ""
                        for s in steps_display:
                            step_text += f"**Step {s.step_num}**\n\n"
                            step_text += f"💭 *{s.thought}*\n\n"
                            if s.action == "search":
                                step_text += f"🔍 Searching: `{s.action_input}`\n\n"
                                if s.observation:
                                    step_text += f"📄 Found: {s.observation[:200]}...\n\n"
                            step_text += "---\n\n"
                        step_placeholder.markdown(step_text)
                    
                    result = st.session_state.agent.run(question)
                    
                    for step in result.steps:
                        display_step(step)
            
            answer_placeholder.markdown(result.answer)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.answer,
                "sources": result.sources,
                "agent_steps": [s.to_dict() for s in result.steps]
            })
            
            # Show sources
            if result.sources:
                with st.expander("📚 View Sources", expanded=False):
                    seen = set()
                    for source in result.sources:
                        key = (source.get("doc_name"), source.get("page"))
                        if key not in seen:
                            seen.add(key)
                            st.markdown(f"**{source['doc_name']}** — Page {source['page']}")
                            content = source["content"]
                            st.text(content[:400] + "..." if len(content) > 400 else content)
                            st.divider()
            
            st.caption(f"📊 {result.total_retrievals} searches performed")
            
        except Exception as e:
            handle_api_error(e)
            st.session_state.messages.pop()
            return


def render_main_content(agent_mode: bool, use_hybrid: bool, use_reranking: bool, style: str):
    """Render the main chat interface."""
    # Header
    col_title, col_links = st.columns([3, 1])
    with col_title:
        st.markdown("""
        <div style="margin-bottom: 1rem;">
            <h1 style="margin-bottom: 0.25rem;">💬 Chat with Your Documents</h1>
            <p style="opacity: 0.7; margin: 0;">Upload PDFs and ask questions about their content</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_links:
        st.markdown(f"""
        <div style="text-align: right; padding-top: 1rem;">
            <a href="{GITHUB_URL}" target="_blank" style="
                display: inline-block;
                padding: 0.4rem 0.8rem;
                background: rgba(255,255,255,0.1);
                border-radius: 6px;
                text-decoration: none;
                font-size: 0.85rem;
            ">⭐ GitHub</a>
        </div>
        """, unsafe_allow_html=True)
    
    # Active features badges
    badges = []
    if agent_mode:
        badges.append("🤖 Agent Mode")
    if use_hybrid:
        badges.append("🔀 Hybrid Search")
    if use_reranking:
        badges.append("📊 Reranking")
    
    if badges:
        badge_html = " ".join([f'<span class="feature-badge">{b}</span>' for b in badges])
        st.markdown(f'<div style="margin-bottom: 1.5rem;">{badge_html}</div>', unsafe_allow_html=True)
    
    # Suggested questions (only show if no messages yet)
    if not st.session_state.messages and st.session_state.suggested_questions:
        st.markdown("**💡 Suggested Questions:**")
        cols = st.columns(min(3, len(st.session_state.suggested_questions)))
        for i, question in enumerate(st.session_state.suggested_questions):
            with cols[i % 3]:
                if st.button(
                    question[:50] + "..." if len(question) > 50 else question,
                    key=f"suggest_{i}",
                    use_container_width=True
                ):
                    if not check_api_key():
                        show_api_key_error()
                    else:
                        st.session_state.suggested_questions = []
                        st.session_state.messages.append({"role": "user", "content": question})
                        st.rerun()
    
    # Chat history
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            
            if message["role"] == "assistant":
                # Show agent reasoning if available
                if "agent_steps" in message and message["agent_steps"]:
                    with st.expander("🧠 View Reasoning"):
                        for step in message["agent_steps"]:
                            st.markdown(f"**Step {step['step']}:** {step['thought'][:150]}...")
                            if step["action"] == "search":
                                st.caption(f"🔍 Searched: {step['action_input']}")
                
                # Show sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("📚 View Sources"):
                        seen = set()
                        for source in message["sources"]:
                            doc_name = source.get("doc_name", "Unknown")
                            page = source.get("page", 0)
                            key = (doc_name, page)
                            if key not in seen:
                                seen.add(key)
                                st.markdown(f"**{doc_name}** — Page {page}")
                                content = source.get("content", "")
                                st.text(content[:400] + "..." if len(content) > 400 else content)
                                st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.documents:
            st.warning("⚠️ Please upload at least one PDF first")
        elif not check_api_key():
            show_api_key_error()
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
            
            if agent_mode:
                process_question_agent(prompt)
            else:
                process_question_standard(prompt, use_hybrid, use_reranking, style)
            
            st.rerun()
    
    # Empty state
    if not st.session_state.documents:
        st.markdown("""
        <div style="
            text-align: center;
            padding: 3rem;
            border: 2px dashed rgba(128,128,128,0.3);
            border-radius: 12px;
            margin-top: 2rem;
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📄</div>
            <h3 style="margin-bottom: 0.5rem;">No Documents Yet</h3>
            <p style="opacity: 0.7;">Upload PDF files using the sidebar to get started</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    init_session_state()
    
    # Render sidebar and get settings
    agent_mode, use_hybrid, use_reranking, style, uploaded_files = render_sidebar()
    
    # Process any uploaded files
    if uploaded_files:
        process_uploads(uploaded_files)
    
    # Render main content
    render_main_content(agent_mode, use_hybrid, use_reranking, style)


if __name__ == "__main__":
    main()