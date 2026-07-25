# app.py
import streamlit as st
import logging
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from rag.rag_chain import create_rag_chain
from generation.llm import LMManager
from retrieval.retrieval_pipeline import create_retrieval_pipeline
from ingestion.pipeline import IngestionPipeline
from ingestion.vector_store import ChromaVectorStore
from ingestion.embeddings import EmbeddingManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stChatMessage.user {
        background-color: #f0f2f6;
    }
    .stChatMessage.assistant {
        background-color: #e8f4fd;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        font-size: 0.8rem;
        border-left: 3px solid #4CAF50;
    }
    .metadata-box {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        font-size: 0.8rem;
        border-left: 3px solid #ffc107;
    }
    .stAlert {
        padding: 0.5rem;
    }
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #e6e6e6;
        margin-bottom: 1rem;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    .status-online {
        background-color: #4CAF50;
    }
    .status-offline {
        background-color: #f44336;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


class StreamlitChatSession:
    """Manages chat session with Streamlit state"""

    def __init__(self):
        self.rag = None
        self.llm = None
        self.is_initialized = False
        self.vector_store = None
        self.embedding_manager = None

    def initialize(self, config, top_k=5, score_threshold=0.70, reranker_model=None):
        """Initialize RAG components"""
        try:
            # Check if Ollama is running
            import requests

            try:
                response = requests.get(f"{config.base_url}/api/tags", timeout=5)
                if response.status_code != 200:
                    st.error(
                        f"⚠️ Ollama server at {config.base_url} is not responding properly. Please check if Ollama is running."
                    )
                    return False
            except requests.exceptions.ConnectionError:
                st.error(
                    f"⚠️ Cannot connect to Ollama at {config.base_url}. Please make sure Ollama is running."
                )
                st.info("💡 Run: `ollama serve` or start Ollama from your system menu")
                return False
            except requests.exceptions.Timeout:
                st.error(
                    f"⚠️ Connection to Ollama at {config.base_url} timed out. Please check your network."
                )
                return False

            # Initialize embedding manager and vector store for cleanup operations
            self.embedding_manager = EmbeddingManager(
                model_name=config.embedding_model,
                base_url=config.base_url,
            )

            self.vector_store = ChromaVectorStore(
                embedding_manager=self.embedding_manager,
                persist_directory=config.chroma_persist_dir,
                collection_name=config.collection_name,
            )

            # Create RAG chain
            self.rag = create_rag_chain(
                config=config,
                top_k=top_k,
                score_threshold=score_threshold,
                reranker_model=reranker_model or config.reranker_model,
            )

            self.llm = LMManager(
                model_name=config.chat_model,
                base_url=config.base_url,
                temperature=config.temperature,
            )

            # Validate connections
            if not self.llm.validate():
                st.warning(
                    "⚠️ LLM validation failed. The model might not be available."
                )
                st.info(
                    f"💡 Make sure model '{config.chat_model}' is pulled: `ollama pull {config.chat_model}`"
                )
                return False

            # Check if vector store has documents
            try:
                doc_count = self.vector_store.get_collection_stats().get(
                    "document_count", 0
                )
                if doc_count == 0:
                    st.info(
                        "📚 Vector store is empty. Upload documents to get started."
                    )
            except Exception as e:
                st.warning(f"Could not check vector store: {e}")

            self.is_initialized = True
            return True

        except Exception as e:
            st.error(f"❌ Failed to initialize RAG system: {str(e)}")
            logger.error(f"Initialization error: {e}")
            return False

    def clean_vector_store(self):
        """Delete all documents from the vector store"""
        try:
            if self.vector_store is None:
                # Initialize if not already done
                self.embedding_manager = EmbeddingManager(
                    model_name=config.embedding_model,
                    base_url=config.base_url,
                )
                self.vector_store = ChromaVectorStore(
                    embedding_manager=self.embedding_manager,
                    persist_directory=config.chroma_persist_dir,
                    collection_name=config.collection_name,
                )

            # Get count before deletion
            stats = self.vector_store.get_collection_stats()
            doc_count = stats.get("document_count", 0)

            if doc_count == 0:
                return {
                    "success": True,
                    "message": "Vector store is already empty.",
                    "deleted_count": 0,
                }

            # Delete the collection
            self.vector_store.delete_collection()

            # Recreate the collection (delete_collection removes it, we need it to exist)
            # The ChromaVectorStore will recreate it on next operation
            # We'll just log that it was deleted

            # Mark as not initialized so it gets recreated
            self.is_initialized = False
            self.rag = None

            return {
                "success": True,
                "message": f"Successfully deleted {doc_count} documents from vector store.",
                "deleted_count": doc_count,
            }

        except Exception as e:
            logger.error(f"Error cleaning vector store: {e}")
            return {
                "success": False,
                "message": f"Failed to clean vector store: {str(e)}",
                "deleted_count": 0,
            }

    def ask(self, question: str) -> dict:
        """Ask a question through the RAG chain"""
        if not self.is_initialized or self.rag is None:
            return {
                "success": False,
                "answer": "System not initialized. Please check the sidebar settings.",
                "sources": [],
                "document_count": 0,
            }

        try:
            return self.rag.ask(question, return_sources=True)
        except Exception as e:
            logger.error(f"Error during ask: {e}")
            return {
                "success": False,
                "answer": f"Error: {str(e)}",
                "sources": [],
                "document_count": 0,
            }


def initialize_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_session" not in st.session_state:
        st.session_state.chat_session = StreamlitChatSession()

    if "is_initialized" not in st.session_state:
        st.session_state.is_initialized = False

    if "ingestion_result" not in st.session_state:
        st.session_state.ingestion_result = None

    if "show_clean_warning" not in st.session_state:
        st.session_state.show_clean_warning = False

    if "vector_store_stats" not in st.session_state:
        st.session_state.vector_store_stats = {}


def display_message(message):
    """Display a single message in the chat"""
    role = message.get("role", "user")
    content = message.get("content", "")
    sources = message.get("sources", [])

    with st.chat_message(role):
        st.markdown(content)

        # Display sources if available
        if sources and role == "assistant":
            with st.expander("📚 Sources"):
                for i, source in enumerate(sources[:5], 1):
                    st.markdown(
                        f"""
                    <div class="source-box">
                        <strong>Source {i}:</strong> {source.get('source', 'Unknown')}<br>
                        <span style="color: #666;">{source.get('content', '')[:150]}...</span>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Show metadata if available
                    if source.get("file_name"):
                        st.caption(f"📄 File: {source.get('file_name')}")


def handle_user_input(prompt: str):
    """Process user input and generate response"""
    if not prompt:
        return

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            if st.session_state.chat_session.is_initialized:
                response = st.session_state.chat_session.ask(prompt)
            else:
                response = {
                    "success": False,
                    "answer": "⚠️ System not initialized. Please check the sidebar settings.",
                    "sources": [],
                    "document_count": 0,
                }

        if response.get("success"):
            st.markdown(response["answer"])

            # Display sources
            sources = response.get("sources", [])
            if sources:
                with st.expander(f"📚 Sources ({len(sources)} retrieved)"):
                    for i, source in enumerate(sources[:5], 1):
                        st.markdown(
                            f"""
                        <div class="source-box">
                            <strong>Source {i}:</strong> {source.get('source', 'Unknown')}<br>
                            <span style="color: #666;">{source.get('content', '')[:200]}...</span>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                        if source.get("file_name"):
                            st.caption(f"📄 File: {source.get('file_name')}")

            # Add assistant message to history
            st.session_state.messages.append(
                {"role": "assistant", "content": response["answer"], "sources": sources}
            )
        else:
            st.error(response.get("answer", "Unknown error occurred"))


def main():
    """Main Streamlit app"""

    # Initialize session state
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")

        # Status indicator
        status_color = (
            "status-online" if st.session_state.is_initialized else "status-offline"
        )
        status_text = "🟢 Online" if st.session_state.is_initialized else "🔴 Offline"
        st.markdown(
            f"""
        <div style="padding: 0.5rem; background-color: #f0f2f6; border-radius: 0.5rem; margin-bottom: 1rem;">
            <span class="status-indicator {status_color}"></span>
            <strong>Status:</strong> {status_text}
        </div>
        """,
            unsafe_allow_html=True,
        )

        if st.session_state.is_initialized:
            st.caption(f"🔗 Connected to: {config.base_url}")
            st.caption(f"🧠 Model: {config.chat_model}")

        # Configuration section
        with st.expander("🔧 Configuration", expanded=True):
            top_k = st.slider(
                "Documents to retrieve (K)",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of relevant documents to retrieve",
            )

            score_threshold = st.slider(
                "Score threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.70,
                step=0.05,
                help="Minimum relevance score for documents",
            )

            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.5,
                value=0.8,
                step=0.1,
                help="Controls randomness in responses",
            )

            config.temperature = temperature

        # Document upload section
        with st.expander("📤 Upload Documents", expanded=True):
            st.markdown("Upload documents to index for RAG:")

            # Show current vector store status
            try:
                if st.session_state.chat_session.vector_store:
                    stats = (
                        st.session_state.chat_session.vector_store.get_collection_stats()
                    )
                    doc_count = stats.get("document_count", 0)
                    st.caption(f"📊 Current documents in vector store: {doc_count}")
                else:
                    st.caption("📊 Vector store not initialized")
            except:
                st.caption("📊 Unable to get vector store stats")

            uploaded_files = st.file_uploader(
                "Choose files",
                type=[
                    "pdf",
                    "doc",
                    "docx",
                    "txt",
                    "md",
                    "csv",
                    "json",
                    "html",
                    "ppt",
                    "pptx",
                    "xlsx",
                    "xls",
                ],
                accept_multiple_files=True,
                help="Upload documents to add to the knowledge base",
            )

            if uploaded_files:
                st.info(f"📁 {len(uploaded_files)} file(s) ready for ingestion")

                # Clean vector store checkbox
                clean_before_upload = st.checkbox(
                    "🧹 Clean vector store before uploading new documents",
                    value=False,
                    help="This will delete ALL existing documents from the vector store before adding new ones.",
                )

                col1, col2 = st.columns(2)

                with col1:
                    if st.button(
                        "🚀 Ingest Documents", type="primary", use_container_width=True
                    ):
                        # If clean checkbox is checked, clean the vector store first
                        if clean_before_upload:
                            with st.spinner("🧹 Cleaning vector store..."):
                                clean_result = (
                                    st.session_state.chat_session.clean_vector_store()
                                )

                                if clean_result["success"]:
                                    st.success(f"✅ {clean_result['message']}")
                                    # Reset initialization state
                                    st.session_state.is_initialized = False
                                    st.session_state.messages = []
                                else:
                                    st.error(f"❌ {clean_result['message']}")
                                    st.stop()

                        # Now ingest the new documents
                        with st.spinner("Processing documents..."):
                            try:
                                # Save uploaded files temporarily
                                temp_dir = Path("./temp_uploads")
                                temp_dir.mkdir(exist_ok=True)

                                saved_paths = []
                                for uploaded_file in uploaded_files:
                                    temp_path = temp_dir / uploaded_file.name
                                    with open(temp_path, "wb") as f:
                                        f.write(uploaded_file.getbuffer())
                                    saved_paths.append(temp_path)

                                # Process through ingestion pipeline
                                pipeline = IngestionPipeline(config)

                                result = {
                                    "status": "success",
                                    "documents_loaded": 0,
                                    "chunks_created": 0,
                                }

                                for file_path in saved_paths:
                                    file_result = pipeline.process_single_file(
                                        str(file_path)
                                    )
                                    if file_result.get("status") == "success":
                                        result["documents_loaded"] += 1
                                        result["chunks_created"] += file_result.get(
                                            "chunks_created", 0
                                        )

                                # Clean up temp files
                                import shutil

                                shutil.rmtree(temp_dir, ignore_errors=True)

                                if result["documents_loaded"] > 0:
                                    st.success(
                                        f"✅ Successfully processed {result['documents_loaded']} document(s), created {result['chunks_created']} chunks"
                                    )
                                    st.session_state.ingestion_result = result

                                    # Reinitialize to refresh vector store connection
                                    chat_session = st.session_state.chat_session
                                    chat_session.is_initialized = False
                                    st.session_state.is_initialized = False

                                    # Reinitialize the system
                                    with st.spinner("🔄 Reinitializing RAG system..."):
                                        success = chat_session.initialize(
                                            config,
                                            top_k=top_k,
                                            score_threshold=score_threshold,
                                        )
                                        st.session_state.is_initialized = success

                                    st.balloons()
                                    st.rerun()
                                else:
                                    st.warning(
                                        "⚠️ No documents were successfully processed"
                                    )

                            except Exception as e:
                                st.error(f"❌ Ingestion failed: {str(e)}")
                                logger.error(f"Ingestion error: {e}")

                with col2:
                    # Separate clean button for manual cleaning
                    if st.button(
                        "🧹 Clean VectorDB", use_container_width=True, type="secondary"
                    ):
                        st.session_state.show_clean_warning = True

                        # Show confirmation dialog
                        st.warning(
                            "⚠️ This will delete ALL documents from the vector store. This action cannot be undone."
                        )

                        col_confirm1, col_confirm2 = st.columns(2)
                        with col_confirm1:
                            if st.button("✅ Yes, Clean", use_container_width=True):
                                with st.spinner("🧹 Cleaning vector store..."):
                                    clean_result = (
                                        st.session_state.chat_session.clean_vector_store()
                                    )
                                    if clean_result["success"]:
                                        st.success(f"✅ {clean_result['message']}")
                                        st.session_state.is_initialized = False
                                        st.session_state.messages = []
                                        st.session_state.show_clean_warning = False
                                        st.rerun()
                                    else:
                                        st.error(f"❌ {clean_result['message']}")
                        with col_confirm2:
                            if st.button("❌ Cancel", use_container_width=True):
                                st.session_state.show_clean_warning = False
                                st.rerun()

        # About section
        with st.expander("ℹ️ About"):
            st.markdown("""
            ### RAG Chat Assistant
            **Retrieval-Augmented Generation** system with:
            - 🔍 Hybrid search (Dense + BM25)
            - 🎯 Cross-encoder reranking
            - 📄 Multi-format document support
            - 💬 Interactive chat with sources
            
            **Models Used:**
            - Embedding: `qwen3-embedding:0.6b`
            - LLM: `llama3:8b` (configurable)
            - Reranker: `BAAI/bge-reranker-v2-m3`
            """)

        # Control buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Reinitialize", use_container_width=True):
                with st.spinner("Reinitializing..."):
                    st.session_state.is_initialized = False
                    st.session_state.chat_session = StreamlitChatSession()
                    success = st.session_state.chat_session.initialize(
                        config, top_k=top_k, score_threshold=score_threshold
                    )
                    st.session_state.is_initialized = success

                    if success:
                        st.success("✅ Reinitialized successfully")
                        st.rerun()
                    else:
                        st.error("❌ Reinitialization failed")

        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    # Main chat area
    st.markdown(
        """
    <div class="main-header">
        <h1>🤖 RAG Chat Assistant</h1>
        <p style="color: #666;">Ask questions about your documents with RAG-powered retrieval</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Initialize the chat session
    if not st.session_state.is_initialized:
        with st.spinner("⏳ Initializing RAG system... This may take a moment."):
            success = st.session_state.chat_session.initialize(
                config,
                top_k=st.session_state.get("top_k", 5),
                score_threshold=st.session_state.get("score_threshold", 0.70),
            )
            st.session_state.is_initialized = success

            if not success:
                st.warning(
                    "⚠️ RAG system could not be initialized. Please check your configuration and try reinitializing."
                )
                st.info("""
                **Troubleshooting tips:**
                1. Make sure Ollama is running (`ollama serve`)
                2. Pull the required models:
                   - `ollama pull qwen3-embedding:0.6b`
                   - `ollama pull llama3:8b`
                3. Check your config.py settings
                4. Try reinitializing from the sidebar
                """)

    # Display chat messages
    for message in st.session_state.messages:
        display_message(message)

    # Handle user input
    if prompt := st.chat_input("Ask a question about your documents..."):
        handle_user_input(prompt)


if __name__ == "__main__":
    main()
