from langchain_core.documents import Document
from retrieval.retrieval_pipeline import create_retrieval_pipeline
from typing import List, Dict, Any, Optional, Generator
import logging
from generation.llm import LMManager
from config import config

logger = logging.getLogger(__name__)


class RAGChain:
    """Simple RAG chain with retrieval and generation"""

    def __init__(
        self,
        retrieval_pipeline,
        llm_manager: Optional[LMManager] = None,
        k: int = 3,
        **kwargs,
    ):
        """
        Initialize RAG chain

        Args:
            retrieval_pipeline: Your retrieval pipeline
            llm_manager: LLM manager (creates default if None)
            k: Number of documents to retrieve
        """
        self.retrieval = retrieval_pipeline
        self.k = k

        # Use provided LLM or create default
        if llm_manager is None:
            self.llm = LMManager(
                model_name=config.chat_model,
                base_url=config.base_url,
            )
        else:
            self.llm = llm_manager

    def ask(
        self, query: str, k: Optional[int] = None, return_sources: bool = True, **kwargs
    ) -> Dict[str, Any]:
        """
        Ask a question using RAG

        Args:
            query: User question
            k: Number of documents to retrieve
            return_sources: Include sources in response

        Returns:
            Dictionary with answer, sources, and metadata
        """
        try:
            # Retrieve documents
            final_k = k or self.k
            documents = self.retrieval.retrieve(query, k=final_k, **kwargs)

            if not documents:
                return {
                    "query": query,
                    "answer": "No relevant documents found.",
                    "sources": [],
                    "document_count": 0,
                    "success": False,
                }

            # Format context
            context = self._format_context(documents)

            # Generate answer
            answer = self.llm.generate_with_context(query, context)

            # Prepare response
            response = {
                "query": query,
                "answer": answer,
                "document_count": len(documents),
                "success": True,
            }

            if return_sources:
                response["sources"] = self._format_sources(documents)

            return response

        except Exception as e:
            logger.error(f"RAG error: {e}")
            return {
                "query": query,
                "answer": f"Error: {str(e)}",
                "sources": [],
                "document_count": 0,
                "success": False,
            }

    def stream_ask(
        self, query: str, k: Optional[int] = None, **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream the answer

        Yields:
            Chunks of the response
        """
        try:
            # Retrieve documents
            final_k = k or self.k
            documents = self.retrieval.retrieve(query, k=final_k, **kwargs)

            if not documents:
                yield {
                    "type": "error",
                    "content": "No relevant documents found.",
                    "done": True,
                }
                return

            # Send sources first
            yield {
                "type": "sources",
                "sources": self._format_sources(documents),
                "document_count": len(documents),
            }

            # Format context
            context = self._format_context(documents)

            # Create prompt
            system_prompt = """You are a helpful assistant. Answer questions based ONLY on the provided context.
If you cannot answer from the context, say "I don't have enough information to answer this."
Be concise and accurate."""

            prompt = f"""Context:
{context}

Question: {query}

Answer based on the context above:"""

            # Stream generation
            full_response = ""
            for chunk in self.llm.stream_generate(prompt, system_prompt):
                # Handle both string and list chunks
                chunk_str = chunk if isinstance(chunk, str) else str(chunk)
                full_response += chunk_str
                yield {"type": "chunk", "content": chunk_str, "partial": full_response}

            # Final completion
            yield {"type": "complete", "content": full_response, "done": True}

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield {"type": "error", "content": str(e), "done": True}

    def _format_context(self, documents: List[Document]) -> str:
        """Format documents as context"""
        parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", f"Document {i}")
            parts.append(f"[{i}] From: {source}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Format sources"""
        sources = []
        for i, doc in enumerate(documents, 1):
            sources.append(
                {
                    "id": i,
                    "content": (
                        doc.page_content[:200] + "..."
                        if len(doc.page_content) > 200
                        else doc.page_content
                    ),
                    "source": doc.metadata.get("source", "Unknown"),
                    "file_name": doc.metadata.get("file_name", "Unknown"),
                }
            )
        return sources


def create_rag_chain(
    config,
    top_k: int = 5,
    score_threshold: float = 0.70,
    reranker_model: str = "qllama/bge-reranker-v2-m3:latest",
    **kwargs,
) -> RAGChain:
    """
    Create a RAG chain

    Args:
        config: PipelineConfig
        top_k: Number of documents to retrieve
        score_threshold: Minimum reranker score to keep a document
        reranker_model: Ollama reranker model name

    Returns:
        RAGChain instance
    """

    # Create retrieval pipeline
    retrieval = create_retrieval_pipeline(
        config=config,
        top_k=top_k,
        score_threshold=score_threshold,
        reranker_model=reranker_model,
        **kwargs,
    )

    # Create LLM
    llm = LMManager(
        model_name=getattr(config, "llm", "llama3:8b"), base_url=config.base_url
    )

    # Create RAG chain
    return RAGChain(retrieval_pipeline=retrieval, llm_manager=llm, k=top_k)
