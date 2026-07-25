from langchain_chroma import Chroma
from langchain_core.documents import Document

from typing import List, Optional, Dict, Any
import logging
import uuid

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Manages ChromaDB operations"""

    def __init__(
        self,
        embedding_manager,
        persist_directory: str = "./chroma_db",
        collection_name: str = "my_rag_collection",
    ):

        self.embedding_manager = embedding_manager
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_manager.embeddings,
            persist_directory=persist_directory,
        )

    def add_texts(
        self, texts: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add texts directly to the vector store"""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        try:
            ids = self.vector_store.add_texts(texts, metadatas=metadatas, ids=ids)
            return ids
        except Exception as e:
            logger.error(f"Error adding texts to ChromaDB: {e}")
            raise

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add Document objects to the vector store"""
        try:
            ids = []
            for document in documents:
                document_id = document.metadata.get("id")
                if not document_id:
                    document_id = str(uuid.uuid4())
                    document.metadata["id"] = document_id
                ids.append(document_id)

            ids = self.vector_store.add_documents(documents, ids=ids)
            return ids
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise

    def search(self, query: str, k: int, **kwargs) -> List[Document]:
        """Search for similar documents"""
        try:
            return self.vector_store.similarity_search(query, k=k, **kwargs)
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []

    def search_with_score(self, query: str, k: int, **kwargs) -> List[tuple]:
        """Search with similarity scores"""
        try:
            return self.vector_store.similarity_search_with_score(query, k=k, **kwargs)
        except Exception as e:
            logger.error(f"Error searching ChromaDB with scores: {e}")
            return []

    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.vector_store.delete_collection()
            logger.info(f"Deleted collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")

    def get_all_documents(self) -> List[Document]:
        """Get all documents in the collection"""
        try:
            return self.vector_store.get()
        except Exception as e:
            logger.error(f"Error getting documents: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.vector_store._collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory,
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
