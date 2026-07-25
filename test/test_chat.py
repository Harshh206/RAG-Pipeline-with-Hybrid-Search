#!/usr/bin/env python
"""
Test your RAG system quickly
"""

from config import config
from rag.rag_chain import create_rag_chain
from generation.chat_interface import run_chat
# from retrieval.retrieval_pipeline import create_retrieval_pipeline
# from genration.llm import LLMManager
import logging

logging.basicConfig(level=logging.INFO)


def test_rag():
    """Simple RAG test"""
    print("\n" + "=" * 60)
    print("Testing RAG System")
    print("=" * 60 + "\n")

    # Create RAG chain
    rag = create_rag_chain(config=config, top_k=5, score_threshold=0.70)

    # Test queries
    test_queries = [
        "What is the main topic of the documents?",
    ]

    for query in test_queries:
        print(f"\n❓ Query: {query}")
        response = rag.ask(query)

        if response.get("success"):
            print(f"🤖 Answer: {response['answer']}")
            print(f"📚 Used {response.get('document_count', 0)} documents")
        else:
            print(f"❌ Error: {response.get('answer', 'Unknown')}")

        print("-" * 40)


def test_chat():
    """Test chat mode"""
    

    print("\nStarting chat mode...")
    run_chat(config=config, top_k=5)


if __name__ == "__main__":
    # Run tests
    # test_rag()

    # Uncomment for chat
    test_chat()
