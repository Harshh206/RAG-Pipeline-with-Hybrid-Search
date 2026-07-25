from ingestion.pipeline import IngestionPipeline
from retrieval.retrieval_pipeline import create_retrieval_pipeline
from rag.rag_chain import create_rag_chain
from generation.chat_interface import run_chat
from config import config
import argparse
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Document RAG Pipeline")

    # Ingestion options
    parser.add_argument("--file", type=str, help="Process single file")
    parser.add_argument("--dir", type=str, help="Process directory")

    # Retrieval options
    parser.add_argument("--k", type=int, default=5, help="Top documents to retrieve")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Reranker score threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="qllama/bge-reranker-v2-m3:latest",
        help="Ollama reranker model",
    )

    # Query options
    parser.add_argument("--query", type=str, help="Single query")
    parser.add_argument(
        "--query-only", action="store_true", help="Skip ingestion, only query"
    )

    # Chat options
    parser.add_argument("--chat", action="store_true", help="Start interactive chat")
    parser.add_argument(
        "--llm", type=str, default=None, help="LLM model name (default: from config)"
    )

    args = parser.parse_args()

    # Update config if directory provided
    if args.dir:
        config.input_dir = args.dir

    # Override LLM if specified
    if args.llm:
        config.chat_model = args.llm

    # Initialize pipeline
    pipeline = IngestionPipeline(config)

    # Process documents if not query-only
    if not args.query_only:
        if args.file:
            result = pipeline.process_single_file(args.file)
        else:
            result = pipeline.process()

        print(json.dumps(result, indent=2))

        if result["status"] != "success":
            logger.error("Ingestion failed")
            return

    # Start chat mode
    if args.chat:
        run_chat(
            config=config,
            top_k=args.k,
            score_threshold=args.threshold,
            reranker_model=args.reranker_model,
        )
        return

    # Single query
    if args.query:
        print(f"\n📝 Query: {args.query}")
        print(f"Top-K: {args.k}")
        print(f"Threshold: {args.threshold}\n")

        try:
            # Create RAG chain
            rag = create_rag_chain(
                config=config,
                top_k=args.k,
                score_threshold=args.threshold,
                reranker_model=args.reranker_model,
            )

            # Get response
            response = rag.ask(args.query)

            if response.get("success"):
                print("=" * 60)
                print("🤖 Answer:")
                print("=" * 60)
                print(f"\n{response['answer']}\n")

                # Show sources
                sources = response.get("sources", [])
                if sources:
                    print("📚 Sources:")
                    for source in sources[:5]:
                        print(f"   • {source.get('source', 'Unknown')}")
                        print(f"     {source.get('content', '')[:100]}...\n")

                print(f"📊 Used {response.get('document_count', 0)} documents")
            else:
                print(f"❌ Error: {response.get('answer', 'Unknown error')}")

        except Exception as e:
            logger.error(f"Query failed: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
