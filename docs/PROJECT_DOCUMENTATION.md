# QA Chatbot - Complete Project Documentation

## 1. Project Overview

A Retrieval-Augmented Generation (RAG) chatbot that ingests documents, splits them into chunks, generates embeddings, stores them in ChromaDB, and answers user questions using an LLM with retrieved context. All models run locally via Ollama. The UI is a Streamlit web app with PDF upload support.

---

## 2. Project Structure

```
QA-Chatbot/
├── app.py                          # Streamlit web UI entry point
├── config.py                       # Central configuration (PipelineConfig dataclass)
├── z.py                            # Scratch/test script for PyPDF loader
├── requirements.txt                # Python dependencies
│
├── ingestion/                      # Document ingestion pipeline
│   ├── __init__.py
│   ├── loader.py                   # Document loading from files
│   ├── splitter.py                 # Text chunking strategies
│   ├── embeddings.py               # Ollama embedding management
│   ├── vector_store.py             # ChromaDB vector store wrapper
│   ├── pipeline.py                 # Orchestrates full ingestion workflow
│   └── processors/
│       ├── cleaner.py              # Text cleaning and normalization
│       └── metadata_extract.py     # Metadata enrichment
│
├── retrieval/                      # Document retrieval pipeline
│   ├── __init__.py
│   ├── retriever.py                # BM25, Dense, and Hybrid retrievers
│   ├── reranker.py                 # Cross-encoder reranking
│   └── retrieval_pipeline.py       # Orchestrates retrieval + reranking
│
├── genration/                      # Generation pipeline (note: typo in folder name)
│   ├── __init__.py
│   ├── llm.py                      # Ollama LLM wrapper (ChatOllama)
│   ├── query_processor.py          # Query preprocessing and expansion
│   ├── chat_interface.py           # Interactive CLI chat session
│   └── (rag_chain.py lives in rag/)
│
├── rag/                            # RAG chain orchestration
│   ├── __init__.py
│   ├── rag_chain.py                # Combines retrieval + generation
│   └── main.py                     # CLI entry point with argparse
│
├── test/                           # Test scripts
│   ├── __init__.py
│   ├── test_pipeline.py            # Full ingestion pipeline test
│   ├── test_chat.py                # Chat and RAG system test
│   ├── test_embeddings.py          # Embedding generation test
│   ├── test_vectorstore.py         # ChromaDB operations test
│   ├── test_retrieval.py           # Retriever and reranker tests
│   └── test_performance.py         # Ingestion benchmark
│
├── data/                           # Input documents directory
├── chroma_db/                      # ChromaDB persistent storage
└── docs/                           # Documentation
    ├── PROJECT_DOCUMENTATION.md
    └── doc.md
```

---

## 3. Dependencies (requirements.txt)

| Package | Purpose |
|---|---|
| `langchain` | Core framework for document handling, chains, and messages |
| `langchain-community` | Community loaders (TextLoader, CSVLoader, JSONLoader, etc.) |
| `langchain-ollama` | Ollama LLM (`ChatOllama`) and embedding (`OllamaEmbeddings`) integration |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` and `MarkdownTextSplitter` for chunking |
| `langchain-chroma` | ChromaDB vector store integration with LangChain |
| `PyMuPDF` | PDF reading (used internally by MarkItDown) |
| `pypdf` | Alternative PDF loader (`PyPDFLoader` in test scripts) |
| `sentence-transformers` | Cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) |
| `chromadb` | Vector database for persistent storage and similarity search |
| `markitdown` | Converts PDF, DOCX, PPTX, XLSX, HTML to Markdown text |
| `streamlit` | Web UI framework for the chat interface |
| `docx2txt` | DOCX text extraction (dependency for unstructured/markitdown) |
| `unstructured` | Additional document parsing for non-standard formats |
| `openpyxl` | Excel file reading (XLSX support) |
| `jq` | JSON file parsing with `jq` schemas |
| `pytest` | Test runner |

### External Services Required

| Service | Purpose |
|---|---|
| **Ollama** (local server at `localhost:11434`) | Runs the LLM and embedding models locally |
| `llama3:8b` | Chat/generation model (configurable) |
| `qwen3-embedding:0.6b` | Embedding model producing 768-dim vectors (configurable) |
| `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker loaded via sentence-transformers |

---

## 4. Configuration (`config.py`)

All settings are centralized in the `PipelineConfig` dataclass:

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `640` | Maximum characters per chunk |
| `chunk_overlap` | `96` | Overlap between consecutive chunks |
| `chat_model` | `"llama3:8b"` | Ollama chat model name |
| `temperature` | `0.8` | LLM sampling temperature |
| `base_url` | `"http://localhost:11434"` | Ollama server URL |
| `embedding_model` | `"qwen3-embedding:0.6b"` | Embedding model name |
| `embedding_dimension` | `768` | Embedding vector dimension |
| `reranker_model` | `"BAAI/bge-reranker-v2-m3"` | Cross-encoder reranker model |
| `threshold` | `0.70` | Minimum reranker score to keep a document |
| `chroma_persist_dir` | `"./chroma_db"` | ChromaDB storage directory |
| `collection_name` | `"my_rag_collection"` | ChromaDB collection name |
| `input_dir` | `"./data"` | Default input directory for documents |

A global `config` singleton is exported for use across all modules.

---

## 5. Entry Points

### 5.1 Streamlit Web UI (`app.py`)

The primary user-facing entry point. Run with:

```bash
streamlit run app.py
```

**Features:**
- Sidebar with PDF upload and ingestion button
- Database stats display (chunk count)
- Database clear button
- Chat history clear button
- Interactive chat input with source display
- Cached resource management (`@st.cache_resource`) for embeddings, vector store, and RAG chain

**Workflow:**
1. Loads embedding model, vector store, and RAG chain on first use (cached)
2. User uploads a PDF via sidebar -> click "Ingest PDF" -> pipeline processes it
3. User types a question in the chat input -> RAG chain retrieves context and generates answer
4. Sources are shown in expandable sections under each response

### 5.2 CLI Entry Point (`rag/main.py`)

Command-line interface with argparse. Run with:

```bash
python -m rag.main --file data/doc.pdf           # Ingest a single file
python -m rag.main --dir ./data                   # Ingest entire directory
python -m rag.main --query-only --query "What is X?"  # Query only (skip ingestion)
python -m rag.main --chat                         # Start interactive chat
python -m rag.main --chat --llm llama3:8b --k 5  # Chat with custom settings
```

**CLI Arguments:**

| Argument | Description |
|---|---|
| `--file` | Process a single file |
| `--dir` | Process a directory |
| `--k` | Top-K documents to retrieve (default: 5) |
| `--threshold` | Reranker score threshold (default: 0.70) |
| `--reranker-model` | Reranker model name |
| `--query` | Single query to answer |
| `--query-only` | Skip ingestion, only query |
| `--chat` | Start interactive chat mode |
| `--llm` | Override LLM model name |

### 5.3 Interactive Chat CLI (`genration/chat_interface.py`)

A standalone interactive chat loop with conversation history. Run directly:

```bash
python genration/chat_interface.py
```

**Commands:** `/help`, `/history`, `/clear`, `/quit`

---

## 6. Module-by-Module Breakdown

### 6.1 Ingestion Module (`ingestion/`)

#### 6.1.1 `loader.py` - DocumentLoader

Loads documents from files into LangChain `Document` objects.

**Supported Formats:**
- **MarkItDown formats** (converted to Markdown): `.pdf`, `.doc`, `.docx`, `.ppt`, `.pptx`, `.xlsx`, `.xls`, `.html`
- **LangChain loaders**: `.txt` (TextLoader), `.md` (UnstructuredMarkdownLoader), `.csv` (CSVLoader), `.json` (JSONLoader)

**Key Methods:**
- `load_single_file(file_path)` -> `List[Document]`: Detects file extension, routes to appropriate loader, attaches metadata (source, file_name, file_type, file_size, is_markdown)
- `load_directory(recursive=True)` -> `List[Document]`: Globs all supported files in the input directory and loads them

**Step-by-step:**
1. Check file extension against supported set
2. If MarkItDown format: call `MarkItDown().convert()` and wrap result in a `Document`
3. Otherwise: instantiate the appropriate LangChain loader class
4. For `.json` files: uses `jq_schema="."` to load entire JSON
5. Attach metadata dict to every document
6. Return list of documents

#### 6.1.2 `processors/cleaner.py` - TextCleaner

Normalizes text content in documents.

**Step-by-step:**
1. Collapse multiple newlines (`\n\s*\n` -> `\n\n`)
2. Collapse multiple spaces -> single space
3. Remove BOM character (`\ufeff`)
4. Replace non-breaking spaces (`\xa0`) with regular space
5. Optionally strip markdown syntax (headers, bold/italic, links) if `preserve_markdown=False`
6. Strip leading/trailing whitespace
7. Mark document with `metadata["cleaned"] = True`

**Default behavior:** `preserve_markdown=True` (keeps markdown structure intact)

#### 6.1.3 `processors/metadata_extract.py` - MetadataExtractor

Enriches each document with computed metadata.

**Step-by-step:**
1. Copy existing document metadata
2. Compute MD5 content hash (`content_hash`) for deduplication
3. Extract title: checks first 20 lines for markdown heading (`#`), falls back to first non-empty line (truncated to 100 chars)
4. Detect language (currently hardcoded to `"en"`)
5. Count words (`word_count`) and characters (`char_count`)
6. Extract top-5 keywords by frequency, filtering out stopwords (`this`, `that`, `with`, etc.)
7. Add ingestion timestamp (`ingestion_timestamp`)

#### 6.1.4 `splitter.py` - Chunking Strategies

Implements a strategy pattern for document chunking.

**Classes:**
- `ChunkStrategy` (base): holds `chunk_size` and `chunk_overlap`, defines `split()` interface
- `RecursiveChunker`: uses `RecursiveCharacterTextSplitter` for plain text
- `MarkdownChunker`: uses `MarkdownTextSplitter`, extracts heading metadata (`heading_level`, `section_title`)
- `AutoChunker`: automatically selects strategy per document:
  - If `loader == "markitdown"` or `file_type == "md"` -> `MarkdownChunker`
  - Otherwise -> `RecursiveChunker`

**Factory:** `get_chunker(strategy="auto", **kwargs)` returns the appropriate chunker instance.

#### 6.1.5 `embeddings.py` - EmbeddingManager

Wraps `OllamaEmbeddings` from langchain-ollama.

**Key Methods:**
- `embed_documents(texts)` -> `List[List[float]]`: Batch embed a list of text strings
- `embed_query(query)` -> `List[float]`: Embed a single query string
- `embed_chunks(chunks)`: Embed document chunks and attach embeddings to their metadata
- `get_embedding_dimension()` -> `Optional[int]`: Test-embeds "test" and returns vector length
- `validate_connection()` -> `bool`: Tests connectivity to Ollama server

**Initialization:** Connects to Ollama at `base_url` with the configured model and dimension.

#### 6.1.6 `vector_store.py` - ChromaVectorStore

Wraps `langchain_chroma.Chroma` for persistent vector storage.

**Key Methods:**
- `add_texts(texts, metadatas, ids)`: Add raw text strings with optional metadata
- `add_documents(documents)`: Add LangChain `Document` objects (auto-generates UUIDs if not present)
- `search(query, k)`: Similarity search returning top-k documents
- `search_with_score(query, k)`: Similarity search returning documents with distance scores
- `delete_collection()`: Drop the entire collection
- `get_all_documents()`: Retrieve all stored documents
- `get_collection_stats()`: Return collection name, document count, and persist directory

#### 6.1.7 `pipeline.py` - IngestionPipeline

Orchestrates the complete ingestion workflow.

**`process()` step-by-step:**
1. **Load documents**: Use `DocumentLoader` to load files from `input_dir` (or a single file)
2. **Clean text**: Run `TextCleaner.batch_clean()` to normalize whitespace and remove artifacts
3. **Extract metadata**: Run `MetadataExtractor.batch_extract()` to add content hashes, titles, keywords, word counts
4. **Chunk documents**: Use `AutoChunker.split()` to split documents into overlapping chunks
5. **Store in ChromaDB**: Call `ChromaVectorStore.add_documents()` which generates embeddings and persists them
6. **Collect stats**: Query `ChromaVectorStore.get_collection_stats()` for document count

**Methods:**
- `process(input_path=None)` -> `dict`: Run full pipeline
- `process_single_file(file_path)` -> `dict`: Process one file
- `process_directory(dir_path)` -> `dict`: Process all files in a directory

**Return format:**
```python
{
    "status": "success" | "failed",
    "documents_loaded": int,
    "chunks_created": int,
    "stored_ids": list,
    "collection_stats": dict
}
```

---

### 6.2 Retrieval Module (`retrieval/`)

#### 6.2.1 `retriever.py` - Retrieval Strategies

Implements three retriever classes:

**BM25SparseRetriever:**
- Operates on the raw document texts stored in ChromaDB
- Implements BM25 scoring from scratch (no external library):
  1. Fetch all documents from ChromaDB collection
  2. Tokenize documents and query (lowercase split)
  3. Compute document frequency (DF) for each term
  4. For each document, compute TF-normalized BM25 score per query term
  5. Use IDF formula: `log((N - df + 0.5) / (df + 0.5) + 1)`
  6. Use TF normalization: `(tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))`
  7. Sort by score descending, return top-k as `(Document, score)` tuples

**DenseRetriever:**
- Uses ChromaDB's built-in similarity search with the Ollama embedding model
- Returns `(Document, distance_score)` tuples

**HybridRetriever:**
- Combines BM25 (sparse) and Dense retrieval using Reciprocal Rank Fusion (RRF)
- Step-by-step:
  1. Retrieve `max(k*3, 30)` results from both BM25 and Dense retrievers
  2. For each result from each retriever, compute RRF score: `1 / (k + rank + 1)` where `k=60`
  3. Merge scores by document content (first 200 chars as key)
  4. Sort merged results by combined RRF score
  5. Return top-k documents

#### 6.2.2 `reranker.py` - CrossEncoderReranker

Reranks retrieved documents using a cross-encoder model.

**Step-by-step:**
1. Initialize `sentence_transformers.CrossEncoder` with `BAAI/bge-reranker-v2-m3`
2. Truncate input to `max_docs=20` documents (excess passed through unranked)
3. Create query-document pairs, truncate each document to 512 chars
4. Run `model.predict(pairs)` to get relevance scores
5. Sort by score descending
6. Filter by `score_threshold` (default 0.70) - keep only documents above threshold
7. If no documents pass threshold, keep the highest-scored one as fallback
8. **Fallback mode** (if model fails to load): keyword overlap scoring (fraction of query words found in document)

#### 6.2.3 `retrieval_pipeline.py` - RetrievalPipeline

Orchestrates the full retrieval workflow.

**Pipeline flow:**
```
User Query
    ├── Dense Vector Search (qwen3-embedding:0.6b)
    └── Sparse BM25 Search
            │
            ▼
    Combined Pool (RRF Fusion)
            │
            ▼
    Cross-Encoder Reranker (bge-reranker-v2-m3)
            │
            ▼
    Dynamic Score Filter (> threshold)
            │
            ▼
    Top N Chunks -> LLM
```

**Key Methods:**
- `retrieve(query, k)`: Runs hybrid retrieval (3x top_k), then reranks, returns top_k
- `retrieve_with_scores(query, k)`: Returns results with rank, content, metadata, and source
- `retrieve_context(query, k)`: Returns formatted context string + source list (for RAG chain)
- `change_threshold(threshold)`: Dynamically adjust reranker threshold

**Factory:** `create_retrieval_pipeline(config, top_k, score_threshold, reranker_model)` creates a fully configured pipeline.

---

### 6.3 Generation Module (`genration/`)

#### 6.3.1 `llm.py` - LMManager

Wraps `ChatOllama` from langchain-ollama for text generation.

**Key Methods:**
- `generate(prompt, system_prompt=None)` -> `str`: Sends messages to Ollama, returns response text. Handles both string and list content responses from the model.
- `generate_with_context(query, context)` -> `str`: Pre-built RAG prompt template:
  - System prompt: "Answer questions based ONLY on the provided context..."
  - User prompt: `"Context: {context}, Question: {query}"`
- `stream_generate(prompt, system_prompt=None)`: Yields response chunks for streaming
- `validate()` -> `bool`: Tests LLM with a simple "Hello" prompt

**Message construction:** Uses LangChain's `SystemMessage` and `HumanMessage` classes.

#### 6.3.2 `query_processor.py` - QueryProcessor

Preprocesses and enhances user queries before retrieval.

**Methods:**
- `preprocess(query)`: Strip whitespace, collapse spaces, remove special characters (keeps word chars, spaces, `?!.,-`)
- `expand(query)`: Generate variations - removes trailing `?`, capitalizes first letter
- `extract_keywords(query)`: Remove stopwords (`what`, `is`, `the`, etc.), return remaining words (min 2 chars)
- `enhance(query)`: Full profile with original, processed, variations, keywords, and optional embedding

---

### 6.4 RAG Module (`rag/`)

#### 6.4.1 `rag_chain.py` - RAGChain

Combines retrieval and generation into a single question-answering chain.

**`ask(query, k, return_sources)` step-by-step:**
1. Call `retrieval_pipeline.retrieve(query, k)` to get relevant documents
2. If no documents found, return early with failure response
3. Format context: join documents with `[1] From: {source}\n{content}` separators
4. Call `llm.generate_with_context(query, context)` to generate the answer
5. Format sources: extract id, content preview, source path, file name from each document
6. Return dict with query, answer, document_count, success flag, and sources

**`stream_ask(query, k)` step-by-step:**
1. Retrieve documents
2. Yield sources metadata first
3. Format context and create prompt
4. Stream LLM response chunk by chunk, yielding partial responses
5. Yield completion signal with full response

**Factory:** `create_rag_chain(config, top_k, score_threshold, reranker_model)` builds the complete chain with retrieval pipeline and LLM.

#### 6.4.2 `chat_interface.py` - ChatSession

Manages interactive chat with conversation history.

**ChatSession class:**
- `ask(question)`: Calls `rag.ask()`, appends to history
- `show_history(limit)`: Prints last N Q&A pairs
- `clear()`: Resets history

**`run_chat()` function:**
1. Create RAG chain via `create_rag_chain()`
2. Create `ChatSession` with the chain
3. Enter loop: read user input, handle commands (`/quit`, `/help`, `/history`, `/clear`)
4. For regular input: call `session.ask()`, display answer and sources

---

## 7. Complete Data Flow

### Ingestion Flow (PDF Upload)

```
PDF file uploaded (app.py)
    │
    ▼
IngestionPipeline.process_single_file()
    │
    ├── 1. DocumentLoader.load_single_file()
    │       MarkItDown converts PDF -> Markdown text
    │       Returns Document with metadata
    │
    ├── 2. TextCleaner.batch_clean()
    │       Normalize whitespace, remove artifacts
    │
    ├── 3. MetadataExtractor.batch_extract()
    │       Add content hash, title, keywords, word count
    │
    ├── 4. AutoChunker.split()
    │       Markdown-aware chunking (640 chars, 96 overlap)
    │
    └── 5. ChromaVectorStore.add_documents()
            Ollama generates 768-dim embeddings
            Stored in ChromaDB (persistent)
```

### Query Flow (User Question)

```
User question (app.py chat input)
    │
    ▼
RAGChain.ask(query)
    │
    ├── 1. RetrievalPipeline.retrieve(query, k=5)
    │       │
    │       ├── HybridRetriever.retrieve()
    │       │       ├── BM25SparseRetriever.retrieve()   (keyword match)
    │       │       └── DenseRetriever.retrieve()         (vector similarity)
    │       │               │
    │       │               ▼
    │       │       RRF Fusion (merge + rerank by rank)
    │       │
    │       └── CrossEncoderReranker.rerank()
    │               Cross-encoder scores each (query, doc) pair
    │               Filter by threshold >= 0.70
    │               Return top-k documents
    │
    ├── 2. Format context from retrieved documents
    │       "[1] From: data/file.pdf\n{chunk content}\n\n---\n\n..."
    │
    ├── 3. LMManager.generate_with_context(query, context)
    │       System: "Answer based ONLY on the provided context..."
    │       User: "Context: {context}, Question: {query}"
    │       Ollama (llama3:8b) generates response
    │
    └── 4. Return { answer, sources, document_count, success }
            Displayed in Streamlit with expandable sources
```

---

## 8. Test Suite (`test/`)

| Script | What It Tests |
|---|---|
| `test_pipeline.py` | Full ingestion: Ollama connection, file discovery, end-to-end pipeline, chunk stats |
| `test_chat.py` | RAG system with test queries, interactive chat mode |
| `test_embeddings.py` | Embedding model connection, single query embedding, batch embedding, cosine similarity |
| `test_vectorstore.py` | ChromaDB collection stats, document addition, similarity search, scored search |
| `test_retrieval.py` | BM25 standalone, Dense standalone, Hybrid with RRF, Cross-encoder reranker, full pipeline |
| `test_performance.py` | Ingestion speed benchmark (chunks/second), timing, document/chunk counts |

Each test script can be run standalone: `python test/test_<name>.py`

---

## 9. Supported File Formats

| Extension | Loader | Notes |
|---|---|---|
| `.pdf` | MarkItDown | Converts to Markdown preserving structure |
| `.doc`, `.docx` | MarkItDown | Word documents to Markdown |
| `.ppt`, `.pptx` | MarkItDown | PowerPoint to Markdown |
| `.xlsx`, `.xls` | MarkItDown | Excel to Markdown |
| `.html` | MarkItDown | HTML to Markdown |
| `.txt` | TextLoader (LangChain) | Plain text |
| `.md` | UnstructuredMarkdownLoader | Markdown files |
| `.csv` | CSVLoader (LangChain) | CSV tabular data |
| `.json` | JSONLoader (LangChain) | JSON with jq schema `"."` |

---

## 10. Key Design Decisions

1. **Hybrid Retrieval**: Combines BM25 (keyword) + Dense (semantic) via RRF to handle both exact-match and meaning-based queries
2. **Cross-Encoder Reranking**: Uses `bge-reranker-v2-m3` for high-quality second-pass ranking with score threshold filtering
3. **Local-First**: All models (LLM, embeddings, reranker) run locally via Ollama/sentence-transformers - no API keys needed
4. **Markdown-Aware Chunking**: Documents from MarkItDown use `MarkdownTextSplitter` to preserve heading structure
5. **Persistent ChromaDB**: Vector store persists to disk at `./chroma_db/`, survives restarts
6. **Streamlit Caching**: `@st.cache_resource` ensures models load only once per session
7. **Fallback Reranking**: If cross-encoder fails to load, keyword-overlap scoring is used as backup
