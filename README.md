


## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline using modern AI/ML tools. It is designed to process various document types (JSON, PDF, DOCX, etc.), chunk the content, convert it into a unified document format, generate embeddings, and store them in a vector database (ChromaDB) for efficient semantic search and retrieval.

### Key Features
- **Multi-format File Loading:** Supports JSON, PDF, DOCX, and more via flexible loaders.
- **Document Chunking:** Splits large documents into manageable text chunks for better embedding and retrieval.
- **Document Conversion:** Converts raw file content into a standardized document format for downstream processing.
- **Embeddings Generation:** Uses SentenceTransformer models to create high-quality vector representations of text chunks.
- **Vector Database Storage:** Stores embeddings in ChromaDB for fast similarity search and retrieval.
- **RAG Search:** Integrates with Groq LLM via LangChain for advanced question answering and summarization over retrieved context.

### Workflow
1. **Load Documents:** All supported files in the `data/` directory are loaded and parsed.
2. **Chunk Documents:** Each document is split into smaller text chunks using recursive character splitting.
3. **Convert to Document Format:** Chunks are wrapped in a document object for embedding.
4. **Embed Chunks:** Each chunk is embedded using a SentenceTransformer model.
5. **Store in ChromaDB:** Embeddings and metadata are stored in a persistent ChromaDB vector store.
6. **Semantic Search & RAG:** Queries are answered by retrieving relevant chunks and generating LLM-based summaries.

### Technologies Used
- Python 3.12
- LangChain
- SentenceTransformers
- ChromaDB
- Groq LLM
- dotenv

### Usage
1. Place your files (PDF, DOCX, JSON, etc.) in the `data/` directory.
2. Set your Groq API key in the `.env` file: `GROQ_API_KEY=your_actual_groq_api_key_here`
3. Run `main.py` to build the vector database and start searching.

---

# Tags: RAG, AI, ML, Chunking, Groq, LangChain, ChromaDB, Embeddings, Retrieval-Augmented Generation

