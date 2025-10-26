from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 200, chunk_overlap: int = 50):
        self.model = SentenceTransformer(model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def chunk_documents(self, documents: List[Any]) -> List[str]:
        all_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)
            # `split_text` returns a list of strings
            all_chunks.extend(chunks)
        return all_chunks
    
    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        # `chunks` is a list of plain strings returned by the text splitter.
        texts = list(chunks)
        print(f"[Debug] Embedding {len(chunks)} chunks using model {self.model.__class__.__name__}")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[Debug] Generated embeddings for {len(chunks)} chunks.")
        print(f"[Debug] Embeddings shape: {embeddings.shape}")
        print(f"[Debug] Sample embedding (first chunk): {embeddings[0][:5]}...")  # Print first 5 dimensions of the first embedding
        return embeddings
    

if __name__ == "__main__":
    # Example usage
    files_dir = "data"
    documents = load_all_documents(files_dir)
    print(f"Loaded {len(documents)} documents.")

    embedder = EmbeddingGenerator()
    chunks = embedder.chunk_documents(documents)
    print(f"Generated {len(chunks)} text chunks.")

    embeddings = embedder.embed_chunks(chunks)
    print(f"Generated embeddings with shape: {embeddings.shape}")