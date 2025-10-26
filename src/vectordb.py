import os
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingGenerator

# Use chromadb as a faiss-free vector store backend
try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None
    Settings = None

class VectorDatabase:
    def __init__(self, persist_dir: str = "vectordb_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")
        # Chromadb client and collection (lazy init)
        self._chromadb_client = None
        self._collection = None
        self._collection_name = "default"

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingGenerator(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        # chunks is a list of strings
        metadatas = [{"text": chunk} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype("float32"), metadatas, documents=chunks)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def _ensure_chroma(self):
        if chromadb is None:
            raise ImportError("chromadb is not installed; please install chromadb or change the vector store implementation")
        if self._chromadb_client is None:
            # create a persistent local chroma client (duckdb+parquet is used by default for persistence)
            settings = Settings(is_persistent=True, persist_directory=self.persist_dir)
            self._chromadb_client = chromadb.Client(settings=settings)
        if self._collection is None:
            try:
                self._collection = self._chromadb_client.get_collection(self._collection_name)
            except Exception:
                self._collection = self._chromadb_client.create_collection(name=self._collection_name)

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None, documents: List[str] = None):
        # Use chromadb collection to store embeddings + metadata
        self._ensure_chroma()
        n = embeddings.shape[0]
        # generate ids
        ids = [str(len(self.metadata) + i) for i in range(n)]
        emb_list = embeddings.tolist()
        docs = documents if documents is not None else [None] * n
        metas = metadatas if metadatas is not None else [None] * n
        self._collection.add(ids=ids, embeddings=emb_list, metadatas=metas, documents=docs)
        self.metadata.extend(metas)
        print(f"[INFO] Added {n} vectors to Chroma collection.")

    def save(self):
        # Persist chroma client state (if using chromadb)
        if self._chromadb_client is not None:
            try:
                self._chromadb_client.persist()
                print(f"[INFO] Persisted Chroma database to {self.persist_dir}")
            except Exception as e:
                print(f"[WARN] Chroma persist failed: {e}")
        # still save metadata for backward compatibility
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved metadata to {self.persist_dir}")

    def load(self):
        # Load metadata
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        # initialize chroma collection
        try:
            self._ensure_chroma()
            print(f"[INFO] Loaded Chroma collection from {self.persist_dir}")
        except Exception as e:
            print(f"[WARN] Could not initialize Chroma collection: {e}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        # Query chroma collection
        self._ensure_chroma()
        q_emb = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        res = self._collection.query(query_embeddings=q_emb, n_results=top_k)
        results = []
        # res contains ids, distances, metadatas, documents
        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        for id_, dist, meta in zip(ids, distances, metadatas):
            results.append({"id": id_, "distance": dist, "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype("float32")
        return self.search(query_emb, top_k=top_k)