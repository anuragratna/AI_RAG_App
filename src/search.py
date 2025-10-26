import os
from dotenv import load_dotenv
from src.vectordb import VectorDatabase
from src.data_loader import load_all_documents
from langchain_groq import ChatGroq  # type: ignore

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
class RAGSearch:
    def __init__(self, vector_store_dir: str = "vectordb_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.3-70b-versatile"):
        self.vector_db = VectorDatabase(persist_dir=vector_store_dir, embedding_model=embedding_model)
        vectordb_store_path = os.path.join(vector_store_dir, "chroma.sqlite3")
        meta_path = os.path.join(vector_store_dir, "metadata.pkl")
        print("[INFO] RAG Search initialized with vector database.")
        if not (os.path.exists(vectordb_store_path) and os.path.exists(meta_path)):
            docs = load_all_documents("data")
            self.vector_db.build_from_documents(docs)
        else:
            self.vector_db.load()
            print("[INFO] Vector database loaded from disk.")
        self.llm_model = llm_model
        # Prefer GROQ_API_KEY from environment; fall back to empty string
        # Read Groq API key from environment
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.llm = ChatGroq(groq_api_key=self.groq_api_key, model_name=self.llm_model)

    def search(self, query: str, top_k: int = 5, groq_api_key: str = None):
        # Use the vector database's query method which handles embedding the query
        results = self.vector_db.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            print("[INFO] No relevant documents found for the query.")
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}--]]]]]]]]]]]]"""
        print(f"[INFO] Generated prompt for LLM:\n{prompt}")
        
        response = self.llm.invoke([prompt])
        return response.content
     