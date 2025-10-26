from src.data_loader import load_all_documents
from src.embedding import EmbeddingGenerator
from src.vectordb import VectorDatabase
from src.search import RAGSearch    
def main():
#    docs = load_all_documents("data")
#    print(f"Loaded {len(docs)} documents.")
   
#    dbStore = VectorDatabase("vectordb_store")
#    dbStore.build_from_documents(docs)
#    dbStore.load()
#    print("Vector database loaded.")
   rag_search = RAGSearch(vector_store_dir="vectordb_store")
   query = "Total Holidays in ABC consulting in one line?"
   results = rag_search.search(query, top_k=3)
   print(f"Search results for query: '{query}' is:\n{results}")   


if __name__ == "__main__":
    main()
