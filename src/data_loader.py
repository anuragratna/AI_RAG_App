from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import JSONLoader, PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document

def load_all_documents(files_dir: str) -> List[Any]:
    """
    Load documents from a list of file paths using appropriate loaders based on file extensions and convert to LangChain document structure.
    """
    # Use project root data folder
    data_path = Path(files_dir).resolve()
    print(f"[Debug] Loading files from: {data_path}")
    all_documents = []

    #PDF files
    pdf_files = list(data_path.rglob("*.pdf"))
    print(f"[Debug] Found {len(pdf_files)} PDF files.")
    for file_path in pdf_files:
        print(f"[Debug] Loading PDF file: {file_path}")
        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            all_documents.extend(documents)
        except Exception as e:
            print(f"[Error] Failed to load PDF file {file_path}: {e}")
    
    #docx files
    docx_files = list(data_path.rglob("*.docx"))
    print(f"[Debug] Found {len(docx_files)} DOCX files.")
    for file_path in docx_files:
        print(f"[Debug] Loading DOCX file: {file_path}")
        try:
            loader = Docx2txtLoader(str(file_path))
            documents = loader.load()
            all_documents.extend(documents)
        except Exception as e:
            print(f"[Error] Failed to load DOCX file {file_path}: {e}")
    
    # JSON files
    json_files = list(data_path.rglob("*.json"))
    print(f"[Debug] Found {len(json_files)} JSON files.")
    for file_path in json_files:
        print(f"[Debug] Loading JSON file: {file_path}")
        try:
            # Simple fallback loader: read the JSON file and store its content as a Document.
            # The community JSONLoader requires `jq` and a jq schema, which may not be
            # available in all environments. This fallback avoids that dependency.
            text = Path(file_path).read_text(encoding="utf-8-sig")
            try:
                import json as _json

                parsed = _json.loads(text)
                # Represent dicts/lists as JSON strings for page_content
                if isinstance(parsed, (dict, list)):
                    content = _json.dumps(parsed)
                else:
                    content = str(parsed)
            except Exception:
                content = text
            doc = Document(page_content=content, metadata={"source": str(file_path)})
            all_documents.append(doc)
        except Exception as e:
            print(f"[Error] Failed to load JSON file {file_path}: {e}")

    return all_documents

# Example usage
if __name__ == "__main__":
    documents = load_all_documents("data")
    print(f"Total documents loaded: {len(documents)}")  
    print(f"Sample document content: {documents[0]}")  # Print first 200 characters of the first document
