import os

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorStore/db_faiss"



# LOAD PDF FILES
def load_pdf_files(data_path):
    print(" Loading PDFs...")

    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    print(f" Loaded {len(documents)} pages")

    if len(documents) == 0:
        raise ValueError(" No PDF files found in /data folder")

    return documents



# CREATE CHUNKS

def create_chunks(documents):
    print("Splitting text...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    return chunks

# EMBEDDINGS
def get_embedding_model():
    print(" Loading embedding model...")

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# MAIN FUNCTION

def create_vector_db():
    documents = load_pdf_files(DATA_PATH)
    chunks = create_chunks(documents)

    embedding_model = get_embedding_model()

    print("Creating FAISS index...")
    db = FAISS.from_documents(chunks, embedding_model)

    print("Saving DB...")

    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    db.save_local(DB_FAISS_PATH)

    print(" Vector DB created at", DB_FAISS_PATH)


if __name__ == "__main__":
    print("SCRIPT STARTED")
    create_vector_db()