import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_data(pdf_path: str):
    print(f"Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages from PDF.")

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    print(f"Created {len(docs)} chunks.")

    print("Initializing Embeddings (Ollama Qwen)...")
    embeddings = OllamaEmbeddings(model="qwen")

    print("Creating FAISS vector store...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save to local directory
    save_path = "faiss_index"
    vectorstore.save_local(save_path)
    print(f"FAISS index saved to '{save_path}/' directory.")
    print("Ingestion complete!")

if __name__ == "__main__":
    pdf_file = "TechStore_FAQ_Knowledge_Base.pdf"
    if os.path.exists(pdf_file):
        ingest_data(pdf_file)
    else:
        print(f"Error: {pdf_file} not found in the current directory.")
