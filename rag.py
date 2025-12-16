"""
Backend script to process and upload GST Council documents
Run this script to index your 110 PDFs before starting the Streamlit app
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
import os
import re
from typing import List
from pathlib import Path
from tqdm import tqdm

# Configuration
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
COLLECTION_NAME = "gst_council_documents"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
PERSIST_DIRECTORY = "./chroma_db"  # Persistent storage location

# Path to your documents
DOCUMENTS_PATH = r"C:\Users\ashutosh.bisht\Downloads\GST Documents"  # Place your 110 PDFs here


def initialize_backend_system():
    """Initialize ChromaDB with persistent storage"""
    print("🚀 Initializing RAG system...")
    
    # Create persistent ChromaDB client
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    # Initialize embedding model
    print(f"📥 Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    
    return chroma_client, embedder


def get_or_create_collection(client):
    """Get or create collection with persistent storage"""
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"✅ Found existing collection with {collection.count()} chunks")
        
        # Ask if user wants to clear and reindex
        response = input("Do you want to clear and reindex? (yes/no): ").lower()
        if response == 'yes':
            client.delete_collection(COLLECTION_NAME)
            collection = client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print("🗑️ Collection cleared. Ready for fresh indexing.")
    except:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print("📦 Created new collection")
    
    return collection


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return ""


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return ""


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return ""


def preprocess_text(text: str) -> str:
    """Clean and preprocess text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\[\]\{\}]', '', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.split()) > 50:
            chunks.append(chunk)
    
    return chunks


def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text"""
    words = text.lower().split()
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that'}
    
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return keywords


def process_documents_backend(collection, embedder, documents_path: str):
    """Process all documents in the specified directory"""
    
    # Get all supported files
    supported_extensions = ['.pdf', '.docx', '.txt']
    files = []
    
    for ext in supported_extensions:
        files.extend(Path(documents_path).glob(f"**/*{ext}"))
    
    if not files:
        print(f"❌ No documents found in {documents_path}")
        print(f"   Please place your PDFs in the '{documents_path}' folder")
        return 0
    
    print(f"\n📚 Found {len(files)} documents to process\n")
    
    total_chunks = 0
    
    # Process each file with progress bar
    for file_path in tqdm(files, desc="Processing documents", unit="file"):
        
        file_name = file_path.name
        
        # Extract text based on file type
        if file_path.suffix == '.pdf':
            text = extract_text_from_pdf(str(file_path))
        elif file_path.suffix == '.docx':
            text = extract_text_from_docx(str(file_path))
        elif file_path.suffix == '.txt':
            text = extract_text_from_txt(str(file_path))
        else:
            continue
        
        if not text or len(text) < 100:
            print(f"⚠️  Skipping {file_name} - insufficient text")
            continue
        
        # Preprocess and chunk
        cleaned_text = preprocess_text(text)
        chunks = chunk_text(cleaned_text)
        
        if not chunks:
            print(f"⚠️  Skipping {file_name} - no valid chunks")
            continue
        
        # Generate embeddings in batches
        print(f"   Processing {file_name}: {len(chunks)} chunks")
        embeddings = embedder.encode(chunks, show_progress_bar=False, batch_size=32).tolist()
        
        # Extract keywords
        chunk_keywords = [extract_keywords(chunk) for chunk in chunks]
        
        # Prepare data for ChromaDB
        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": file_name,
                "chunk_id": i,
                "keywords": " ".join(kw[:20]),
                "file_path": str(file_path)
            }
            for i, kw in enumerate(chunk_keywords)
        ]
        
        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        
        total_chunks += len(chunks)
    
    return total_chunks


def main():
    """Main backend processing function"""
    print("=" * 80)
    print("GST COUNCIL RAG SYSTEM - BACKEND DOCUMENT PROCESSOR")
    print("=" * 80)
    
    # Check if documents directory exists
    if not os.path.exists(DOCUMENTS_PATH):
        print(f"\n📁 Creating documents directory: {DOCUMENTS_PATH}")
        os.makedirs(DOCUMENTS_PATH)
        print(f"   Please place your 110 GST Council PDFs in this folder and run again.")
        return
    
    # Initialize system
    chroma_client, embedder = initialize_backend_system()
    collection = get_or_create_collection(chroma_client)
    
    # Process documents
    print(f"\n📂 Processing documents from: {DOCUMENTS_PATH}\n")
    total_chunks = process_documents_backend(collection, embedder, DOCUMENTS_PATH)
    
    print("\n" + "=" * 80)
    print(f"✅ SUCCESS! Indexed {total_chunks} chunks from documents")
    print(f"📊 Total documents in collection: {collection.count()}")
    print(f"💾 Data persisted to: {PERSIST_DIRECTORY}")
    print("=" * 80)
    print("\n🚀 You can now run the Streamlit app: streamlit run app.py\n")


if __name__ == "__main__":
    main()