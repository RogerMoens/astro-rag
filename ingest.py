## Load Files
import os
import fitz  # PyMuPDF

def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def load_all_docs(folder):
    docs = []

    for root, _, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)

            if file.endswith(".txt"):
                docs.append({"text": load_txt(path), "source": path})

            elif file.endswith(".pdf"):
                docs.append({"text": load_pdf(path), "source": path})

    return docs


## Chunk the documents
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = []

    for doc in docs:
        split_texts = splitter.split_text(doc["text"])
        for i, chunk in enumerate(split_texts):
            chunks.append({
                "text": chunk,
                "source": doc["source"]
            })

    return chunks


## Build vector database
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

def build_vector_db(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path="chroma_db")

    collection = client.get_or_create_collection(
        name="scientific_rag"
    )

    texts = [c["text"] for c in chunks]
    metas = [{"source": c["source"]} for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    embeddings = model.encode(texts).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metas,
        ids=ids
    )

    return collection

## Run Ingestion
if __name__ == "__main__":
    notes = load_all_docs("data/notes")
    papers = load_all_docs("data/papers")

    all_docs = notes + papers
    chunks = chunk_docs(all_docs)

    db = build_vector_db(chunks)

    print(f"Ingested {len(chunks)} chunks")
