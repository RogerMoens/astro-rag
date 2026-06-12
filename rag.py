from sentence_transformers import SentenceTransformer
import chromadb

def load_db():
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_collection("scientific_rag")


def query_rag(question, k=5):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    collection = load_db()

    query_embedding = model.encode([question])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    contexts = results["documents"][0]

    return "\n\n".join(contexts)
