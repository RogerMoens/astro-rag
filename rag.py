from sentence_transformers import SentenceTransformer
import chromadb

# Load model once when the script starts
model = SentenceTransformer("all-MiniLM-L6-v2")


def load_db():
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_collection("scientific_rag")


def query_rag(question, k=100, max_distance=0.4):
    """
    Retrieve up to k chunks from Chroma and keep only those
    below the specified distance threshold.
    """

    collection = load_db()

    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "distances"]
    )

    contexts = []

    for doc, dist in zip(
        results["documents"][0],
        results["distances"][0]
    ):
        if dist <= max_distance:
            contexts.append(doc)

    # Fallback: return the best result if nothing passes threshold
    if not contexts and results["documents"][0]:
        contexts.append(results["documents"][0][0])

    return "\n\n".join(contexts)
