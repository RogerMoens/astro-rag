from sentence_transformers import SentenceTransformer
import chromadb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_NAME = "all-MiniLM-L6-v2"


def load_db():
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_collection("scientific_rag")


def analyze_query(
    query,
    n_results=100,
    output_file="distance_distribution.png"
):

    print(f"\nQuery: {query}\n")

    model = SentenceTransformer(MODEL_NAME)
    collection = load_db()

    embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["distances"]
    )

    distances = results["distances"][0]

    print(f"Retrieved {len(distances)} chunks")
    print(f"Min distance : {min(distances):.4f}")
    print(f"Max distance : {max(distances):.4f}")
    print(f"Mean distance: {sum(distances)/len(distances):.4f}")

    plt.figure(figsize=(12, 6))

    plt.plot(
        range(1, len(distances) + 1),
        distances,
        marker="o"
    )

    plt.xlabel("Rank")
    plt.ylabel("Distance")
    plt.title(f"Distance Distribution\n{query}")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nSaved graph to: {output_file}")


if __name__ == "__main__":

    query = input("Enter query: ").strip()

    analyze_query(query)
