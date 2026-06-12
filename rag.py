from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

def load_db():
    embeddings = OpenAIEmbeddings()

    db = Chroma(
        collection_name="astro_rag",
        embedding_function=embeddings,
        persist_directory="chroma_db"
    )
    return db


def query_rag(question, k=5):
    db = load_db()

    results = db.similarity_search(question, k=k)

    context = "\n\n".join(
        [f"Source: {r.metadata['source']}\n{r.page_content}" for r in results]
    )

    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = f"""
You are a scientific writing assistant.

Use the context below to answer the question accurately.
If you don't know, say so.

CONTEXT:
{context}

QUESTION:
{question}

Answer in a structured academic style.
"""

    return llm.invoke(prompt).content
