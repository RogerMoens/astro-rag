from datetime import datetime
from rag import query_rag


def export_context(query: str, output_file: str = "rag_context.txt"):
    """
    Retrieve context from the RAG system and export it to a text file.
    """

    context = query_rag(query)

    prompt_instructions = """
SYSTEM INSTRUCTIONS

The following context was retrieved from a Retrieval-Augmented Generation (RAG) system.

Important notes:
- The retrieved context may be incomplete.
- The retrieved context may contain errors, outdated information, or irrelevant passages.
- Do not assume the context is fully correct.
- Use the context as supporting evidence rather than absolute truth.
- If information is missing, acknowledge uncertainty.
- Prefer conclusions that are directly supported by the context.
- When multiple sources disagree, mention the disagreement.
- If answering scientific questions, distinguish clearly between established facts and hypotheses.
- Cite source references when available.
- Mathematical expressions should be written using LaTeX notation.
- Do not fabricate references that are not present in the context.

END OF INSTRUCTIONS
"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content = f"""
==================================================
RAG CONTEXT EXPORT
Generated: {timestamp}
==================================================

Query:
{query}

{prompt_instructions}

==================================================
RETRIEVED CONTEXT
==================================================

{context}
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Context exported to: {output_file}")


if __name__ == "__main__":
    query = input("Enter search term or question: ").strip()

    if not query:
        print("No query provided.")
    else:
        export_context(query)
