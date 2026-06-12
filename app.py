from rag import query_rag

while True:
    q = input("\nAsk a question: ")
    if q.lower() in ["exit", "quit"]:
        break

    answer = query_rag(q)
    print("\n--- ANSWER ---\n")
    print(answer)
