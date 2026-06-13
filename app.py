import streamlit as st
import ollama

from rag import query_rag

st.set_page_config(
    page_title="Local RAG Chat",
    layout="wide"
)

st.title("💬 Local RAG Assistant")

# Get installed Ollama models
available_models = [
    model.model
    for model in ollama.list().models
]

selected_model = st.sidebar.selectbox(
    "Choose Model",
    available_models
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question...")

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve context from Chroma
    context = query_rag(prompt)

    rag_prompt = f"""
You are a scientific assistant.

Use the supplied context to answer the question.

If the answer is not contained in the context,
say that the information was not found.

Context:
{context}

Question:
{prompt}
"""

    response = ollama.chat(
        model=selected_model,
        messages=[
            {
                "role": "user",
                "content": rag_prompt
            }
        ]
    )

    answer = response["message"]["content"]

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
