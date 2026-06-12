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
