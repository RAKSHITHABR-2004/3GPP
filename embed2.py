import json
from pathlib import Path
from uuid import uuid4
import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.text_splitter import SentenceSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

file_path_str = filedialog.askopenfilename(
    title="Select a PDF, DOCX, or HTML file",
    filetypes=[("Supported Files", "*.pdf *.docx *.html")]
)

if not file_path_str:
    print("No file selected. Exiting.")
    exit()

file_path = Path(file_path_str)

def load_document(path: Path):
    return SimpleDirectoryReader(input_files=[path]).load_data()


def chunk_documents(documents, chunk_size=512, chunk_overlap=50):
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in documents:
        split_texts = splitter.split_text(doc.text)
        for chunk_text in split_texts:
            chunks.append({
                "id": str(uuid4()),
                "text": chunk_text
            })
    return chunks


def get_embeddings_from_chunks(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    texts = [chunk["text"] for chunk in chunks]
    embed_model = HuggingFaceEmbeddings(model_name=model_name)
    embeddings = embed_model.embed_documents(texts)

    return [
        {
            "id": chunks[i]["id"],
            "text": chunks[i]["text"],
            "embedding": embeddings[i]
        }
        for i in range(len(chunks))
    ]


def main():
    print(f"Loading document: {file_path.name}")
    document = load_document(file_path)

    print("Chunking text...")
    chunks = chunk_documents(document)

    print("Generating embeddings...")
    version_embeddings = get_embeddings_from_chunks(chunks)

    release_name = file_path.stem
    output_path = Path(f"{release_name}_embeddings.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(version_embeddings, f, ensure_ascii=False, indent=2)

    print(f"Embeddings saved to {output_path}")


    from chromadb import PersistentClient
    CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(os.path.dirname(__file__), "../chroma_storage"))
    print("Using ChromaDB path:", CHROMA_PATH)
    client = PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=release_name)
    collection.add(
        documents=[item["text"] for item in version_embeddings],
        embeddings=[item["embedding"] for item in version_embeddings],
        ids=[item["id"] for item in version_embeddings],
    )
    print(f"Stored {len(version_embeddings)} embeddings in ChromaDB under collection: {release_name}")


if __name__ == "__main__":
    main()