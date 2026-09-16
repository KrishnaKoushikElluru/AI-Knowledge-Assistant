import os
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.environ["GEMINI_API_KEY"],
    )


def build_vectorstore(chunks, index_path: str | Path):
    embeddings = get_embeddings()
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(index_path))
    return store


def load_vectorstore(index_path: str | Path):
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(index_path), embeddings, allow_dangerous_deserialization=True
    )
