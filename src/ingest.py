from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(path: str | Path, display_name: str | None = None):
    loader = PyPDFLoader(str(path))
    documents = loader.load()
    if display_name:
        for doc in documents:
            doc.metadata["source"] = display_name
    return documents


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def load_and_chunk_pdfs(files, chunk_size: int = 1000, chunk_overlap: int = 150):
    all_chunks = []
    for path, display_name in files:
        documents = load_pdf(path, display_name=display_name)
        all_chunks.extend(split_documents(documents, chunk_size, chunk_overlap))
    return all_chunks
