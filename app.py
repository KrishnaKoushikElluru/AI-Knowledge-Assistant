import shutil
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.ingest import load_and_chunk_pdfs
from src.vectorstore import build_vectorstore
from src.rag import answer_question

load_dotenv("secret.env")

INDEX_PATH = Path("vectorstore/faiss_index")

st.title("Mini AI Knowledge Assistant")
st.write("Upload one or more PDFs and ask questions about them.")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files and st.button("Process PDFs"):
    with st.spinner("Processing documents..."):
        temp_paths = []
        files_for_ingest = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                temp_paths.append(tmp.name)
                files_for_ingest.append((tmp.name, uploaded_file.name))

        chunks = load_and_chunk_pdfs(files_for_ingest)

        if INDEX_PATH.exists():
            shutil.rmtree(INDEX_PATH)
        st.session_state.vectorstore = build_vectorstore(chunks, INDEX_PATH)

        for path in temp_paths:
            Path(path).unlink()

    st.session_state.processed_files = [f.name for f in uploaded_files]
    st.success(f"Processed {len(chunks)} chunks from {len(uploaded_files)} document(s).")

if st.session_state.vectorstore is not None:
    st.caption("Knowledge base: " + ", ".join(st.session_state.processed_files))
    question = st.text_input("Ask a question about the documents")
    if question:
        with st.spinner("Generating answer..."):
            answer, sources = answer_question(st.session_state.vectorstore, question)
        st.write(answer)

        st.markdown("---")
        st.subheader("Sources")
        for source in sources:
            st.write(f"{source['filename']} — page {source['page']}")
