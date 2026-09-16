import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.ingest import load_pdf, split_documents
from src.vectorstore import build_vectorstore
from src.rag import answer_question

load_dotenv("secret.env")

st.title("Mini AI Knowledge Assistant")
st.write("Upload a PDF and ask questions about it.")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None and st.button("Process PDF"):
    with st.spinner("Processing document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        documents = load_pdf(tmp_path)
        chunks = split_documents(documents)
        st.session_state.vectorstore = build_vectorstore(chunks, "vectorstore/faiss_index")
        Path(tmp_path).unlink()

    st.success(f"Processed {len(chunks)} chunks from the document.")

if st.session_state.vectorstore is not None:
    question = st.text_input("Ask a question about the document")
    if question:
        with st.spinner("Generating answer..."):
            answer, sources = answer_question(st.session_state.vectorstore, question)
        st.write(answer)

        st.markdown("---")
        st.subheader("Sources")
        for source in sources:
            st.write(f"{source['filename']} — page {source['page']}")
