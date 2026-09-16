import shutil
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai._common import GoogleGenerativeAIError

from src.ingest import load_and_chunk_pdfs
from src.vectorstore import build_vectorstore
from src.rag import answer_question
from src import metrics

load_dotenv("secret.env", override=True)
metrics.instrument()

INDEX_PATH = Path("vectorstore/faiss_index")

st.title("Mini AI Knowledge Assistant")
st.write("Upload one or more PDFs and ask questions about them.")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "processed_fingerprint" not in st.session_state:
    st.session_state.processed_fingerprint = None
if "answer_cache" not in st.session_state:
    st.session_state.answer_cache = {}
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "vectorstore_reused" not in st.session_state:
    st.session_state.vectorstore_reused = None


def is_quota_error(exc) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "429" in text


uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files and st.button("Process PDFs"):
    fingerprint = tuple(sorted((f.name, f.size) for f in uploaded_files))
    already_processed = (
        fingerprint == st.session_state.processed_fingerprint
        and st.session_state.vectorstore is not None
    )

    if already_processed:
        st.session_state.vectorstore_reused = True
        st.info("These documents are already processed. Reusing the existing vector store.")
    else:
        temp_paths = []
        try:
            with st.spinner("Processing documents..."):
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
                st.session_state.processed_fingerprint = fingerprint
                st.session_state.processed_files = [f.name for f in uploaded_files]
                st.session_state.answer_cache = {}
                st.session_state.conversation_history = []
                st.session_state.chunk_count = len(chunks)
                st.session_state.vectorstore_reused = False

            st.success(f"Processed {len(chunks)} chunks from {len(uploaded_files)} document(s).")
        except GoogleGenerativeAIError as e:
            if is_quota_error(e):
                st.error("Gemini API quota exceeded while generating embeddings. Please wait and try again later.")
            else:
                st.error(f"Embedding generation failed: {e}")
        finally:
            for path in temp_paths:
                Path(path).unlink()

if st.session_state.vectorstore is not None:
    st.caption("Knowledge base: " + ", ".join(st.session_state.processed_files))

    with st.form("ask_form"):
        question = st.text_input("Ask a question about the documents")
        submitted = st.form_submit_button("Ask")

    if submitted and question:
        history_so_far = [(q, a) for q, a, _ in st.session_state.conversation_history]
        if history_so_far and history_so_far[-1][0] == question:
            history_for_key = history_so_far[:-1]
        else:
            history_for_key = history_so_far

        cache_key = (st.session_state.processed_fingerprint, tuple(history_for_key), question)

        answer = None
        if cache_key in st.session_state.answer_cache:
            answer, sources = st.session_state.answer_cache[cache_key]
        else:
            try:
                with st.spinner("Generating answer..."):
                    answer, sources = answer_question(
                        st.session_state.vectorstore, question, history=history_for_key
                    )
                st.session_state.answer_cache[cache_key] = (answer, sources)
            except GoogleGenerativeAIError as e:
                if is_quota_error(e):
                    st.error("Gemini API quota exceeded. Please wait and try again later.")
                else:
                    st.error(f"Answer generation failed: {e}")

        if answer is not None:
            last_turn = st.session_state.conversation_history[-1] if st.session_state.conversation_history else None
            if last_turn is None or last_turn[0] != question:
                st.session_state.conversation_history.append((question, answer, sources))

    for past_question, past_answer, past_sources in st.session_state.conversation_history:
        with st.chat_message("user"):
            st.write(past_question)
        with st.chat_message("assistant"):
            st.write(past_answer)
            st.caption("Sources: " + "; ".join(f"{s['filename']} — page {s['page']}" for s in past_sources))

with st.sidebar:
    st.subheader("Gemini usage (this session)")
    st.write(f"Embedding API calls: {metrics.counters['embedding_calls']}")
    st.write(f"Generation API calls: {metrics.counters['generation_calls']}")
    st.write(f"Chunks indexed: {st.session_state.chunk_count}")
    st.write(f"Last processing reused cache: {st.session_state.vectorstore_reused}")
