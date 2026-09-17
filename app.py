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

st.set_page_config(page_title="Mini AI Knowledge Assistant", page_icon="📚", layout="centered")

st.markdown(
    """
    <style>
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    div[data-testid="stFileUploaderDropzone"] {
        border-radius: 10px;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 10px;
    }
    h1 { font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 Mini AI Knowledge Assistant")
st.caption("Ask questions about your PDFs — every answer is grounded strictly in the documents you upload, with sources cited.")

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


st.subheader("1. Knowledge base")
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True, label_visibility="collapsed")

if uploaded_files and st.button("Process PDFs", type="primary", use_container_width=True):
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
    st.success(f"📁 Knowledge base ready — {', '.join(st.session_state.processed_files)}")
else:
    st.info("Upload one or more PDFs to build your knowledge base.")

if st.session_state.vectorstore is not None:
    st.divider()
    st.subheader("2. Ask a question")

    with st.form("ask_form"):
        question = st.text_input(
            "Your question",
            placeholder="e.g. What is the submission deadline?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask →", type="primary", use_container_width=True)

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

    if st.session_state.conversation_history:
        st.divider()
        st.subheader("Conversation")
        for past_question, past_answer, past_sources in reversed(st.session_state.conversation_history):
            with st.container(border=True):
                with st.chat_message("user"):
                    st.write(past_question)
                with st.chat_message("assistant"):
                    st.write(past_answer)
                    if past_sources:
                        st.caption(
                            "📎 Sources: " + "; ".join(f"{s['filename']} — page {s['page']}" for s in past_sources)
                        )

with st.sidebar:
    st.subheader("Session usage")
    col1, col2 = st.columns(2)
    col1.metric("Embedding calls", metrics.counters["embedding_calls"])
    col2.metric("Generation calls", metrics.counters["generation_calls"])
    st.divider()
    st.metric("Chunks indexed", st.session_state.chunk_count)
    st.caption(f"Last processing reused cache: {st.session_state.vectorstore_reused}")
