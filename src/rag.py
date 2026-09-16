import os
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI

SYSTEM_PROMPT = (
    "You are a document assistant. Answer the question using ONLY the context below.\n"
    "Do not use any outside knowledge. "
    "If the context does not contain the answer, respond exactly with: "
    "\"The answer is not available in the provided documents.\"\n"
    "Conversation history, if present, is only to help you understand what the current "
    "question refers to (e.g. pronouns like \"it\" or \"that\"). "
    "The answer itself must still come only from the context.\n\n"
    "{history_block}"
    "Context:\n{context}\n\nQuestion: {question}"
)


def format_history(history, max_turns: int = 3) -> str:
    if not history:
        return ""
    recent = history[-max_turns:]
    turns = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in recent)
    return f"Conversation history:\n{turns}\n\n"


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )


def extract_sources(docs):
    sources = []
    seen = set()
    for doc in docs:
        filename = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page_label") or doc.metadata.get("page", 0) + 1
        key = (filename, page)
        if key not in seen:
            seen.add(key)
            sources.append({"filename": filename, "page": page})
    return sources


def answer_question(vectorstore, question: str, history=None, k: int = 4):
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join(doc.page_content for doc in docs)
    history_block = format_history(history)
    prompt = SYSTEM_PROMPT.format(history_block=history_block, context=context, question=question)
    llm = get_llm()
    response = llm.invoke(prompt)
    sources = extract_sources(docs)
    return response.content, sources
