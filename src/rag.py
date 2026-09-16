import os

from langchain_google_genai import ChatGoogleGenerativeAI

SYSTEM_PROMPT = (
    "You are a document assistant. Answer the question using ONLY the context below.\n"
    "Do not use any outside knowledge. "
    "If the context does not contain the answer, respond exactly with: "
    "\"The answer is not available in the provided documents.\"\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ["GEMINI_API_KEY"],
    )


def answer_question(vectorstore, question: str, k: int = 4):
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = SYSTEM_PROMPT.format(context=context, question=question)
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content, docs
