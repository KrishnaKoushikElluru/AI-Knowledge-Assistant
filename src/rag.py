import os

from langchain_google_genai import ChatGoogleGenerativeAI

SYSTEM_PROMPT = (
    "Answer the question using only the context below. "
    "If the answer is not in the context, say you don't know.\n\n"
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
