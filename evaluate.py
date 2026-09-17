import shutil
from pathlib import Path

import pypdf
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.ingest import load_and_chunk_pdfs
from src.rag import SYSTEM_PROMPT, format_history, extract_sources

OLLAMA_LLM_MODEL = "llama3.2"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

BC_MODULE1_PAGE_LIMIT = 30

Path("vectorstore").mkdir(exist_ok=True)


def trim_pdf(source_path, max_pages, output_path):
    reader = pypdf.PdfReader(source_path)
    writer = pypdf.PdfWriter()
    for page in reader.pages[:max_pages]:
        writer.add_page(page)
    with open(output_path, "wb") as f:
        writer.write(f)
    return output_path


FILES = [
    (r"C:\Users\koush\Downloads\Lunorsoft Recruitments (ROUND 1).pdf", "Lunorsoft Recruitments (ROUND 1).pdf"),
    (r"C:\Users\koush\Downloads\Document Creation.pdf", "Document Creation.pdf"),
    (r"C:\Users\koush\Downloads\Bitcoin_Transaction_Forensics_Review_Report.pdf", "Bitcoin_Transaction_Forensics_Review_Report.pdf"),
    (trim_pdf(r"C:\Users\koush\Downloads\BC module1.pdf", BC_MODULE1_PAGE_LIMIT, "vectorstore/_bc_module1_trimmed.pdf"), "BC module1.pdf"),
]

TEST_CASES = [
    ("What is the submission deadline for the Lunorsoft Round 1 assignment?", "Lunorsoft Recruitments (ROUND 1).pdf", ["18 September", "12:00"]),
    ("What suggested technologies are listed for the AI Developer Option 1 assignment?", "Lunorsoft Recruitments (ROUND 1).pdf", ["LangChain", "FAISS"]),
    ("What is the core requirement for AI Developer Option 2?", "Lunorsoft Recruitments (ROUND 1).pdf", ["Fine-Tuning"]),
    ("What are the three rounds in the Lunorsoft recruitment process?", "Lunorsoft Recruitments (ROUND 1).pdf", ["Assignment", "Interview"]),
    ("What existing platform should UI/UX Designer candidates review?", "Lunorsoft Recruitments (ROUND 1).pdf", ["lunor.online"]),

    ("What does the S-F-V-C framework stand for?", "Document Creation.pdf", ["Search", "Verify", "Confirm"]),
    ("What accuracy did the AI attendance system achieve?", "Document Creation.pdf", ["92"]),
    ("What five things should a standard MoM template capture?", "Document Creation.pdf", ["discussed", "decided"]),
    ("What technologies were used to build the student attendance monitoring system?", "Document Creation.pdf", ["Python", "OpenCV"]),
    ("What does version v2.0 mean in the standard document versioning scheme?", "Document Creation.pdf", ["Major revision"]),

    ("How many transaction nodes are in the Elliptic++ dataset used in this project?", "Bitcoin_Transaction_Forensics_Review_Report.pdf", ["203,769", "203769"]),
    ("Who introduced the Elliptic++ dataset and in what year?", "Bitcoin_Transaction_Forensics_Review_Report.pdf", ["Elmougy", "2023"]),
    ("What evaluation metrics will be used given the class imbalance?", "Bitcoin_Transaction_Forensics_Review_Report.pdf", ["precision", "recall"]),
    ("What percentage of transactions in the dataset are illicit?", "Bitcoin_Transaction_Forensics_Review_Report.pdf", ["2.2"]),
    ("What dataset does this Bitcoin forensics project use?", "Bitcoin_Transaction_Forensics_Review_Report.pdf", ["Elliptic++"]),

    ("Who published the Bitcoin whitepaper and when?", "BC module1.pdf", ["Satoshi Nakamoto", "2008"]),
    ("What are the three properties in the CAP theorem?", "BC module1.pdf", ["Consistency", "Availability", "Partition"]),
    ("What is a satoshi?", "BC module1.pdf", ["smallest unit"]),
    ("What is Blockchain 2.0 associated with?", "BC module1.pdf", ["Smart Contract"]),
    ("What are the three types of network architecture mentioned for blockchain networks?", "BC module1.pdf", ["Centralized", "Decentralized", "Distributed"]),

    ("What is the capital of France?", None, ["not available"]),
]


def build_ollama_vectorstore(chunks):
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
    return FAISS.from_documents(chunks, embeddings)


def ollama_answer_question(vectorstore, question, history=None, k=4):
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join(doc.page_content for doc in docs)
    history_block = format_history(history)
    prompt = SYSTEM_PROMPT.format(history_block=history_block, context=context, question=question)
    llm = ChatOllama(model=OLLAMA_LLM_MODEL)
    response = llm.invoke(prompt)
    sources = extract_sources(docs)
    return response.content, sources


def process_documents():
    chunks = load_and_chunk_pdfs(FILES)
    store = build_ollama_vectorstore(chunks)
    return store, len(chunks)


def run_case(store, question, expected_file, expected_keywords):
    answer, sources = ollama_answer_question(store, question)
    source_files = {s["filename"] for s in sources}

    if expected_file is None:
        retrieval_hit = None
        answer_ok = "not available" in answer.lower()
    else:
        retrieval_hit = expected_file in source_files
        answer_ok = any(kw.lower() in answer.lower() for kw in expected_keywords)

    return {
        "question": question,
        "expected_file": expected_file,
        "retrieved_files": sorted(source_files),
        "retrieval_hit": retrieval_hit,
        "answer": answer,
        "answer_ok": answer_ok,
    }


def main():
    print(f"Using Ollama locally: {OLLAMA_LLM_MODEL} (generation), {OLLAMA_EMBED_MODEL} (embeddings)")
    print("Processing 4 PDFs into one combined index...")
    store, chunk_count = process_documents()
    print(f"Indexed {chunk_count} chunks from {len(FILES)} documents.\n")

    results = []
    for i, (question, expected_file, keywords) in enumerate(TEST_CASES):
        print(f"[{i + 1}/{len(TEST_CASES)}] {question}")
        result = run_case(store, question, expected_file, keywords)
        results.append(result)
        print(f"  retrieval_hit={result['retrieval_hit']} answer_ok={result['answer_ok']}")
        print(f"  answer: {result['answer'][:150]}")

    retrieval_cases = [r for r in results if r["retrieval_hit"] is not None]
    retrieval_hits = sum(1 for r in retrieval_cases if r["retrieval_hit"])
    answer_hits = sum(1 for r in results if r["answer_ok"])

    print("\n=== SUMMARY ===")
    print(f"Retrieval recall@4: {retrieval_hits}/{len(retrieval_cases)}")
    print(f"Answer correctness (keyword check): {answer_hits}/{len(results)}")

    write_report(results, chunk_count, retrieval_hits, len(retrieval_cases), answer_hits)


def write_report(results, chunk_count, retrieval_hits, retrieval_total, answer_hits):
    lines = [
        "# Retrieval Evaluation",
        "",
        f"Local evaluation run using Ollama ({OLLAMA_LLM_MODEL} generation, {OLLAMA_EMBED_MODEL} embeddings) — "
        "this script is for local evaluation only; the deployed app continues to use Gemini.",
        "",
        f"4 documents combined into one index, {chunk_count} total chunks. "
        f"{len(results)} test questions (5 per document + 1 deliberate out-of-scope question).",
        "",
        f"**Retrieval recall@4:** {retrieval_hits}/{retrieval_total} — correct source document present in the top-4 retrieved chunks.",
        f"**Answer correctness (keyword check):** {answer_hits}/{len(results)} — generated answer contains an expected fact.",
        "",
        "| # | Question | Expected source | Retrieved sources | Retrieval hit | Answer OK | Answer |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(results):
        expected = r["expected_file"] or "(out-of-scope)"
        retrieved = ", ".join(r["retrieved_files"]) or "(none)"
        hit = "-" if r["retrieval_hit"] is None else ("yes" if r["retrieval_hit"] else "NO")
        ok = "yes" if r["answer_ok"] else "NO"
        answer_preview = r["answer"].replace("\n", " ")[:120]
        lines.append(f"| {i + 1} | {r['question']} | {expected} | {retrieved} | {hit} | {ok} | {answer_preview} |")

    Path("EVALUATION.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nWrote EVALUATION.md")


if __name__ == "__main__":
    main()
