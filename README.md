# Mini AI Knowledge Assistant

A RAG (Retrieval-Augmented Generation) application that answers questions grounded in uploaded PDF documents, built for the Lunorsoft AI Developer Round 1 assignment (Option 1).

**Live demo:** https://ai-knowledge-assistant-njpexybza2h7vkg4z2h7dp.streamlit.app
**Repo:** https://github.com/KrishnaKoushikElluru/AI-Knowledge-Assistant

## Features

Core requirements:
- Upload one or more PDF documents
- Text extraction, chunking, embedding, and FAISS-based vector retrieval
- Answers generated only from retrieved document content — explicit refusal ("The answer is not available in the provided documents.") when the answer isn't in the corpus, instead of the model falling back on outside knowledge
- Simple Streamlit chat interface

Implemented beyond the core requirement:
- **Source citations** — every answer shows the originating filename and page number, taken directly from document metadata (never invented by the model)
- **Multiple documents** — any number of PDFs are combined into a single FAISS index in one processing pass, with per-chunk metadata tracking which file each chunk came from
- **Conversation history** — follow-up questions ("What about its cost?") are resolved using recent chat history, without any extra LLM call (see [Approach](#approach) below)
- **API-call minimization** — generation only fires on an explicit "Ask" submit (never on a Streamlit rerun), processing is skipped if the same document set was already indexed, and repeated questions are served from a session-local answer cache
- **Usage instrumentation** — a sidebar panel shows live embedding/generation call counts, chunk count, and cache-reuse status, for transparency into API usage
- **Graceful quota handling** — a 429/`RESOURCE_EXHAUSTED` error surfaces as a clean message instead of a crash
- **Deployment** — live on Streamlit Community Cloud

## Tech Stack

Python, LangChain, Google Gemini (`gemini-2.5-flash` for generation, `gemini-embedding-001` for embeddings), FAISS, Streamlit.

## Setup

1. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Add your Gemini API key to `secret.env` in the project root:
   ```
   GOOGLE_API_KEY=your_key_here
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```

## Project Structure

- `app.py` — Streamlit interface: upload, processing, chat, sidebar usage panel
- `src/ingest.py` — PDF loading and chunking (`RecursiveCharacterTextSplitter`, 1000-char chunks, 150-char overlap)
- `src/vectorstore.py` — Gemini embeddings and FAISS index build/load
- `src/rag.py` — Grounded retrieval + generation, source-citation extraction, conversation-history formatting
- `src/metrics.py` — Lightweight instrumentation wrapping the underlying Gemini SDK calls to count real API requests
- `data/` — Placeholder for source documents (uploads are processed from temp files at runtime, not stored here)

## Approach

**Pipeline:** PDF → `PyPDFLoader` → `RecursiveCharacterTextSplitter` → `gemini-embedding-001` embeddings → FAISS → similarity search (k=4) → grounded prompt → `gemini-2.5-flash` → answer + citations.

**Grounding:** the system prompt explicitly instructs the model to answer only from retrieved context and to return an exact refusal string when the context doesn't contain the answer. This was tested against both in-scope and deliberately out-of-scope questions to confirm the model doesn't fall back on its own knowledge.

**Conversation history without extra API calls:** the "standard" way to handle follow-up questions is a query-condensation step — an LLM call that rewrites a vague follow-up into a self-contained question before retrieval. That doubles the generation cost per question. Instead, history is folded directly into the *same* generation prompt (the last 3 turns, as plain `Q:/A:` text) so the model can resolve references like "it" when composing its answer, while retrieval itself still searches on the raw question text. This is a deliberate tradeoff: retrieval on a very vague follow-up (with no shared vocabulary with the original topic) can be less precise than it would be with a rewritten query, in exchange for keeping the call count fixed at exactly one generation call per question.

**Caching:** the answer cache is keyed on `(document set, prior conversation history, question)`, not just the question text — because the same question text can have a different correct answer depending on what was discussed before it (e.g., "What about its cost?" after two different preceding topics). A repeated question is only served from cache when the preceding context genuinely matches; a literal back-to-back repeat of the same question is also correctly cache-hit rather than falsely treated as a new context (this required excluding a question's own immediately-preceding occurrence from the cache key — otherwise every answered question would change its own future cache key).

**Batching:** Gemini's embedding endpoint caps requests at 100 texts. LangChain batches at that limit automatically — verified experimentally (76 chunks → 1 request, 140 chunks → 2 requests) rather than assumed. Processing 2 real-world PDFs can therefore cost 1 or 2 embedding requests depending on total chunk count; this is the API's own hard limit, not an inefficiency in the code.

## Known Limitations

- Follow-up-question retrieval can be imprecise for very vague references, as described above — a conscious tradeoff against extra API calls, not an oversight.
- The Gemini free tier used during development caps at 5 generation requests/minute; rapid-fire testing can hit this (handled gracefully in the UI, not a crash).
- No formal retrieval evaluation (e.g. precision/recall against a labeled question set) was built — testing was manual/targeted (in-scope vs. out-of-scope questions, multi-document source attribution, cache-correctness scenarios) rather than a systematic evaluation harness, given the assignment timeframe.

## AI Tools Used

This project was developed with Claude (Anthropic, Claude Code) as a pair-programming and implementation assistant.

The primary project direction, architecture, feature selection, design decisions, testing requirements, and trade-offs were determined and reviewed by me. Claude was used to assist with implementation, debugging, testing, and code refinement.

Claude was used for tasks including:
- Implementing the RAG pipeline based on the selected architecture
- Implementing the Streamlit interface
- Implementing caching and API-call minimization
- Adding API usage instrumentation
- Implementing graceful Gemini quota-error handling
- Debugging implementation issues
- Writing and running mocked tests
- Performing regression testing
- Assisting with code cleanup and incremental development

Key design decisions made during development included:
- Selecting the RAG-based approach for the assignment
- Choosing LangChain, Gemini, FAISS, and Streamlit
- Designing the grounded-answer and refusal behavior
- Preserving document metadata for source citations
- Supporting multiple PDF documents
- Designing conversation history without an additional history-summarization LLM call
- Prioritizing API-call minimization and caching
- Using explicit user submission to trigger generation
- Configuring environment loading with `load_dotenv(..., override=True)` to ensure the intended API-key configuration is used
- Testing the system against real documents and deliberate out-of-scope questions
- Validating API usage through instrumentation and mocked tests

I reviewed and tested the implemented changes throughout development and can explain the architecture, implementation, trade-offs, and design decisions during the next round.
