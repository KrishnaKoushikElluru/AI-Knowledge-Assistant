# Lunorsoft AI Developer Round 1 — Mini AI Knowledge Assistant

RAG-based Q&A app over a PDF/document knowledge source. Built with LangChain, Gemini, and FAISS.

## Setup

1. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Add your Gemini API key to `secret.env`:
   ```
   GEMINI_API_KEY=your_key_here
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```

## Project Structure

- `src/ingest.py` — PDF loading and chunking
- `src/vectorstore.py` — Embeddings and FAISS index
- `src/rag.py` — Retrieval and answer generation
- `app.py` — Streamlit interface
- `data/` — Source documents

## Approach

_(to be filled in as the implementation progresses)_

## AI Tools Used

_(to be filled in — disclose tools and how they contributed)_
