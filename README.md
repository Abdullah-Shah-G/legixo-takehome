# Legixo Take-Home — Grounded Q&A API over Documents

A document Q&A **HTTP API** built with **Python + LangGraph + Pinecone**. Ask questions about a corpus of fictional legal-style notes; the system retrieves relevant chunks from Pinecone, grades whether they are good enough (branching the LangGraph), and writes a **grounded answer with citations**. If the documents cannot answer the question, it says so instead of hallucinating.

Built for the Legixo Thinklabs Gen AI intern take-home.

## Stack

- **Python 3.10+** (developed on 3.14)
- **FastAPI** — HTTP API (`POST /ask`, `POST /api/ingest`)
- **LangGraph** — `StateGraph` with 5 nodes and a conditional branch + max-steps guard
- **Pinecone** — serverless vector index (real service, official `pinecone` client)
- **Google Gemini** — `gemini-3.1-flash-lite` for grading/answering/rewriting, `gemini-embedding-001` (dim 768) for embeddings

## Project layout

```
.
├── main.py            # FastAPI app: POST /ask, POST /api/ingest, GET /, GET /health
├── graph.py           # LangGraph StateGraph (retrieve → grade → branch → answer/rewrite/not_found)
├── ingest.py          # Corpus → chunks → embeddings → Pinecone upsert
├── config.py          # env vars + constants
├── data/              # sample corpus (fictional legal notes, .md)
├── eval/self_test.json# 19 self-test questions with expected sources + pass/fail notes
├── docs/langgraph.md  # LangGraph map (nodes, branch, loop guard, diagram)
├── templates/index.html  # minimal web UI (optional; use curl if you prefer)
├── .env.example       # dummy values only
└── requirements.txt
```

## Setup

### 1. Get API keys

- **Gemini:** [aistudio.google.com](https://aistudio.google.com/apikey) → create API key
- **Pinecone (free):** [app.pinecone.io](https://app.pinecone.io) → sign up (Starter plan, no card) → API Keys → "Create API Key"

The current Pinecone SDK (`pinecone>=5`) uses **3072-dimension embeddings** for `gemini-embedding-001` — the index is auto-created with dimension 3072, so no manual configuration is needed.

### 2. Prepare environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

copy .env.example .env     # Windows
cp .env.example .env       # macOS/Linux
```

Edit `.env` and fill in your keys. Note your Pinecone region (shown in the Pinecone console — free tier defaults to e.g. `us-east-1` / `aws`) and set `PINECONE_REGION`/`PINECONE_CLOUD` to match.

### 3. Create the Pinecone index

The index (`legixo-qa`, dim **3072**, metric **cosine**, serverless) is created **automatically on first ingest** — no manual step needed.

### 4. Ingest the corpus (one-time)

```bash
python ingest.py                  # adds only new chunks (idempotent)
python ingest.py --reset          # wipe all vectors first, then re-ingest
```

**Running ingest twice:** chunk ids are deterministic (`<file>::chunk-<n>`), so re-running overwrites the same vectors in place — nothing duplicates. `--reset` deletes the whole namespace first if you want a clean slate.

## Start the API server

```bash
uvicorn main:app --reload --port 8000
# UI:    http://localhost:8000
# Docs:  http://localhost:8000/docs
```

## Example API calls

### Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What notice period applies when Bluecrest ends the employment agreement?"}'
```

Response:

```json
{
  "question": "What notice period applies when Bluecrest ends the employment agreement?",
  "answer": "Either party may end the agreement by giving **60 days** written notice. During notice, the employee must hand over all laptops, badges, and source code access. [1]",
  "citations": [
    { "chunk_id": "02_employment_agreement_excerpt.md::chunk-1",
      "source_file": "02_employment_agreement_excerpt.md" }
  ],
  "grounded": true,
  "steps": 1,
  "trace": ["retrieve(query=original, candidates=4)", "grade=pass (step 1)", "answer(citations=1)"]
}
```

### Ask something the corpus cannot answer

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the population of Riverside city?"}'
```

The grader fails the retrieved chunks, the question is rewritten and re-retrieved up to `MAX_STEPS` (3), then the graph routes to `not_found`:

```json
{
  "answer": "I could not find the answer in the document corpus. The documents do not contain enough information to answer this question.",
  "citations": [],
  "grounded": false,
  "steps": 3,
  "trace": ["retrieve(query=original, candidates=4)", "grade=fail (step 1)", "rewrite(query)", "retrieve(query=rewritten, candidates=4)", "grade=fail (step 2)", "rewrite(query)", "retrieve(query=rewritten, candidates=4)", "grade=fail (step 3)", "not_found"]
}
```

### Ingest via API (alternative to the CLI)

```bash
curl -X POST http://localhost:8000/api/ingest -H "Content-Type: application/json" -d '{"reset": false}'
```

### Health check

```bash
curl http://localhost:8000/health
```

## The LangGraph (summary)

```
START → retrieve → grade ──(pass)──► answer → END
                       └──(fail, steps<3)──► rewrite → retrieve
                       └──(fail, steps≥3)──► not_found → END
```

Full node-by-node map, state shape, and diagram: **`docs/langgraph.md`**.

## Self-test

`eval/self_test.json` contains **19 questions** (16 in-corpus + 3 out-of-corpus) from the assignment's gold set, each with expected source files and pass/fail notes.

**Result (2026-08-12, live run): 19/19 passed.**
- 16/16 in-corpus questions answered with the correct source file cited (`grounded: true`)
- 3/3 out-of-corpus questions refused without hallucination (`grounded: false`, empty citations, max steps reached)
- Re-ingest verified idempotent: re-running upserts the same deterministic chunk ids (overwrite, no duplicates)

## What I skipped / known limitations

Being honest about the rough edges (the brief said to say what's missing):

- **No LangSmith tracing** — graph runs are only observable through the `trace` array in the response
- **Dense-only retrieval** — no hybrid (BM25) search or reranker; retrieval can struggle with legal phrasing that barely overlaps the query
- **No streaming** — the answer endpoint returns the final JSON only
- **In-memory Pinecone namespace** — single shared namespace, no per-user or per-corpus isolation
- **Not a real legal system** — the corpus is fiction and answers are only as good as the retrieval; no citation formatting (e.g. pinpoint citations) beyond file-level source references
- **Embedding dimension assumption** — the index is created at 3072 dims to match what `gemini-embedding-001` currently returns; if the SDK/model default changes, the index needs recreation (`ingest.py --reset` after deleting the old index)

## Extras I'd add if I had more time

- LangSmith tracing for the graph runs
- Hybrid search (BM25 + dense) for better legal phrasing recall
- A reranker over the top-k candidates
- Streaming answers over SSE

## Notes

- All corpus documents are **fiction** (made-up parties/courts); no real client data is used.
- API keys live only in `.env` (gitignored). `.env.example` ships dummy values.