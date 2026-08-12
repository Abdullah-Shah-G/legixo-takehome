# LangGraph Map

Single `StateGraph` with 5 nodes and 1 conditional branch. Built in `graph.py`, compiled once at import.

```
 START
   │
   ▼
retrieve ──► grade ──► (conditional branch)
                 │
                 ├── grade == "pass" ──────────────► answer ──► END
                 ├── grade == "fail" AND steps < 3 ──► rewrite ──► retrieve (loop, retries)
                 └── grade == "fail" AND steps >= 3 ─► not_found ──► END
```

## Nodes

| Node | Function (async) | What it does |
|------|------------------|--------------|
| `retrieve` | `retrieve(state)` | Embeds the (possibly rewritten) query with `gemini-embedding-001`, queries the Pinecone index (`top_k=4`), returns candidate chunks with `chunk_id`, `source_file`, `text`, `score`. Increments `steps`. |
| `grade` | `grade(state)` | Sends question + retrieved context to the LLM and asks for a JSON `{"grade": "pass"/"fail", "reason"}`. Decides good path vs bad path. |
| `rewrite` | `rewrite(state)` | LLM rewrites the original question for better retrieval, then the graph loops back to `retrieve`. |
| `answer` | `answer(state)` | LLM writes a grounded answer using ONLY the retrieved context, with inline `[1]`–style citations. Collects deduplicated source files into `citations`. |
| `not_found` | `not_found(state)` | Returns "I could not find the answer in the document corpus" with `grounded: false` and empty citations. |

## Loop guard (max steps)

`QAState.steps` is incremented in `retrieve` and checked in the conditional router `route_after_grade()`:

- `pass` → `answer`
- `fail` with `steps < MAX_STEPS (3)` → `rewrite` (retry with a rewritten query)
- `fail` with `steps >= 3` → `not_found` — the graph **cannot spin forever**

## State shape

```python
class QAState(TypedDict, total=False):
    question: str
    query: str            # rewritten query after a fail (optional)
    chunks: list[dict]    # retrieved candidates
    grade: str            # "pass" | "fail"
    grade_reason: str
    steps: int            # retrieval attempts
    answer: str
    citations: list[dict]
    trace: list[str]      # per-node log, returned in the API response
    grounded: bool
```

## API response

```json
{
  "question": "...",
  "answer": "...",
  "citations": [{"chunk_id": "...", "source_file": "01_....md"}],
  "grounded": true,
  "steps": 1,
  "trace": ["retrieve(...)", "grade=pass (step 1)", "answer(citations=1)"]
}
```