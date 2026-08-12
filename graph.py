import os
import json
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

import config
from ingest import get_index, get_embeddings


class QAState(TypedDict, total=False):
    question: str
    query: str
    chunks: list[dict]
    grade: str
    grade_reason: str
    steps: int
    answer: str
    citations: list[dict]
    trace: list[str]
    grounded: bool


def _llm():
    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        api_key=config.GEMINI_API_KEY,
        max_output_tokens=1024,
        temperature=0.1,
    )


def _clean(content) -> str:
    if isinstance(content, list):
        return "".join(c.get("text", "") for c in content).strip()
    return str(content).strip()


async def retrieve(state: QAState) -> QAState:
    query = state.get("query") or state["question"]
    index = get_index()
    embeddings = get_embeddings()
    vector = embeddings.embed_query(query)

    res = index.query(
        vector=vector,
        top_k=config.TOP_K,
        include_metadata=True,
        namespace="",
    )

    chunks = []
    for match in res.get("matches", []):
        md = match.get("metadata", {})
        chunks.append(
            {
                "chunk_id": match["id"],
                "source_file": md.get("source_file", "unknown"),
                "text": md.get("text", ""),
                "score": match.get("score", 0),
            }
        )

    trace = state.get("trace", [])
    trace.append(f"retrieve(query={'query' if state.get('query') else 'original'}, candidates={len(chunks)})")
    return {
        "chunks": chunks,
        "steps": state.get("steps", 0) + 1,
        "trace": trace,
    }


async def grade(state: QAState) -> QAState:
    question = state["question"]
    chunks = state.get("chunks", [])
    context = "\n\n---\n\n".join(
        f"[{i+1}] ({c['source_file']})\n{c['text']}" for i, c in enumerate(chunks)
    )

    prompt = f"""You are a retrieval grader. Decide whether the retrieved context below is good enough to answer the question.

Question: {question}

Context:
{context if context else "(no context retrieved)"}

Respond with a single JSON object: {{"grade": "pass" or "fail", "reason": "one short sentence"}}.
- "pass" means the context contains facts that answer the question.
- "fail" means the context is empty, irrelevant, or insufficient (e.g. the question asks something the documents never mention)."""

    llm = _llm()
    result = await llm.ainvoke([{"role": "user", "content": prompt}])
    raw = _clean(result.content)
    grade = "fail"
    reason = "parse error"
    try:
        # LLMs sometimes wrap the JSON in markdown — grab the first {...} block
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        grade = parsed.get("grade", "fail")
        reason = parsed.get("reason", "")
    except Exception:
        if "pass" in raw.lower():
            grade = "pass"
            reason = raw[:120]

    trace = state.get("trace", [])
    trace.append(f"grade={grade} (step {state.get('steps', 0)})")
    return {"grade": grade, "grade_reason": reason, "trace": trace}


async def rewrite(state: QAState) -> QAState:
    llm = _llm()
    prompt = f"""Rewrite the question below so a vector search over a legal-document corpus is more likely to find relevant passages. Return ONLY the rewritten question, no explanation.

Original: {state['question']}"""
    result = await llm.ainvoke([{"role": "user", "content": prompt}])
    query = _clean(result.content) or state["question"]

    trace = state.get("trace", [])
    trace.append("rewrite(query)")
    return {"query": query, "trace": trace}


async def answer(state: QAState) -> QAState:
    question = state["question"]
    chunks = state.get("chunks", [])
    context = "\n\n---\n\n".join(
        f"[{i+1}] ({c['source_file']})\n{c['text']}" for i, c in enumerate(chunks)
    )

    llm = _llm()
    system = """You are a grounded legal research assistant. Answer ONLY from the provided context.
- Cite your sources inline like [1], [2] matching the numbered passages.
- If a fact is not in the context, do not invent it.
- If the context cannot answer, say so explicitly.
- Keep the answer concise and factual."""
    user = f"Question: {question}\n\nContext:\n{context}"

    result = await llm.ainvoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    text = _clean(result.content)

    seen = {}
    for c in chunks:
        seen.setdefault(c["source_file"], {"chunk_id": c["chunk_id"], "source_file": c["source_file"]})

    trace = state.get("trace", [])
    trace.append(f"answer(citations={len(seen)})")
    return {
        "answer": text,
        "citations": list(seen.values()),
        "grounded": True,
        "trace": trace,
    }


async def not_found(state: QAState) -> QAState:
    trace = state.get("trace", [])
    trace.append("not_found")
    return {
        "answer": (
            "I could not find the answer in the document corpus. The documents do not "
            "contain enough information to answer this question."
        ),
        "citations": [],
        "grounded": False,
        "trace": trace,
    }


def route_after_grade(state: QAState) -> str:
    if state.get("grade") == "pass":
        return "answer"
    # hard stop after MAX_STEPS so the graph can never loop forever
    if state.get("steps", 0) >= config.MAX_STEPS:
        return "not_found"
    return "rewrite"


def build_graph():
    graph = StateGraph(QAState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("answer", answer)
    graph.add_node("not_found", not_found)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"answer": "answer", "rewrite": "rewrite", "not_found": "not_found"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("answer", END)
    graph.add_edge("not_found", END)

    return graph.compile()


graph = build_graph()


async def ask(question: str) -> dict:
    if not question.strip():
        return {"error": "question must not be empty"}

    result = await graph.ainvoke(
        {
            "question": question,
            "chunks": [],
            "grade": "fail",
            "steps": 0,
            "answer": "",
            "citations": [],
            "trace": [],
            "grounded": False,
        }
    )

    return {
        "question": question,
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "grounded": result.get("grounded", False),
        "steps": result.get("steps", 0),
        "trace": result.get("trace", []),
    }