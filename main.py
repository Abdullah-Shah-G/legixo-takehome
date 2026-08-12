from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import config
from graph import ask
from ingest import ingest

app = FastAPI(title="Legixo Take-Home Q&A API")
TEMPLATES_DIR = Path("templates")


def render(name: str) -> str:
    return (TEMPLATES_DIR / name).read_text(encoding="utf-8")


class AskRequest(BaseModel):
    question: str


class IngestRequest(BaseModel):
    reset: bool = False


@app.get("/")
async def index():
    return HTMLResponse(render("index.html"))


@app.get("/health")
async def health():
    return JSONResponse(
        {
            "status": "ok",
            "gemini_key_set": bool(config.GEMINI_API_KEY),
            "pinecone_key_set": bool(config.PINECONE_API_KEY),
            "index": config.PINECONE_INDEX_NAME,
        }
    )


@app.post("/api/ingest")
async def api_ingest(req: IngestRequest):
    try:
        result = ingest(reset=req.reset)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/ask")
async def ask_question(req: AskRequest):
    if not req.question.strip():
        return JSONResponse({"error": "question must not be empty"}, status_code=400)
    try:
        return JSONResponse(await ask(req.question))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)