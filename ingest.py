from typing import TypedDict, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

import config


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBED_MODEL,
        api_key=config.GEMINI_API_KEY,
        dimension=config.EMBED_DIMENSION,
    )


def get_index():
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    if config.PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.EMBED_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
        config.logger.info("Created index %s", config.PINECONE_INDEX_NAME)
    return pc.Index(config.PINECONE_INDEX_NAME)


def load_markdown_files(corpus_dir: str | Path) -> list[dict]:
    path = Path(corpus_dir)
    files = sorted(path.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md files found in {corpus_dir}")
    docs = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        if text.strip():
            docs.append({"file": f.name, "text": text})
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=120, separators=["\n## ", "\n\n", "\n", " ", ""]
    )
    chunks = []
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append(
                {
                    "chunk_id": f"{doc['file']}::chunk-{i}",
                    "source_file": doc["file"],
                    "text": piece,
                }
            )
    return chunks


def ingest(corpus_dir: str | Path = config.CORPUS_DIR, reset: bool = False) -> dict:
    embeddings = get_embeddings()
    index = get_index()

    if reset:
        index.delete(delete_all=True, namespace="")
        config.logger.info("Wiped existing vectors")

    chunks = chunk_documents(load_markdown_files(corpus_dir))

    # chunk ids are deterministic (<file>::chunk-<n>), so running ingest twice
    # just overwrites the same vectors instead of duplicating them
    to_upsert = []
    for chunk in chunks:
        vector = embeddings.embed_query(chunk["text"])
        to_upsert.append(
            {
                "id": chunk["chunk_id"],
                "values": vector,
                "metadata": {
                    "source_file": chunk["source_file"],
                    "text": chunk["text"],
                },
            }
        )

    res = index.upsert(vectors=to_upsert, namespace="")
    upserted = res.get("upserted_count", len(to_upsert))

    return {
        "indexed": len(chunks),
        "upserted": upserted,
        "index": config.PINECONE_INDEX_NAME,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest corpus into Pinecone")
    parser.add_argument("--reset", action="store_true", help="Delete existing vectors first")
    args = parser.parse_args()
    print(ingest(reset=args.reset))