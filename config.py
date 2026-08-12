import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("legixo-qa")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legixo-qa")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
CORPUS_DIR = os.getenv("CORPUS_DIR", "data")

EMBED_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-3.1-flash-lite"
EMBED_DIMENSION = 3072
MAX_STEPS = 3
TOP_K = 4