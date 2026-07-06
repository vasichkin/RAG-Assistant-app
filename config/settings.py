"""
Central configuration: loads env vars and initialises the Vertex AI client.

Usage:
    from config.settings import client, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION
    from config.settings import LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_HOST
    from config.settings import QDRANT_HOST, QDRANT_PORT
"""

import os
from dotenv import load_dotenv
from google import genai
import vertexai

load_dotenv()

# --- Google Cloud / Vertex AI ---
GOOGLE_CLOUD_PROJECT: str = os.environ["GOOGLE_CLOUD_PROJECT"]
GOOGLE_CLOUD_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# Vertex AI client — ADC credentials via `gcloud auth application-default login`.
# Never pass api_key here.
client = genai.Client(
    vertexai=True,
    project=GOOGLE_CLOUD_PROJECT,
    location=GOOGLE_CLOUD_LOCATION,
)

# Required by vertexai.vision_models (MultiModalEmbeddingModel).
vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)

# Model identifiers
EMBEDDING_MODEL: str = "text-embedding-005"
GENERATION_MODEL: str = "gemini-2.5-flash"

# --- Langfuse ---
LANGFUSE_SECRET_KEY: str = os.environ["LANGFUSE_SECRET_KEY"]
LANGFUSE_PUBLIC_KEY: str = os.environ["LANGFUSE_PUBLIC_KEY"]
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# --- Qdrant ---
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))

# --- Chunking ---
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 64

# --- Retrieval ---
RETRIEVAL_TOP_K: int = 5

# --- Vector store paths ---
FAISS_INDEX_PATH: str = "data/faiss_index"
