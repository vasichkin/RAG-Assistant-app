"""
Qdrant vector store helpers: build and load.
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config.settings import QDRANT_HOST, QDRANT_PORT

QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"


def build_qdrant(
    documents: List[Document],
    embeddings: Embeddings,
    collection_name: str = "ifc_rag",
) -> QdrantVectorStore:
    """Create a Qdrant collection and index documents into it."""
    return QdrantVectorStore.from_documents(
        documents,
        embeddings,
        url=QDRANT_URL,
        collection_name=collection_name,
    )


def load_qdrant(
    embeddings: Embeddings,
    collection_name: str = "ifc_rag",
) -> QdrantVectorStore:
    """Load an existing Qdrant collection."""
    return QdrantVectorStore(
        client=QdrantClient(url=QDRANT_URL),
        embedding=embeddings,
        collection_name=collection_name,
    )


# ---------------------------------------------------------------------------
# Phase 6: ColPali — separate Qdrant collection for page-image embeddings
# ---------------------------------------------------------------------------

COLPALI_COLLECTION = "ifc_colpali"


def build_colpali_qdrant(
    documents: List[Document],
    embeddings: Embeddings,
) -> QdrantVectorStore:
    """Build a separate Qdrant collection for ColPali page embeddings."""
    return build_qdrant(documents, embeddings, collection_name=COLPALI_COLLECTION)


def load_colpali_qdrant(
    embeddings: Embeddings,
) -> QdrantVectorStore:
    """Load the ColPali Qdrant collection."""
    return load_qdrant(embeddings, collection_name=COLPALI_COLLECTION)


def colpali_collection_info() -> dict | None:
    """Return {'points_count': int} if the ColPali collection exists, None if not.

    Raises on connection error so callers can show a meaningful message.
    """
    client = QdrantClient(url=QDRANT_URL)
    names = [c.name for c in client.get_collections().collections]
    if COLPALI_COLLECTION not in names:
        return None
    info = client.get_collection(COLPALI_COLLECTION)
    return {"points_count": info.points_count}
