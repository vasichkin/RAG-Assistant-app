"""
ingestion/chunker.py — Split text Documents into chunks; pass through
image and table Documents unchanged.

Uses RecursiveCharacterTextSplitter with CHUNK_SIZE / CHUNK_OVERLAP from
config.settings.
"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Chunk text Documents; pass tables and images through as single chunks.

    For content_type="text": the document is split into one or more smaller
    Documents using RecursiveCharacterTextSplitter.  All original metadata
    fields are preserved and a ``chunk_index`` (0-based) is added.

    For content_type="table" or "image": the Document is passed through
    unchanged (treated as a single chunk) with ``chunk_index`` set to 0.

    Args:
        documents: List of Documents produced by the ingestion modules.

    Returns:
        List of chunked / passed-through Documents, each with a ``chunk_index``
        key added to its metadata.
    """
    result: List[Document] = []

    for doc in documents:
        content_type = doc.metadata.get("content_type", "text")

        if content_type == "text":
            chunks = _splitter.split_documents([doc])
            for idx, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = idx
            result.extend(chunks)
            logger.debug(
                "Split page %s into %d chunks.",
                doc.metadata.get("page_number"),
                len(chunks),
            )
        else:
            # Tables and images are single-chunk pass-throughs
            doc.metadata["chunk_index"] = 0
            result.append(doc)

    logger.info(
        "chunk_documents: %d input documents → %d output chunks.",
        len(documents),
        len(result),
    )
    return result
