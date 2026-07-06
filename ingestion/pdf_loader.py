"""
ingestion/pdf_loader.py — Extract text from PDF pages using PyMuPDF.

Returns a List[Document] where each Document corresponds to one page.
"""

import logging
import os
from typing import List

import fitz  # PyMuPDF
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf(pdf_path: str, _doc: "fitz.Document | None" = None) -> List[Document]:
    """Load a PDF file and return one Document per page with text content.

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        _doc: Optional already-open fitz.Document. When provided the caller
              owns its lifecycle and it will not be closed here.

    Returns:
        List of Documents, one per page, with metadata:
            - page_number (int, 1-based)
            - source (str, basename of the file)
            - content_type (str, "text")
    """
    source = os.path.basename(pdf_path)
    documents: List[Document] = []
    _owned = _doc is None
    pdf = fitz.open(pdf_path) if _owned else _doc
    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            text = page.get_text()
            if not text.strip():
                logger.debug("Page %d of '%s' has no text — skipping.", page_index + 1, source)
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page_number": page_index + 1,
                        "source": source,
                        "content_type": "text",
                    },
                )
            )
    finally:
        if _owned:
            pdf.close()

    logger.info("Loaded %d text pages from '%s'.", len(documents), source)
    return documents
