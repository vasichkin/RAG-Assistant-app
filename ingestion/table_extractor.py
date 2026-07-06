"""
ingestion/table_extractor.py — Extract tables from PDF pages using PyMuPDF
and convert each table to a Markdown string.

Returns a List[Document] where each Document holds one table as Markdown.
"""

import logging
import os
from typing import List

import fitz  # PyMuPDF
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _table_to_markdown(table) -> str:
    """Convert a PyMuPDF TableFinder result to a Markdown table string."""
    rows = table.extract()
    if not rows:
        return ""

    lines: List[str] = []
    for row_index, row in enumerate(rows):
        # Normalise None cells to empty string
        cells = [str(cell) if cell is not None else "" for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
        if row_index == 0:
            # Add the Markdown separator after the header row
            lines.append("| " + " | ".join(["---"] * len(cells)) + " |")

    return "\n".join(lines)


def extract_tables(pdf_path: str, _doc: "fitz.Document | None" = None) -> List[Document]:
    """Extract tables from a PDF and return each as a Markdown Document.

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        _doc: Optional already-open fitz.Document. When provided the caller
              owns its lifecycle and it will not be closed here.

    Returns:
        List of Documents, one per non-empty table, with metadata:
            - page_number (int, 1-based)
            - source (str, basename of the file)
            - content_type (str, "table")
            - table_index (int, 0-based index within the page)
    """
    source = os.path.basename(pdf_path)
    documents: List[Document] = []
    _owned = _doc is None
    pdf = fitz.open(pdf_path) if _owned else _doc
    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            try:
                tables = page.find_tables()
            except Exception as exc:
                logger.warning(
                    "Failed to find tables on page %d of '%s': %s",
                    page_index + 1,
                    source,
                    exc,
                )
                continue

            for table_index, table in enumerate(tables):
                try:
                    markdown = _table_to_markdown(table)
                    if not markdown.strip():
                        logger.debug(
                            "Empty table %d on page %d — skipping.",
                            table_index,
                            page_index + 1,
                        )
                        continue
                    documents.append(
                        Document(
                            page_content=markdown,
                            metadata={
                                "page_number": page_index + 1,
                                "source": source,
                                "content_type": "table",
                                "table_index": table_index,
                            },
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to extract table %d on page %d of '%s': %s",
                        table_index,
                        page_index + 1,
                        source,
                        exc,
                    )
                    continue
    finally:
        if _owned:
            pdf.close()

    logger.info("Extracted %d tables from '%s'.", len(documents), source)
    return documents
