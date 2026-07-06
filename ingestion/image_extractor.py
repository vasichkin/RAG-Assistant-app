"""
ingestion/image_extractor.py — Extract images from PDF pages and generate
captions via Gemini (Vertex AI).

Returns a List[Document] where each Document holds a Gemini-generated caption
for one extracted image.
"""

import base64
import logging
import os
from typing import List

import fitz  # PyMuPDF
from langchain_core.documents import Document

from config.settings import client, GENERATION_MODEL

logger = logging.getLogger(__name__)


def extract_images(pdf_path: str, _doc: "fitz.Document | None" = None) -> List[Document]:
    """Extract images from a PDF and caption each one with Gemini.

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        _doc: Optional already-open fitz.Document. When provided the caller
              owns its lifecycle and it will not be closed here.

    Returns:
        List of Documents, one per successfully captioned image, with metadata:
            - page_number (int, 1-based)
            - source (str, basename of the file)
            - content_type (str, "image")
            - image_index (int, 0-based index within the page)
    """
    source = os.path.basename(pdf_path)
    documents: List[Document] = []
    _owned = _doc is None
    pdf = fitz.open(pdf_path) if _owned else _doc
    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    # Encode to PNG via pixmap for consistent format
                    pixmap = fitz.Pixmap(pdf, xref)
                    if pixmap.colorspace and pixmap.colorspace.n > 3:
                        # Convert CMYK or other colour spaces to RGB
                        pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
                    png_bytes = pixmap.tobytes("png")
                    encoded = base64.b64encode(png_bytes).decode("utf-8")

                    caption_response = client.models.generate_content(
                        model=GENERATION_MODEL,
                        contents=[
                            {
                                "role": "user",
                                "parts": [
                                    {
                                        "inline_data": {
                                            "mime_type": "image/png",
                                            "data": encoded,
                                        }
                                    },
                                    {
                                        "text": (
                                            "Describe this image in detail. "
                                            "Focus on any data, charts, graphs, "
                                            "figures, or meaningful visual content."
                                        )
                                    },
                                ]
                            }
                        ],
                    )
                    caption = caption_response.text.strip()

                    documents.append(
                        Document(
                            page_content=caption,
                            metadata={
                                "page_number": page_index + 1,
                                "source": source,
                                "content_type": "image",
                                "image_index": img_index,
                            },
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to process image %d on page %d of '%s': %s",
                        img_index,
                        page_index + 1,
                        source,
                        exc,
                    )
                    continue
    finally:
        if _owned:
            pdf.close()

    logger.info(
        "Extracted and captioned %d images from '%s'.", len(documents), source
    )
    return documents
