"""
ColPali-style page-patch visual retrieval.

Each PDF page is split into image patches; Gemini describes each patch.
Descriptions are embedded for semantic search via MaxSim over multi-vector
patch representations (see maxsim_retriever.py).
"""

import base64
import logging

import fitz  # PyMuPDF

from langchain_core.documents import Document

from config.settings import client, GENERATION_MODEL

logger = logging.getLogger(__name__)


def render_page_to_image(pdf_path: str, page_number: int) -> bytes:
    """Render a PDF page (0-indexed) to PNG bytes at 150 dpi."""
    pdf = fitz.open(pdf_path)
    try:
        page = pdf[page_number]
        pixmap = page.get_pixmap(dpi=150)
        return pixmap.tobytes("png")
    finally:
        pdf.close()


def _render_patches_from_doc(doc: fitz.Document, page_number: int, n_patches_per_side: int = 2) -> list[bytes]:
    """Render patches from an already-open fitz.Document (0-indexed page)."""
    page = doc[page_number]
    rect = page.rect
    pw = rect.width / n_patches_per_side
    ph = rect.height / n_patches_per_side
    patches = []
    for row in range(n_patches_per_side):
        for col in range(n_patches_per_side):
            clip = fitz.Rect(col * pw, row * ph, (col + 1) * pw, (row + 1) * ph)
            pixmap = page.get_pixmap(dpi=150, clip=clip)
            patches.append(pixmap.tobytes("png"))
    return patches


def render_page_patches(pdf_path: str, page_number: int, n_patches_per_side: int = 2) -> list[bytes]:
    """Render a PDF page (0-indexed) into n×n patch PNG bytes using fitz clip rects."""
    pdf = fitz.open(pdf_path)
    try:
        return _render_patches_from_doc(pdf, page_number, n_patches_per_side)
    finally:
        pdf.close()


def describe_page_with_gemini(image_bytes: bytes, page_number: int, patch_index: int = None) -> str:
    """Use Gemini to describe a PDF page or patch image."""
    b64_image = base64.b64encode(image_bytes).decode()
    if patch_index is not None:
        prompt = (
            f"Describe patch {patch_index} of this financial document page. "
            "Include all visible text, numbers, table cells, chart elements, and financial figures in this region. "
            "Be thorough so the description can be used for semantic search."
        )
    else:
        prompt = (
            "Describe this financial document page in detail. "
            "Include: all text content, table data with numbers, "
            "chart descriptions, and any key financial figures. "
            "Be comprehensive so the description can be used for semantic search."
        )
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=[{
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": "image/png", "data": b64_image}},
                {"text": prompt}
            ]
        }]
    )
    return response.text


def build_colpali_documents(pdf_path: str, n_patches_per_side: int = 2) -> list[Document]:
    """
    Generate ColPali-style patch documents: n_patches_per_side² docs per PDF page.

    Each document's page_content is Gemini's description of one image patch.
    Metadata includes page_number, patch_index, source, and content_type.
    """
    documents: list[Document] = []
    n_patches = n_patches_per_side * n_patches_per_side

    with fitz.open(pdf_path) as pdf:
        total_pages = len(pdf)
        for page_num in range(total_pages):
            try:
                patches = _render_patches_from_doc(pdf, page_num, n_patches_per_side)
                for patch_idx, patch_bytes in enumerate(patches):
                    description = describe_page_with_gemini(
                        patch_bytes, page_num + 1, patch_index=patch_idx
                    )
                    documents.append(Document(
                        page_content=description,
                        metadata={
                            "page_number": page_num + 1,
                            "patch_index": patch_idx,
                            "source": pdf_path,
                            "content_type": "colpali_patch",
                        },
                    ))
                logger.info("Indexed page %d/%d (%d patches)", page_num + 1, total_pages, n_patches)
            except Exception as e:
                logger.warning("Failed to process page %d: %s", page_num + 1, e)

    return documents


def build_colpali_with_image_embeddings(
    pdf_path: str,
    multimodal_emb,
    n_patches_per_side: int = 2,
) -> tuple[list[Document], list[list[float]]]:
    """
    Build ColPali patch documents and compute IMAGE embeddings directly.

    Each patch is rendered as PNG and embedded via multimodal_emb.embed_image()
    so the stored vector derives from visual content, not a text caption.
    Gemini descriptions are still generated for page_content (used at generation
    time to provide context alongside the page image).

    Args:
        pdf_path:          Path to the PDF file.
        multimodal_emb:    MultimodalVertexEmbeddings instance.
        n_patches_per_side: Grid size — n² patches per page.

    Returns:
        (documents, image_embeddings_matrix)
    """
    documents: list[Document] = []
    embeddings_matrix: list[list[float]] = []

    with fitz.open(pdf_path) as pdf:
        total_pages = len(pdf)
        for page_num in range(total_pages):
            try:
                patches = _render_patches_from_doc(pdf, page_num, n_patches_per_side)
                for patch_idx, patch_bytes in enumerate(patches):
                    description = describe_page_with_gemini(
                        patch_bytes, page_num + 1, patch_index=patch_idx
                    )
                    image_vec = multimodal_emb.embed_image(patch_bytes)
                    documents.append(Document(
                        page_content=description,
                        metadata={
                            "page_number": page_num + 1,
                            "patch_index": patch_idx,
                            "source": pdf_path,
                            "content_type": "colpali_patch",
                        },
                    ))
                    embeddings_matrix.append(image_vec)
                logger.info(
                    "Indexed page %d/%d (%d patches with image embeddings)",
                    page_num + 1, total_pages, n_patches_per_side ** 2,
                )
            except Exception as e:
                logger.warning("Failed to process page %d: %s", page_num + 1, e)

    return documents, embeddings_matrix
