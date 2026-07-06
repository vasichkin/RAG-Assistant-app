"""
Run RAGAS evaluation on the Phase 6 ColPali retrieval pipeline.

Saves results to evaluation/results/phase6_ragas.json.

Usage:
    python -m scripts.run_phase6_eval
"""

import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from vectorstore.qdrant_store import COLPALI_COLLECTION

PDF_PATH = "resources/documents/ifc-annual-report-2024-financials.pdf"
CSV_PATH = "resources/datasets/rag_evaluation_dataset.csv"
OUTPUT_PATH = "evaluation/results/phase6_ragas.json"


COLPALI_VECTOR_SIZE = 1408  # multimodalembedding@001 output dimension


def _scroll_collection(qclient, collection_name: str):
    """Recover all documents and their embeddings from a Qdrant collection."""
    from langchain_core.documents import Document

    documents = []
    embeddings_matrix = []
    offset = None
    while True:
        result, offset = qclient.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for point in result:
            payload = point.payload or {}
            doc = Document(
                page_content=payload.get("page_content", ""),
                metadata=payload.get("metadata", {}),
            )
            documents.append(doc)
            vec = point.vector
            embeddings_matrix.append(list(vec) if not isinstance(vec, list) else vec)
        if offset is None:
            break
    return documents, embeddings_matrix


def _upsert_colpali_collection(qclient, collection_name: str, docs, emb_matrix: list, vector_size: int):
    """Store pre-computed image embeddings in Qdrant, bypassing the text embedding path."""
    from qdrant_client.models import Distance, VectorParams, PointStruct

    existing = [c.name for c in qclient.get_collections().collections]
    if collection_name in existing:
        qclient.delete_collection(collection_name)

    qclient.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=i,
            vector=emb,
            payload={"page_content": doc.page_content, "metadata": doc.metadata},
        )
        for i, (doc, emb) in enumerate(zip(docs, emb_matrix))
    ]
    qclient.upsert(collection_name=collection_name, points=points)
    logger.info("Upserted %d image-embedding vectors to '%s'.", len(points), collection_name)


def main():
    from embeddings.multimodal_embeddings import MultimodalVertexEmbeddings
    from multimodal.colpali_embeddings import build_colpali_with_image_embeddings, render_page_to_image
    from multimodal.maxsim_retriever import MaxSimRetriever
    from evaluation.ragas_eval import run_ragas_evaluation
    from observability.langfuse_handler import get_handler
    from langchain_core.runnables import RunnableLambda
    from qdrant_client import QdrantClient
    from config.settings import QDRANT_HOST, QDRANT_PORT, client as genai_client, GENERATION_MODEL

    multimodal_emb = MultimodalVertexEmbeddings()

    qdrant_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
    qclient = QdrantClient(url=qdrant_url)
    existing = [c.name for c in qclient.get_collections().collections]

    # Check if existing collection was built with image embeddings (1408-dim).
    # Old collections used text-embedding-005 (768-dim) and must be rebuilt.
    collection_ready = False
    if COLPALI_COLLECTION in existing:
        info = qclient.get_collection(COLPALI_COLLECTION)
        stored_size = info.config.params.vectors.size
        if stored_size == COLPALI_VECTOR_SIZE:
            collection_ready = True
        else:
            logger.info(
                "Existing '%s' has %d-dim vectors (text embeddings) — rebuilding with image embeddings.",
                COLPALI_COLLECTION, stored_size,
            )

    if collection_ready:
        logger.info("Loading existing '%s' collection via scroll.", COLPALI_COLLECTION)
        docs, emb_matrix = _scroll_collection(qclient, COLPALI_COLLECTION)
        logger.info("Recovered %d patch documents from Qdrant.", len(docs))
        retriever = MaxSimRetriever(
            documents=docs,
            embeddings_matrix=emb_matrix,
            top_k=5,
            embeddings_model=multimodal_emb,
        )
    else:
        logger.info(
            "Collection '%s' not found — building from PDF with patch segmentation "
            "and IMAGE embeddings (calls Gemini + multimodalembedding once per patch).",
            COLPALI_COLLECTION,
        )
        docs, emb_matrix = build_colpali_with_image_embeddings(PDF_PATH, multimodal_emb)
        logger.info("Built %d patch documents with image embeddings.", len(docs))
        _upsert_colpali_collection(qclient, COLPALI_COLLECTION, docs, emb_matrix, COLPALI_VECTOR_SIZE)
        retriever = MaxSimRetriever(
            documents=docs,
            embeddings_matrix=emb_matrix,
            top_k=5,
            embeddings_model=multimodal_emb,
        )

    # Multimodal generation: pass actual page images + patch descriptions to Gemini
    def _multimodal_generate(query: str) -> str:
        retrieved = retriever.invoke(query)
        parts = []
        seen_pages = set()
        for doc in retrieved:
            pn = doc.metadata.get("page_number")
            source = doc.metadata.get("source", PDF_PATH)
            if pn and pn not in seen_pages:
                try:
                    img_bytes = render_page_to_image(source, pn - 1)
                    b64 = base64.b64encode(img_bytes).decode()
                    parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})
                except Exception as exc:
                    logger.warning("Could not render page %d: %s", pn, exc)
                seen_pages.add(pn)
            parts.append({"text": f"[Page {pn} description]\n{doc.page_content}"})
        parts.append({
            "text": (
                f"\nQuestion: {query}\n"
                "Answer comprehensively using the page images and descriptions above:"
            )
        })
        response = genai_client.models.generate_content(
            model=GENERATION_MODEL,
            contents=[{"role": "user", "parts": parts}],
        )
        return response.text

    chain = RunnableLambda(_multimodal_generate)
    handler = get_handler()

    logger.info("Running RAGAS evaluation on Phase 6 ColPali pipeline…")
    scores = run_ragas_evaluation(
        csv_path=CSV_PATH,
        retriever=retriever,
        chain=chain,
        output_path=OUTPUT_PATH,
        langfuse_handler=handler,
    )
    handler.flush()

    logger.info("Phase 6 RAGAS scores: %s", scores)
    print(f"\nSaved to {OUTPUT_PATH}")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}" if value is not None else f"  {metric}: N/A")


if __name__ == "__main__":
    main()
