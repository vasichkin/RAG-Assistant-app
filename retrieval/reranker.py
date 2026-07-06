"""
Cross-encoder re-ranking using sentence-transformers.

Usage:
    from retrieval.reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    docs = reranker.rerank(query, docs, top_k=5)
"""

from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self):
        self.model = CrossEncoder(MODEL_NAME)

    def rerank(self, query: str, docs: list, top_k: int = 5) -> list:
        """
        Re-rank docs by relevance to query using cross-encoder.
        Returns top_k docs sorted by descending score.
        """
        if not docs:
            return docs
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)
        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]
