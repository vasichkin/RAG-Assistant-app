"""
MaxSim retriever for ColPali-style multi-vector late interaction.

Implements the ColBERT scoring formula:
    score(query, page) = Σ_{q_i} max_{p_j ∈ page} cos(q_i, p_j)

Document side: multiple patch vectors per page (image embeddings).
Query side: overlapping word-window sub-vectors of the query text.
"""

import numpy as np
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class MaxSimRetriever(BaseRetriever):
    """
    Late-interaction MaxSim retriever over per-patch document embeddings.

    Scoring uses the ColBERT formula — sum over query sub-vectors of the
    max cosine similarity to any patch on a page — giving both query and
    document multi-vector representation.

    Attributes:
        documents:         Ordered list of Documents (one per patch).
        embeddings_matrix: Corresponding embedding vectors (one per patch).
        top_k:             Number of top pages to return.
        embeddings_model:  Embeddings instance with embed_query(text).
        query_window:      Word-window size for query sub-vector chunking.
        query_stride:      Stride for the sliding word window.
    """

    documents: list
    embeddings_matrix: list
    top_k: int = 5
    embeddings_model: object = None
    query_window: int = 4
    query_stride: int = 2

    class Config:
        arbitrary_types_allowed = True

    def _embed_query_multivec(self, query: str) -> np.ndarray:
        """Embed query as overlapping word-window sub-vectors.

        Produces Q sub-vectors: one per sliding window over query words
        plus the full query. Short queries (≤ window) get a single vector.
        Returns ndarray of shape (Q, D).
        """
        words = query.split()
        if len(words) <= self.query_window:
            return np.array([self.embeddings_model.embed_query(query)])

        chunks = [
            " ".join(words[i : i + self.query_window])
            for i in range(0, len(words) - self.query_window + 1, self.query_stride)
        ]
        if query not in chunks:
            chunks.append(query)
        return np.array([self.embeddings_model.embed_query(c) for c in chunks])

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        # query_vecs: (Q, D)  doc_vecs: (P, D)
        query_vecs = self._embed_query_multivec(query)
        doc_vecs = np.array(self.embeddings_matrix)

        # Unit-normalise for cosine similarity
        q_unit = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-8)
        p_unit = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)

        # cos_sim[q, p] = cos(q_i, p_j)  shape (Q, P)
        cos_sim = q_unit @ p_unit.T

        # Group patch indices by page
        page_patches: dict = {}
        for idx, doc in enumerate(self.documents):
            pn = doc.metadata.get("page_number", id(doc))
            page_patches.setdefault(pn, []).append(idx)

        # ColBERT score: Σ_q max_{p ∈ page} cos(q, p)
        page_scores = {}
        for pn, patch_idxs in page_patches.items():
            sims = cos_sim[:, patch_idxs]           # (Q, n_patches)
            score = float(sims.max(axis=1).sum())   # Σ_q max_p
            # Representative doc: patch with highest total cross-query contribution
            rep_idx = patch_idxs[int(sims.sum(axis=0).argmax())]
            page_scores[pn] = (score, self.documents[rep_idx])

        sorted_pages = sorted(page_scores.values(), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in sorted_pages[: self.top_k]]
