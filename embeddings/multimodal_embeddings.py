"""
Vertex AI multimodalembedding@001 — shared 1408-dim text+image embedding space.

Unlike text-embedding-005, this model embeds image patches from their visual
content, enabling cross-modal similarity (text query vs. image patch vector).
"""

import logging
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class MultimodalVertexEmbeddings(Embeddings):
    """
    Wraps Vertex AI multimodalembedding@001.

    Both image patches and text queries are embedded in the same 1408-dim space,
    so cosine similarity between a text query vector and an image patch vector
    is a valid cross-modal relevance score.
    """

    def __init__(self):
        from vertexai.vision_models import MultiModalEmbeddingModel
        self._model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

    def embed_image(self, image_bytes: bytes) -> list[float]:
        """Embed a single image patch — vector derived from visual content."""
        from vertexai.vision_models import Image as VertexImage
        img = VertexImage(image_bytes=image_bytes)
        return list(self._model.get_embeddings(image=img).image_embedding)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            list(self._model.get_embeddings(contextual_text=t).text_embedding)
            for t in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        return list(self._model.get_embeddings(contextual_text=text).text_embedding)
