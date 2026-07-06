"""
Custom LangChain Embeddings subclass using Vertex AI text-embedding-004.
"""

from langchain_core.embeddings import Embeddings
from config.settings import client, EMBEDDING_MODEL


class GeminiEmbeddings(Embeddings):
    """Vertex AI embeddings via google-genai client."""

    BATCH_SIZE = 5     # Vertex AI: 20k total token limit per request; 5 chunks * ~3750 tokens max = 18750
    MAX_CHARS = 15000  # ~20k token safety margin

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Truncate texts that would exceed the token limit
        safe = [t[:self.MAX_CHARS] for t in texts]
        results = []
        for i in range(0, len(safe), self.BATCH_SIZE):
            batch = safe[i : i + self.BATCH_SIZE]
            response = client.models.embed_content(
                model=EMBEDDING_MODEL, contents=batch
            )
            results.extend(e.values for e in response.embeddings)
        return results

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
