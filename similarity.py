"""
similarity.py - how much of the original MEANING an attacked text keeps.

An attack that removes the watermark by turning the text into nonsense is not
a meaningful success. We measure meaning with sentence embeddings:

    1. Turn each text into a vector with the sentence-transformers model
       "all-MiniLM-L6-v2" (~90 MB, runs fine on CPU).
    2. Cosine similarity between the two vectors:
         ~1.0 = same meaning, ~0.5 = loosely related, ~0 = unrelated.

The model is loaded only when first needed, then reused.
"""

from __future__ import annotations

SIMILARITY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # CPU keeps the GPU free for Qwen; this model is small and fast.
        _model = SentenceTransformer(SIMILARITY_MODEL, device="cpu")
    return _model


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity of the two texts' embeddings (roughly 0 to 1).
    Raises ValueError for empty input."""
    if not text_a or not text_a.strip() or not text_b or not text_b.strip():
        raise ValueError("semantic_similarity: both texts must be non-empty")
    from sentence_transformers.util import cos_sim
    emb = _get_model().encode([text_a, text_b], convert_to_tensor=True,
                              normalize_embeddings=True)
    return float(cos_sim(emb[0], emb[1]))
