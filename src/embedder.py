"""
embedder.py — Step 2 of the RAG pipeline

WHAT THIS FILE DOES:
  Converts a list of text chunks into a list of vectors (arrays of numbers).
  Each vector "represents" the meaning of that chunk in mathematical space.

WHY VECTORS?
  You can't search text by meaning directly. But vectors allow you to measure
  similarity mathematically: two chunks about the same topic will have vectors
  that point in similar directions (high cosine similarity / low L2 distance).

  This is the core idea behind semantic search — you search by MEANING,
  not by keyword matching.

THE MODEL: sentence-transformers/all-MiniLM-L6-v2
  - Runs entirely on your local machine (no API call, no cost)
  - Downloads once (~90MB) and is cached locally after that
  - Produces vectors of size 384 (384 numbers per chunk)
  - Fast on CPU, accurate enough for most retrieval tasks
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Load the model once at module level so it's not reloaded on every call
# The first time this runs it will download the model (~90MB)
_model = None


def get_model() -> SentenceTransformer:
    """Returns the embedding model, loading it only once (singleton pattern)."""
    global _model
    if _model is None:
        print("[embedder] Loading embedding model (first run downloads ~90MB)...")
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("[embedder] Model loaded.")
    return _model


def embed_chunks(chunks: list[str]) -> np.ndarray:
    """
    Converts a list of text strings into a 2D numpy array of embeddings.

    Args:
        chunks: list of text strings (your document chunks)

    Returns:
        numpy array of shape (num_chunks, 384)
        Each row is the vector for one chunk.

    Example:
        chunks = ["The onboarding process starts...", "Employee must sign..."]
        embeddings = embed_chunks(chunks)
        # embeddings.shape → (2, 384)
    """
    model = get_model()

    # encode() runs all chunks through the model and returns a numpy array
    # show_progress_bar=True prints a progress bar for large batches
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    print(f"[embedder] Embedded {len(chunks)} chunks → shape {embeddings.shape}")
    return embeddings
