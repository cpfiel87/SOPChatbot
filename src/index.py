"""
index.py — Step 3 of the RAG pipeline

WHAT THIS FILE DOES:
  1. Takes all the chunk embeddings and stores them in a FAISS index
  2. Given a user query, finds the top-k most similar chunks

WHAT IS FAISS?
  FAISS (Facebook AI Similarity Search) is a library that stores vectors
  and lets you search them extremely fast — even with millions of vectors.

  Think of it like a dictionary where the "keys" are vectors (not words),
  and you look up by similarity instead of exact match.

HOW SIMILARITY WORKS:
  We use L2 distance (Euclidean distance) — the straight-line distance
  between two vectors in 384-dimensional space. Smaller distance = more
  similar meaning.

  When you search, FAISS returns the indices of the closest vectors,
  which we then map back to the original text chunks.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Creates a FAISS index from a set of embeddings.

    Args:
        embeddings: numpy array of shape (num_chunks, 384)

    Returns:
        A FAISS index with all embeddings stored inside it

    Note:
        IndexFlatL2 is an "exact" index — it checks every single vector.
        This is fine for hundreds or low thousands of documents.
        For millions of documents you'd use an approximate index (IndexIVFFlat).
    """
    # The dimension must match the embedding size (384 for all-MiniLM-L6-v2)
    dimension = embeddings.shape[1]

    # Create the index
    index = faiss.IndexFlatL2(dimension)

    # FAISS requires float32
    index.add(embeddings.astype(np.float32))

    print(f"[index] Built FAISS index with {index.ntotal} vectors (dim={dimension})")
    return index


def retrieve(
    query: str,
    index: faiss.IndexFlatL2,
    chunks: list[str],
    model: SentenceTransformer,
    k: int = 3,
) -> list[dict]:
    """
    Finds the top-k chunks most relevant to the query.

    Args:
        query:  the user's question, e.g. "What is the onboarding procedure?"
        index:  the FAISS index built from build_index()
        chunks: the original list of text strings (same order as the index)
        model:  the SentenceTransformer model used to embed the query
        k:      how many chunks to return (default 3)

    Returns:
        A list of dicts, each with:
          - "chunk":    the text of the retrieved chunk
          - "distance": the L2 distance (lower = more similar)
          - "rank":     1 = best match, 2 = second best, etc.

    HOW IT WORKS:
        1. Embed the query into a vector (same model, same 384 dimensions)
        2. Ask FAISS to find the k nearest vectors in the index
        3. FAISS returns indices + distances
        4. We map those indices back to the original text chunks
    """
    # Embed the query — must be shape (1, 384) for FAISS
    query_embedding = model.encode([query], convert_to_numpy=True).astype(np.float32)

    # Search the index
    # distances: shape (1, k) — L2 distances to each result
    # indices:   shape (1, k) — positions in the chunks list
    distances, indices = index.search(query_embedding, k)

    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        results.append({
            "rank": rank,
            "chunk": chunks[idx],
            "distance": float(dist),
        })

    return results
