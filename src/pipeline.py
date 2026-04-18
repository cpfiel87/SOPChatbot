"""
pipeline.py — The test harness that connects all three modules

WHAT THIS FILE DOES:
  Runs the full retrieval pipeline end-to-end and prints the results so you
  can verify that retrieval is working before adding the LLM response layer.

HOW TO RUN:
  From the project-1-sop-chatbot folder:
    python src/pipeline.py

WHAT TO EXPECT:
  - The embedding model downloads on first run (~90MB, one time only)
  - You'll see progress printed at each stage
  - At the end: the top 3 most relevant chunks for your test query

BEFORE RUNNING:
  - Drop a PDF into the data/ folder
  - Update PDF_PATH below to match your filename
  - Update TEST_QUERY to something relevant to your document
"""

import os
import sys

# Add the src folder to Python's module search path so imports work
sys.path.insert(0, os.path.dirname(__file__))

from loader import load_pdf
from embedder import embed_chunks, get_model
from index import build_index, retrieve
from answerer import answer

# ── Configuration ──────────────────────────────────────────────────────────────
# Update these two values before running

PDF_PATH = "data/sample_sop.pdf"       # path to your PDF (relative to project root)
TEST_QUERY = "what is the risk of using AI?"  # question to test retrieval with

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("RAG Pipeline — Retrieval Test")
    print("=" * 60)

    # Step 1: Load and chunk the PDF
    print("\n[1/5] Loading and chunking PDF...")
    chunks = load_pdf(PDF_PATH)

    # Step 2: Load the embedding model
    print("\n[2/5] Loading embedding model...")
    model = get_model()

    # Step 3: Embed all chunks and build the FAISS index
    print("\n[3/5] Embedding chunks and building index...")
    embeddings = embed_chunks(chunks)
    index = build_index(embeddings)

    # Step 4: Run the test query and print results
    print(f"\n[4/5] Retrieving top 3 chunks for query:")
    print(f"      \"{TEST_QUERY}\"")

    results = retrieve(TEST_QUERY, index, chunks, model, k=3)

    print("\n" + "=" * 60)
    print("RETRIEVAL RESULTS")
    print("=" * 60)

    for result in results:
        print(f"\n--- Rank #{result['rank']} (distance: {result['distance']:.4f}) ---")
        print(result["chunk"])

    # Step 5: Send retrieved chunks + query to Claude for an answer
    print("\n" + "=" * 60)
    print("GENERATING ANSWER")
    print("=" * 60)

    retrieved_texts = [r["chunk"] for r in results]
    response = answer(TEST_QUERY, retrieved_texts)

    print(f"\nAnswer:\n{response['answer']}")

    print("\n" + "=" * 60)
    print(f"Sources: {len(response['sources'])} chunk(s) sent as context")
    print("=" * 60)


if __name__ == "__main__":
    main()
