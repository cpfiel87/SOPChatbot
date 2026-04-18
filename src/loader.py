"""
loader.py — Step 1 of the RAG pipeline

WHAT THIS FILE DOES:
  1. Opens a PDF file and reads all the text out of it
  2. Splits that text into small overlapping chunks

WHY WE CHUNK:
  - Embedding models have a token limit (usually 256–512 tokens)
  - Smaller chunks = more precise retrieval (you get only the relevant section,
    not an entire 20-page document)
  - Overlap (50 tokens) prevents a sentence from being cut off at a chunk
    boundary and losing its meaning
"""

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(path: str) -> list[str]:
    """
    Reads a PDF and returns a list of text chunks.

    Args:
        path: file path to the PDF, e.g. "data/sample_sop.pdf"

    Returns:
        A list of strings, each string is one chunk (~500 tokens)
    """

    # --- STEP A: Read all pages from the PDF ---
    # PdfReader opens the file and gives us access to each page
    reader = PdfReader(path)

    # Extract text from every page and join into one big string
    # Some pages may return None if they have no text (e.g. image-only pages)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:  # skip pages with no extractable text
            full_text += text + "\n"

    # --- STEP B: Split the text into chunks ---
    # RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
    # then sentences, then words — so chunks stay as semantically coherent
    # as possible.
    #
    # chunk_size=2000 (characters, not tokens — ~500 tokens at ~4 chars/token)
    # chunk_overlap=200 characters (~50 tokens) so context isn't lost at edges
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(full_text)

    print(f"[loader] Loaded '{path}' → {len(reader.pages)} pages → {len(chunks)} chunks")
    return chunks
