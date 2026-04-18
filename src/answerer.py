"""
answerer.py — Sends retrieved chunks + query to Claude and returns an answer

WHAT THIS FILE DOES:
  Takes the top-k chunks from FAISS retrieval and calls the Anthropic API
  to generate a grounded, citation-aware answer.
  Maintains conversation history across turns so Claude has full context.

REQUIRES:
  ANTHROPIC_API_KEY in a .env file at the project root (project-1-sop-chatbot/.env)
"""

import os
from dotenv import load_dotenv
import anthropic
import streamlit as st

load_dotenv()

SYSTEM_PROMPT = (
    "You are a GxP compliance assistant. Answer questions based ONLY "
    "on the provided SOP sections. Always cite which section your answer "
    "comes from. If the answer is not in the provided sections, respond "
    "with: 'This topic is not covered in the loaded documents.'"
)

# Conversation history: list of {"role": "user"|"assistant", "content": str}
_history: list[dict] = []


def answer(query: str, retrieved_chunks: list[str]) -> dict:
    """
    Call the Claude API with the query and retrieved context chunks.
    Conversation history is maintained across calls.

    Args:
        query:            The user's question.
        retrieved_chunks: Ordered list of chunk strings (most relevant first).

    Returns:
        {
            "answer":   str,        # Claude's response text
            "sources":  list[str],  # The chunks that were sent as context
        }
    """
    if not retrieved_chunks:
        return {
            "answer": "This topic is not covered in the loaded documents.",
            "sources": [],
        }

    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not found. Add it to your .env file or Streamlit secrets."
        )

    # Build numbered context block
    context_lines = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_lines.append(f"[Section {i}]\n{chunk}")
    context_block = "\n\n".join(context_lines)

    user_message = (
        f"Context sections:\n\n{context_block}\n\n"
        f"Question: {query}"
    )

    _history.append({"role": "user", "content": user_message})

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=_history,
    )

    answer_text = message.content[0].text
    _history.append({"role": "assistant", "content": answer_text})

    return {
        "answer": answer_text,
        "sources": retrieved_chunks,
    }


def reset_history() -> None:
    """Clear the conversation history to start a fresh session."""
    _history.clear()
