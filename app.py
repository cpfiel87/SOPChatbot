"""
app.py — Streamlit UI for the SOP Q&A chatbot

HOW TO RUN:
    From the project-1-sop-chatbot folder:
        streamlit run app.py
"""

import os
import sys
import tempfile
import time

import streamlit as st

SESSION_LIMIT_SECONDS = 900  # 15 minutes

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from loader import load_pdf
from embedder import embed_chunks, get_model
from index import build_index, retrieve
from answerer import answer, reset_history

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SOP Compliance Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Flip user messages to the right */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
    [data-testid="stChatMessageContent"] {
    align-items: flex-end;
}

/* User bubble background */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
    [data-testid="stMarkdownContainer"] p {
    background: #1e3a5f;
    border-radius: 16px 16px 4px 16px;
    padding: 10px 14px;
    display: inline-block;
    max-width: 80%;
}

/* Assistant bubble background */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
    [data-testid="stMarkdownContainer"] p {
    background: #1e1e2e;
    border-radius: 16px 16px 16px 4px;
    padding: 10px 14px;
    display: inline-block;
    max-width: 80%;
}

/* Disclaimer banner */
.disclaimer {
    background: #12121f;
    border-left: 4px solid #e05c5c;
    padding: 12px 18px;
    border-radius: 0 6px 6px 0;
    font-size: 0.84rem;
    color: #bbb;
    margin-bottom: 1.2rem;
    line-height: 1.5;
}

/* Sidebar label */
section[data-testid="stSidebar"] .stMetric label {
    font-size: 0.78rem;
    color: #888;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────

_DEFAULTS = {
    "messages": [],           # list of {"role", "content", "sources"}
    "index": None,            # faiss.IndexFlatL2
    "chunks": [],             # list[str]
    "embed_model": None,      # SentenceTransformer
    "pdf_name": None,         # str
    "authenticated": False,
    "session_start_time": None,
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Access code gate ───────────────────────────────────────────────────────────

if not st.session_state.authenticated:
    st.title("SOP Compliance Assistant — Access Required")
    code_input = st.text_input("Enter your access code:", type="password")
    if st.button("Submit"):
        valid_codes = st.secrets.get("ACCESS_CODES", [])
        if code_input in valid_codes:
            st.session_state.authenticated = True
            st.session_state.session_start_time = time.time()
            st.rerun()
        else:
            st.error("Invalid access code. Please contact the administrator.")
    st.stop()

# ── Session expiry check ───────────────────────────────────────────────────────

_elapsed = time.time() - st.session_state.session_start_time
if _elapsed >= SESSION_LIMIT_SECONDS:
    st.warning("Your 15-minute session has expired. Please contact the administrator for a new code.")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📋 Document Setup")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload SOP (PDF)",
        type="pdf",
        help="Upload the Standard Operating Procedure document you want to query.",
    )

    build_clicked = st.button(
        "Build Index",
        disabled=(uploaded_file is None),
        use_container_width=True,
        type="primary",
    )

    if build_clicked and uploaded_file:
        with st.spinner("Processing PDF — this may take a moment…"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                chunks = load_pdf(tmp_path)
                model = get_model()
                embeddings = embed_chunks(chunks)
                index = build_index(embeddings)
            finally:
                os.unlink(tmp_path)

        st.session_state.index = index
        st.session_state.chunks = chunks
        st.session_state.embed_model = model
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.messages = []
        reset_history()

        st.success(f"Indexed **{uploaded_file.name}**")

    if st.session_state.chunks:
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Chunks", len(st.session_state.chunks))
        col2.metric("Top-k", 3)
        st.caption(f"📄 {st.session_state.pdf_name}")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        reset_history()
        st.rerun()

    st.divider()
    _mins_left = max(1, int((SESSION_LIMIT_SECONDS - _elapsed) / 60))
    st.caption(f"Session time remaining: ~{_mins_left} min")
    st.caption("Powered by **Claude** · sentence-transformers · FAISS")

# ── Main area ──────────────────────────────────────────────────────────────────

st.title("SOP Compliance Assistant")

st.markdown(
    '<div class="disclaimer">'
    "⚠️ <strong>For informational use only.</strong> "
    "Answers are generated exclusively from the uploaded SOP document. "
    "They may not reflect the latest regulatory guidance or amendments. "
    "Always verify with your QA / RA team before making any compliance decision."
    "</div>",
    unsafe_allow_html=True,
)

if st.session_state.index is None:
    st.info(
        "Upload a PDF in the sidebar and click **Build Index** to get started.",
        icon="👈",
    )
    st.stop()

# ── Chat history ───────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("View source sections", expanded=False):
                for i, src in enumerate(msg["sources"], start=1):
                    st.markdown(f"**Section {i}**")
                    st.caption(src)
                    if i < len(msg["sources"]):
                        st.divider()

# ── Chat input ─────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a question about the SOP…"):

    # Render and store user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "sources": []}
    )

    # Retrieve chunks and generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching document…"):
            results = retrieve(
                prompt,
                st.session_state.index,
                st.session_state.chunks,
                st.session_state.embed_model,
                k=3,
            )
            retrieved_texts = [r["chunk"] for r in results]
            response = answer(prompt, retrieved_texts)

        st.markdown(response["answer"])

        if response["sources"]:
            with st.expander("View source sections", expanded=False):
                for i, src in enumerate(response["sources"], start=1):
                    st.markdown(f"**Section {i}**")
                    st.caption(src)
                    if i < len(response["sources"]):
                        st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "sources": response["sources"],
    })
