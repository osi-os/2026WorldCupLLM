"""
Streamlit chat interface for the FIFA World Cup 2026 RAG assistant.

Run from the PROJECT ROOT:
    uv run streamlit run app.py

The RAG engine (documents, text index, vector index, embeddings) is built
ONCE via @st.cache_resource and reused across every interaction and session,
so the app loads the index a single time at startup rather than per query.
"""

import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic

from rag import get_rag


# ---------------------------------------------------------------------------
# One-time setup (cached across reruns and sessions)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the tournament knowledge base...")
def load_rag():
    load_dotenv()
    return get_rag(llm_client=Anthropic(), search_type="hybrid")


EXAMPLE_QUESTIONS = [
    "How did Mexico do against South Africa?",
    "Which team scored the most goals in the group stage?",
    "Where did Argentina play Algeria?",
    "Who was player of the match in the final?",
]


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="World Cup 2026 Assistant", page_icon="⚽", layout="centered")

rag = load_rag()

# Sidebar: retrieval mode + examples
with st.sidebar:
    st.header("⚽ World Cup 2026")
    st.caption("Ask about matches, players, teams, and venues.")

    search_type = st.selectbox(
        "Retrieval method",
        options=["hybrid", "vector", "text"],
        index=0,
        help="Hybrid (text + vector) was the best in evaluation and is the default.",
    )
    rag.search_type = search_type

    st.divider()
    st.subheader("Try an example")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("⚽ FIFA World Cup 2026 Assistant")
st.caption("Grounded in tournament data — answers come only from the dataset.")

# Conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**[{s['doc_type']}] {s['doc_id']}**")
                    st.caption(s["content"].splitlines()[0])


def handle_question(question):
    # Show the user's message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate and show the assistant's answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag.answer(question)
        st.markdown(result["answer"])
        with st.expander("Sources"):
            for s in result["sources"]:
                st.markdown(f"**[{s['doc_type']}] {s['doc_id']}**")
                st.caption(s["content"].splitlines()[0])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "response_time": result["response_time"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
    })


# Example-question button clicked
if "pending_question" in st.session_state:
    q = st.session_state.pop("pending_question")
    handle_question(q)

# Chat input
if prompt := st.chat_input("Ask about the World Cup..."):
    handle_question(prompt)