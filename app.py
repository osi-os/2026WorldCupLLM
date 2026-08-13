"""
Streamlit chat interface for the FIFA World Cup 2026 RAG assistant,
with monitoring: every Q&A is logged to Postgres and users can leave
thumbs up/down feedback. Grafana reads the tables for the dashboard.
 
Run from the PROJECT ROOT (with Postgres running via docker-compose):
    uv run streamlit run app.py

The RAG engine (documents, text index, vector index, embeddings) is built
ONCE via @st.cache_resource and reused across every interaction and session,
so the app loads the index a single time at startup rather than per query.
"""

import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic
 
from rag import get_rag
from db import init_db, save_conversation, save_feedback, compute_cost
 
 
# ---------------------------------------------------------------------------
# One-time setup (cached across reruns and sessions)
# ---------------------------------------------------------------------------
 
@st.cache_resource(show_spinner="Loading the tournament knowledge base...")
def load_rag():
    load_dotenv()
    return get_rag(llm_client=Anthropic(), search_type="hybrid")
 
 
@st.cache_resource
def setup_db():
    """Create tables once at startup. Returns True if the DB is reachable."""
    try:
        init_db()
        return True
    except Exception as e:
        print(f"Monitoring DB unavailable: {e}")
        return False
 
 
EXAMPLE_QUESTIONS = [
    "How did Mexico do against South Africa?",
    "Which team scored the most goals in the group stage?",
    "Where did Argentina play Algeria?",
    "Who won the final?",
]
 
 
# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
 
st.set_page_config(page_title="World Cup 2026 Assistant", page_icon="⚽", layout="centered")
 
rag = load_rag()
db_ready = setup_db()
 
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
 
    if not db_ready:
        st.warning("Monitoring database not connected — answers work, but "
                   "conversations and feedback aren't being logged. Start it "
                   "with `docker compose up -d`.")
 
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
 
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}
 
 
def render_sources(sources):
    with st.expander("Sources"):
        for s in sources:
            st.markdown(f"**[{s['doc_type']}] {s['doc_id']}**")
            st.caption(s["content"].splitlines()[0])
 
 
def render_feedback(conv_id):
    """Thumbs up/down for a logged conversation."""
    if conv_id is None:
        return
    given = st.session_state.feedback_given
    if conv_id in given:
        st.caption("✓ Thanks for your feedback " + ("👍" if given[conv_id] == 1 else "👎"))
        return
    c1, c2, _ = st.columns([1, 1, 8])
    if c1.button("👍", key=f"up_{conv_id}"):
        save_feedback(conv_id, 1)
        given[conv_id] = 1
        st.rerun()
    if c2.button("👎", key=f"down_{conv_id}"):
        save_feedback(conv_id, -1)
        given[conv_id] = -1
        st.rerun()
 
 
# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("sources"):
                render_sources(msg["sources"])
            render_feedback(msg.get("conversation_id"))
 
 
def handle_question(question):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
 
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag.answer(question)
        st.markdown(result["answer"])
        render_sources(result["sources"])
 
        # Log to the monitoring DB
        conv_id = None
        if db_ready:
            cost = compute_cost(result["model"],
                                result["input_tokens"], result["output_tokens"])
            try:
                conv_id = save_conversation(
                    question=question,
                    answer=result["answer"],
                    search_type=rag.search_type,
                    model=result["model"],
                    input_tokens=result["input_tokens"],
                    output_tokens=result["output_tokens"],
                    cost=cost,
                    response_time=result["response_time"],
                )
            except Exception as e:
                print(f"Failed to log conversation: {e}")
 
        render_feedback(conv_id)
 
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "conversation_id": conv_id,
    })
 
 
if "pending_question" in st.session_state:
    handle_question(st.session_state.pop("pending_question"))
 
if prompt := st.chat_input("Ask about the World Cup..."):
    handle_question(prompt)