"""
RAG flow for the FIFA World Cup 2026 project.

Retrieval options:
    - text_search    : minsearch keyword index over document content
    - vector_search  : minsearch VectorSearch over ONNX all-MiniLM-L6-v2 embeddings
    - hybrid_search  : reciprocal rank fusion (RRF) of the two

text_search builds a minsearch text index over the documents produced by ingest_docs_for_rag.py,
and wraps retrieval + prompting + the LLM call in a RAGFifa class
(same shape as the RAGBase from LLMzoomcamp).

Embeddings are the expensive part, so they are persisted to disk
(embeddings.npy + embeddings_ids.json) and only recomputed when the
underlying documents change. This is the "build once, refresh when the
dataset updates" behaviour.

Both indexes are built once via get_rag() and cached at module level, so
they are not rebuilt on every query. In Streamlit this is wrapped again
with @st.cache_resource.

Usage:
    from anthropic import Anthropic
    from rag import get_rag
 
    rag = get_rag(llm_client=Anthropic())
    print(rag.rag("How did Mexico do against South Africa?"))
"""

import os
import json
 
import numpy as np
from minsearch import Index, VectorSearch
 
from data_and_ingestion.ingest_docs_for_rag import load_documents
from embedder_scripts.embedder import Embedder
 
 
INSTRUCTIONS = """
You're a FIFA World Cup 2026 assistant.
You answer questions about the tournament using only the provided context,
which contains match results, player stats, team records, and venues.
 
Use the context to find relevant information and give an accurate, specific
answer. Prefer exact numbers, names, and dates from the context. If the answer
is not in the context, say "I don't have that information in the tournament data."
""".strip()
 
PROMPT_TEMPLATE = """
QUESTION: {question}
 
CONTEXT:
{context}
""".strip()
 
 
# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (for hybrid search)
# ---------------------------------------------------------------------------
 
def rrf(result_lists, k=60, num_results=5):
    """Fuse several ranked result lists into one, keyed on doc_id."""
    scores = {}
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = doc["doc_id"]
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]
 
 
# ---------------------------------------------------------------------------
# The RAG class
# ---------------------------------------------------------------------------
 
class RAGFifa:
 
    def __init__(
        self,
        text_index,
        vector_index,
        embedder,
        llm_client,
        search_type="hybrid",
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model="claude-haiku-4-5-20251001",
    ):
        self.text_index = text_index
        self.vector_index = vector_index
        self.embedder = embedder
        self.llm_client = llm_client
        self.search_type = search_type
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model
 
    # --- retrieval methods ---
 
    def text_search(self, query, num_results=5):
        return self.text_index.search(query, num_results=num_results)
 
    def vector_search(self, query, num_results=5):
        q = self.embedder.encode(query)
        return self.vector_index.search(q, num_results=num_results)
 
    def hybrid_search(self, query, num_results=5, k=60):
        text_results = self.text_index.search(query, num_results=10)
        q = self.embedder.encode(query)
        vector_results = self.vector_index.search(q, num_results=10)
        return rrf([text_results, vector_results], k=k, num_results=num_results)
 
    def search(self, query, num_results=5):
        """Dispatch to the configured search_type (text / vector / hybrid)."""
        if self.search_type == "text":
            return self.text_search(query, num_results=num_results)
        if self.search_type == "vector":
            return self.vector_search(query, num_results=num_results)
        return self.hybrid_search(query, num_results=num_results)
 
    # --- prompting + LLM ---
 
    def build_context(self, search_results):
        lines = []
        for doc in search_results:
            lines.append(f"[{doc['doc_type']}] {doc['doc_id']}")
            lines.append(doc["content"])
            lines.append("")
        return "\n".join(lines).strip()
 
    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)
 
    def llm(self, prompt):
        response = self.llm_client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.instructions,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        return response
 
    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)
        return response.content[0].text
 
 
# ---------------------------------------------------------------------------
# Index building with embedding persistence (Option 2)
# ---------------------------------------------------------------------------
 
EMB_PATH = "embeddings.npy"
EMB_IDS_PATH = "embeddings_ids.json"
 
 
def _get_embeddings(documents, embedder, emb_path=EMB_PATH, ids_path=EMB_IDS_PATH):
    """
    Return an (n_docs x dim) matrix of embeddings for the documents.
 
    Loads from disk when the cached embeddings match the current documents
    (same doc_ids in the same order); otherwise recomputes and saves.
    """
    doc_ids = [d["doc_id"] for d in documents]
 
    if os.path.exists(emb_path) and os.path.exists(ids_path):
        with open(ids_path, encoding="utf-8") as f:
            cached_ids = json.load(f)
        if cached_ids == doc_ids:
            print(f"Loading cached embeddings ({len(doc_ids)} docs) from {emb_path}")
            return np.load(emb_path)
        print("Documents changed since last run - recomputing embeddings.")
 
    print(f"Computing embeddings for {len(doc_ids)} documents (one-time)...")
    texts = [d["content"] for d in documents]
    X = embedder.encode_batch(texts)
    X = np.asarray(X, dtype=np.float32)
 
    np.save(emb_path, X)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)
    print(f"Saved embeddings to {emb_path}")
    return X
 
 
def build_indexes(documents=None, embedder=None):
    """Build the text index and the vector index over the FIFA documents."""
    if documents is None:
        documents = load_documents()
    if embedder is None:
        embedder = Embedder()
 
    # Text index
    text_index = Index(
        text_fields=["content"],
        keyword_fields=["doc_id", "doc_type"],
    )
    text_index.fit(documents)
 
    # Vector index (embeddings persisted to disk)
    X = _get_embeddings(documents, embedder)
    vector_index = VectorSearch(keyword_fields=["doc_id", "doc_type"])
    vector_index.fit(X, documents)
 
    return text_index, vector_index, embedder
 
 
# ---------------------------------------------------------------------------
# Build-once cached rag object
# ---------------------------------------------------------------------------
 
_RAG = None
 
 
def get_rag(llm_client=None, search_type="hybrid"):
    """
    Return a cached RAGFifa instance, building both indexes only on first call.
    """
    global _RAG
    if _RAG is None:
        text_index, vector_index, embedder = build_indexes()
        _RAG = RAGFifa(
            text_index=text_index,
            vector_index=vector_index,
            embedder=embedder,
            llm_client=llm_client,
            search_type=search_type,
        )
    else:
        if llm_client is not None:
            _RAG.llm_client = llm_client
        _RAG.search_type = search_type
    return _RAG
 
 
if __name__ == "__main__":
    # Retrieval-only smoke test (no LLM needed)
    rag = get_rag()
    query = "Who was the player of the match in the Mexico South Africa game?"
    for stype in ["text", "vector", "hybrid"]:
        rag.search_type = stype
        print(f"\n=== {stype.upper()} SEARCH: {query} ===")
        for r in rag.search(query, num_results=3):
            print(f"  [{r['doc_type']}] {r['doc_id']}: {r['content'].splitlines()[0]}")
 