"""
RAG flow for the FIFA World Cup 2026 project.

Builds a minsearch text index over the documents produced by ingest_docs_for_rag.py,
and wraps retrieval + prompting + the LLM call in a RAGFifa class
(same shape as the RAGBase from LLMzoomcamp).

The index is built once via get_rag() and cached, so it is not rebuilt
on every query. In the Streamlit app this is wrapped again with
@st.cache_resource; here get_rag() uses a simple module-level cache so
scripts and notebooks get the same "build once" behaviour.

Usage:
    from rag import get_rag
    rag = get_rag()
    print(rag.rag("How did Mexico do in the group stage?"))
"""

from minsearch import Index

from data_and_ingestion.ingest_docs_for_rag import load_documents


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


class RAGFifa:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model="claude-haiku-4-5-20251001",
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        return self.index.search(query, num_results=num_results)

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
# Build-once index + rag object
# ---------------------------------------------------------------------------

def build_index(documents=None):
    """Build and fit a minsearch Index over the FIFA documents."""
    if documents is None:
        documents = load_documents()
    index = Index(
        text_fields=["content"],
        keyword_fields=["doc_id", "doc_type"],
    )
    index.fit(documents)
    return index


_RAG = None


def get_rag(llm_client=None):
    """
    Return a cached RAGFifa instance, building the index only on first call.

    Pass an llm_client (e.g. an OpenAI() client). If none is given, the object
    is still returned so you can call .search()/.build_prompt() without an LLM.
    """
    global _RAG
    if _RAG is None:
        index = build_index()
        _RAG = RAGFifa(index=index, llm_client=llm_client)
    elif llm_client is not None:
        _RAG.llm_client = llm_client
    return _RAG


if __name__ == "__main__":
    # Retrieval-only smoke test (no LLM needed)
    rag = get_rag()
    for q in [
        "How did Mexico do against South Africa?",
        "Who was the player of the match in the Mexico South Africa game?",
        "What is the capacity of Estadio Azteca?",
    ]:
        print(f"\n=== QUERY: {q} ===")
        results = rag.search(q, num_results=3)
        for r in results:
            first_line = r["content"].splitlines()[0]
            print(f"  [{r['doc_type']}] {r['doc_id']}: {first_line}")