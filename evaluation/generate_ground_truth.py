"""
Generate a ground-truth evaluation set for retrieval.

For a layered sample of documents, ask Claude to write questions that
each document answers. Each (question, doc_id) pair becomes one ground-truth
row: the "correct" document that retrieval should return for that question.

Run from the PROJECT ROOT:
    uv run python evaluation/generate_ground_truth.py

Output: evaluation/ground_truth.csv  (columns: question, doc_id, doc_type)
"""

import os
import sys
import csv
import random
from concurrent.futures import ThreadPoolExecutor

# make the project root importable no matter where we're run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from dotenv import load_dotenv
from anthropic import Anthropic

from data_and_ingestion.ingest_docs_for_rag import load_documents

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
N_QUESTIONS = 3          # questions generated per document
MAX_WORKERS = 4          # parallel API calls

# How many documents to sample from each type (~300 total, all types represented)
SAMPLE_PER_TYPE = {
    "match": 80,
    "player": 180,
    "team": 30,
    "venue": 10,
}

client = Anthropic()

# Structured output via a forced tool call (Anthropic's clean way to get JSON)
QUESTIONS_TOOL = {
    "name": "record_questions",
    "description": "Record the list of generated questions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": f"A list of {N_QUESTIONS} distinct questions.",
            }
        },
        "required": ["questions"],
    },
}

PROMPT = """You are building an evaluation set for a FIFA World Cup 2026 search system.

Below is one document from the knowledge base. Generate {n} distinct, specific
questions a real user might type, where THIS document contains the answer.

Rules:
- Each question must be answerable from this document alone.
- Use natural phrasing, as a user would actually search.
- Do not mention document ids or the word "document".
- Make the questions genuinely different from each other.

DOCUMENT:
{content}
"""


def generate_for_doc(doc):
    """Return a list of {question, doc_id, doc_type} for one document."""
    try:
        prompt = PROMPT.format(n=N_QUESTIONS, content=doc["content"])
        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            tools=[QUESTIONS_TOOL],
            tool_choice={"type": "tool", "name": "record_questions"},
            messages=[{"role": "user", "content": prompt}],
        )
        tool_block = next(b for b in response.content if b.type == "tool_use")
        questions = tool_block.input["questions"]
        return [
            {"question": q, "doc_id": doc["doc_id"], "doc_type": doc["doc_type"]}
            for q in questions
        ]
    except Exception as e:
        # one bad document shouldn't kill the whole run
        print(f"  skipped {doc['doc_id']}: {e}")
        return []


def sample_documents(documents, seed=1):
    random.seed(seed)
    by_type = {}
    for d in documents:
        by_type.setdefault(d["doc_type"], []).append(d)
    sampled = []
    for dtype, n in SAMPLE_PER_TYPE.items():
        docs = by_type.get(dtype, [])
        sampled += random.sample(docs, min(n, len(docs)))
    return sampled


def main():
    documents = load_documents()
    sample = sample_documents(documents)
    print(f"Sampled {len(sample)} documents; generating {N_QUESTIONS} questions each...")

    records = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for result in tqdm(pool.map(generate_for_doc, sample), total=len(sample)):
            records.extend(result)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "doc_id", "doc_type"])
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved {len(records)} question/doc pairs to {out}")


if __name__ == "__main__":
    main()