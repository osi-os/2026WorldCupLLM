"""
LLM/prompt evaluation: compare three system-prompt variants for the
generation step, using Claude as an LLM-as-judge.

For a sample of ground-truth questions, each prompt variant runs the full
RAG flow (hybrid retrieval + generation). Each generated answer is then
scored by a judge model as RELEVANT / PARTLY_RELEVANT / NON_RELEVANT.
The variant with the best mean score wins.

The judge is GROUNDED: it sees the same retrieved context the answer was
generated from, and is instructed to judge ONLY against that context and
never against its own prior/world knowledge. This prevents the judge's
training cutoff (which predates the 2026 tournament) from wrongly failing
answers that are correct according to the dataset.

Run from the PROJECT ROOT (after generate_ground_truth.py):
    uv run python evaluation/llm_judge.py

Output: evaluation/llm_eval_results.csv  (per-answer judgements)
"""

import os
import sys
import csv
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from dotenv import load_dotenv
from anthropic import Anthropic

from rag import get_rag, RAGFifa

load_dotenv()

JUDGE_MODEL = "claude-haiku-4-5-20251001"
N_QUESTIONS = 50
MAX_WORKERS = 4

# --- The three prompt variants under test ---
PROMPT_VARIANTS = {
    "baseline": (
        "You're a FIFA World Cup 2026 assistant. You answer questions about the "
        "tournament using only the provided context, which contains match results, "
        "player stats, team records, and venues. Use the context to find relevant "
        "information and give an accurate, specific answer. If the answer is not in "
        'the context, say "I don\'t have that information in the tournament data."'
    ),
    "concise": (
        "You are a FIFA World Cup 2026 assistant. Answer the question using ONLY the "
        "provided context. Be direct and concise: give just the facts that answer the "
        "question (names, numbers, dates) in one or two sentences, with no preamble. "
        "If the context doesn't contain the answer, say you don't have that information."
    ),
    "detailed": (
        "You are a knowledgeable FIFA World Cup 2026 analyst. Using ONLY the provided "
        "context, give a complete, well-organized answer. Include the specific "
        "supporting details from the context (scores, xG, minutes, player and team "
        "names) that back up your answer. If the answer isn't in the context, clearly "
        "state that you don't have that information rather than guessing."
    ),
}

# --- Judge setup (structured output via forced tool call) ---
JUDGE_TOOL = {
    "name": "record_judgement",
    "description": "Record the relevance judgement for the answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "relevance": {
                "type": "string",
                "enum": ["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"],
            },
            "explanation": {"type": "string"},
        },
        "required": ["relevance", "explanation"],
    },
}

JUDGE_PROMPT = """You are evaluating an answer produced by a FIFA World Cup 2026
assistant. The tournament data is provided below as CONTEXT.

Judge the answer ONLY against the CONTEXT provided. This is critical:
- Treat the CONTEXT as the sole source of truth. It reflects what actually
  happened in the tournament.
- Do NOT use any outside or prior knowledge about football, players, or the
  World Cup. Do NOT reason about what you think is real or plausible.
- Do NOT penalize an answer for facts you personally doubt. If the answer
  matches the CONTEXT, it is correct, full stop.

Classify how well the answer addresses the question, given the context:
- RELEVANT: correctly answers the question and is consistent with the context.
- PARTLY_RELEVANT: partially answers, or is missing key detail from the context.
- NON_RELEVANT: does not answer, contradicts the context, or the context does
  not contain the needed information.

QUESTION: {question}

CONTEXT:
{context}

ANSWER: {answer}
"""

SCORE_MAP = {"RELEVANT": 1.0, "PARTLY_RELEVANT": 0.5, "NON_RELEVANT": 0.0}

client = Anthropic()


def judge_answer(question, answer, context):
    try:
        response = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=512,
            tools=[JUDGE_TOOL],
            tool_choice={"type": "tool", "name": "record_judgement"},
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=question, answer=answer, context=context)}],
        )
        block = next(b for b in response.content if b.type == "tool_use")
        return block.input["relevance"], block.input["explanation"]
    except Exception as e:
        return "NON_RELEVANT", f"[judge error: {e}]"


def answer_question(rag_variant, question):
    """Return (answer_text, retrieved_context) so the judge can be grounded."""
    try:
        results = rag_variant.search(question)
        context = rag_variant.build_context(results)
        prompt = rag_variant.build_prompt(question, results)
        response = rag_variant.llm(prompt)
        return response.content[0].text, context
    except Exception as e:
        return f"[generation error: {e}]", ""


def run_variant(name, rag_variant, sample):
    """Generate answers (with their context) then judge them, for one variant."""
    answers = [None] * len(sample)
    contexts = [None] * len(sample)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(answer_question, rag_variant, q["question"]): i
                for i, q in enumerate(sample)}
        for fut in tqdm(as_completed(futs), total=len(sample),
                        desc=f"{name}: answering", leave=False):
            i = futs[fut]
            answers[i], contexts[i] = fut.result()

    rows = [None] * len(sample)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(judge_answer, sample[i]["question"], answers[i], contexts[i]): i
                for i in range(len(sample))}
        for fut in tqdm(as_completed(futs), total=len(sample),
                        desc=f"{name}: judging", leave=False):
            i = futs[fut]
            relevance, explanation = fut.result()
            rows[i] = {
                "variant": name,
                "question": sample[i]["question"],
                "answer": answers[i],
                "relevance": relevance,
                "explanation": explanation,
            }
    return rows


def summarize(all_rows):
    """Aggregate mean score and label counts per variant."""
    by_variant = {}
    for r in all_rows:
        by_variant.setdefault(r["variant"], []).append(r["relevance"])

    summary = {}
    for name, labels in by_variant.items():
        mean_score = sum(SCORE_MAP[l] for l in labels) / len(labels)
        counts = {k: labels.count(k) for k in SCORE_MAP}
        summary[name] = {"mean_score": mean_score, "counts": counts, "n": len(labels)}
    return summary


def main():
    # Load and sample ground-truth questions
    gt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth.csv")
    with open(gt_path, encoding="utf-8") as f:
        ground_truth = list(csv.DictReader(f))
    random.seed(1)
    sample = random.sample(ground_truth, min(N_QUESTIONS, len(ground_truth)))
    print(f"Evaluating {len(PROMPT_VARIANTS)} prompt variants on {len(sample)} questions\n")

    # Build one shared set of indexes, then a RAG per variant (share indexes)
    base = get_rag(llm_client=client, search_type="hybrid")
    variants = {
        name: RAGFifa(
            text_index=base.text_index,
            vector_index=base.vector_index,
            embedder=base.embedder,
            llm_client=client,
            search_type="hybrid",
            instructions=instr,
        )
        for name, instr in PROMPT_VARIANTS.items()
    }

    all_rows = []
    for name, rag_variant in variants.items():
        all_rows.extend(run_variant(name, rag_variant, sample))

    # Save per-answer results for the writeup
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_eval_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["variant", "question", "answer", "relevance", "explanation"])
        writer.writeheader()
        writer.writerows(all_rows)

    # Report
    summary = summarize(all_rows)
    print(f"\n{'variant':<12}{'mean_score':>12}{'RELEVANT':>10}{'PARTLY':>8}{'NON':>6}")
    print("-" * 48)
    for name, s in sorted(summary.items(), key=lambda kv: -kv[1]["mean_score"]):
        c = s["counts"]
        print(f"{name:<12}{s['mean_score']:>12.3f}"
              f"{c['RELEVANT']:>10}{c['PARTLY_RELEVANT']:>8}{c['NON_RELEVANT']:>6}")

    best = max(summary, key=lambda n: summary[n]["mean_score"])
    print(f"\nBest prompt variant: {best} (mean_score={summary[best]['mean_score']:.3f})")
    print(f"Per-answer results saved to {out}")


if __name__ == "__main__":
    main()