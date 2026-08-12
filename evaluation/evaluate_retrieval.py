"""
Evaluate retrieval quality for text, vector, and hybrid search.

For each ground-truth (question, doc_id) pair, run each retrieval method and
check where (if anywhere) the correct doc_id appears in the results. Reports
hit rate and MRR per method, and names the best.

Run from the PROJECT ROOT (after generate_ground_truth.py):
    uv run python evaluation/evaluate_retrieval.py
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from rag import get_rag


def hit_rate(relevance_total):
    return sum(any(rel) for rel in relevance_total) / len(relevance_total)


def mrr(relevance_total):
    total = 0.0
    for rel in relevance_total:
        for rank, is_rel in enumerate(rel):
            if is_rel:
                total += 1 / (rank + 1)
                break
    return total / len(relevance_total)


def evaluate(ground_truth, search_fn, num_results=5):
    relevance_total = []
    for q in tqdm(ground_truth, leave=False):
        results = search_fn(q["question"], num_results=num_results)
        relevance = [d["doc_id"] == q["doc_id"] for d in results]
        relevance_total.append(relevance)
    return hit_rate(relevance_total), mrr(relevance_total)


def load_ground_truth():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth.csv")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} ground-truth questions\n")

    rag = get_rag()  # no LLM client needed for retrieval evaluation

    methods = {
        "text": rag.text_search,
        "vector": rag.vector_search,
        "hybrid": rag.hybrid_search,
    }

    print(f"{'method':<10}{'hit_rate':>12}{'mrr':>12}")
    print("-" * 34)

    scores = {}
    for name, fn in methods.items():
        hr, mr = evaluate(ground_truth, fn)
        scores[name] = (hr, mr)
        print(f"{name:<10}{hr:>12.3f}{mr:>12.3f}")

    best = max(scores, key=lambda m: scores[m][1])  # rank by MRR
    print(f"\nBest method by MRR: {best} "
          f"(hit_rate={scores[best][0]:.3f}, mrr={scores[best][1]:.3f})")


if __name__ == "__main__":
    main()


"""
After running this script against the text, vector, and hybrid search methods, you 
can compare the hit rate and MRR results to determine which retrieval method performs best,
and make it your default search_type in get_rag()
"""