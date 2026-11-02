"""Retrieval evaluation on a hand-written question set (data/eval_questions.json).

Each question lists the section number(s) that actually answer it. Metrics: hit@1, hit@3, MRR.
"""
import json
from pathlib import Path

from rag import Retriever

QUESTIONS = Path(__file__).resolve().parent.parent / "data" / "eval_questions.json"


def evaluate(retriever, questions):
    ranks = []
    for q in questions:
        ranked = retriever.ranked_sections(q["question"], k=5)
        rank = next((i + 1 for i, s in enumerate(ranked) if s in q["answer_sections"]), None)
        ranks.append(rank)
    n = len(ranks)
    return {"n": n,
            "hit@1": sum(r == 1 for r in ranks) / n,
            "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
            "mrr": sum(1 / r for r in ranks if r) / n}, ranks


if __name__ == "__main__":
    qs = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    for name, kw in [("unigram", {"ngram_range": (1, 1)}), ("uni+bigram", {"ngram_range": (1, 2)})]:
        m, ranks = evaluate(Retriever.from_dir(**kw), qs)
        print(f"{name:11s} n={m['n']}  hit@1={m['hit@1']:.2f}  hit@3={m['hit@3']:.2f}  MRR={m['mrr']:.2f}")
        for q, r in zip(qs, ranks):
            if r != 1:
                print(f"   miss/low: rank={r}  {q['question']}")
