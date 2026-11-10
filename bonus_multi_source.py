"""Bonus: question 12 needs two documents (OF-E5 and OF-E7).
Compares one search for the whole question with two searches, one per car.

  python bonus_multi_source.py          # retrieval only, free
  python bonus_multi_source.py --ask    # also ask Claude (needs ANTHROPIC_API_KEY, a fraction of a cent)
"""
import argparse

from eval_retrieval import build_index
from rag.search import multi_search, search

QUESTION = "לאיזה רכב חשמלי של החברה יש טווח נסיעה ארוך יותר, ובכמה?"
SUB_QUERIES = ["טווח נסיעה של אופק E5", "טווח נסיעה של אופק E7"]  # one per car
NEEDLES = ["420 ק״מ", "510 ק״מ"]


def report(title, chunks):
    text = "\n".join(c.text for c in chunks)
    found = [n for n in NEEDLES if n in text]
    print(f"\n{title}")
    print(f"  chunks: {[c.id for c in chunks]}")
    print(f"  found {len(found)}/{len(NEEDLES)}: {found}  {'✓' if len(found) == len(NEEDLES) else '✗'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--ask", action="store_true")
    args = ap.parse_args()

    index = build_index("headings")
    single = search(index, QUESTION, "hybrid", args.k, current_only=True)
    multi = multi_search(index, SUB_QUERIES, "hybrid", args.k, current_only=True)
    report("one search, the whole question:", single)
    report(f"two searches: {SUB_QUERIES}", multi)

    if args.ask:
        from rag.answer import ask, verify_citations
        answer, usage = ask(QUESTION, multi)
        bad = verify_citations(answer, multi)
        print(f"\nClaude: {answer.answer}")
        for c in answer.citations:
            print(f"  {'✗' if c in bad else '✓'} {c.chunk_id}: \"{c.quote}\"")
        print(f"  insufficient_context={answer.insufficient_context}")


if __name__ == "__main__":
    main()
