"""Run-scoped chat: answers questions about THIS run only, from its ledger and report.
Retrieval picks the relevant ledger entries; the LLM (if any) may phrase the answer, and every
number it writes must come from those entries."""
from __future__ import annotations

import re

from app.ledger import Ledger
from app.llm import BaseLLM, narrate
from app.session import Session


def _docs(s: Session) -> list[dict]:
    docs = []
    for e in Ledger(s.id).entries():
        r = e.recommendation
        docs.append({"kind": "ledger", "step": e.step, "decision_point": e.decision_point, "chosen": e.chosen,
                     "text": f"step {e.step} {e.decision_point.replace('_', ' ').replace(':', ' ')} chosen {e.chosen}. "
                             f"{r.recommendation}. {r.rationale}. {e.note or ''} decided by {e.decided_by}",
                     "confidence": r.confidence})
    if s.audit_report:
        for f in s.audit_report.findings:
            docs.append({"kind": "audit", "text": f"audit {f.issue_type.replace('_', ' ')} column {f.column}: {f.rationale}. "
                                                  f"action {f.recommended_action}", "column": f.column, "severity": f.severity})
    md = s.results.get("report_markdown") or ""
    for para in re.split(r"\n\s*\n", md):
        if len(para) > 40:
            docs.append({"kind": "report", "text": para[:1200]})
    return docs


def answer(s: Session, question: str, llm: BaseLLM, k: int = 4) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = _docs(s)
    if not docs:
        return {"answer": "This run has no decisions recorded yet.", "source": "template", "cited": []}
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    mat = vec.fit_transform([d["text"] for d in docs] + [question])
    sims = (mat[:-1] @ mat[-1].T).toarray().ravel()
    order = [i for i in sims.argsort()[::-1][:k] if sims[i] > 0] or list(range(min(k, len(docs))))
    top = [docs[i] for i in order]
    fallback = "From this run's record: " + " ".join(
        (f"Step {d['step']} ({d['decision_point']}): {d['text'].split('. ', 1)[-1]}" if d["kind"] == "ledger" else d["text"])
        for d in top[:3])
    text, source = narrate(llm, f"Answer this question about the run using only the context: {question}",
                           {"question": question, "context": top}, fallback)
    return {"answer": text, "source": source,
            "cited": [{"kind": d["kind"], "step": d.get("step"), "decision_point": d.get("decision_point")} for d in top]}
