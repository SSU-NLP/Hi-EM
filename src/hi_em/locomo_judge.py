"""LoCoMo official F1 judge — port of benchmarks/locomo/task_eval/evaluation.py.

Pure-Python, no LLM call. Per-category metric:
    * cat 1 (multi-hop): split prediction & gold on ',', max-F1 across
      sub-answers, mean over gold sub-answers (``f1_multi_answer``).
    * cat 2/3/4 (temporal/open-domain/single-hop): single-string token
      F1 with porter stemming (``f1_score``).
    * cat 5 (adversarial): "Not mentioned" / "no information available"
      detection on the model's free-text output. If the model used the
      LoCoMo-official multi-choice setup, the caller must first map the
      model's a/b selection back through ``answer_key`` (see
      ``decode_cat5_choice``) before passing the resolved string here.

We deliberately do NOT depend on the upstream module: it pulls in
``bert_score``, ``nltk``, ``rouge``, ``regex`` which aren't part of our
runtime. The numeric behavior of the functions we DO need (token
normalization, porter-stemmed F1, multi-answer F1) is reproduced
faithfully — verified by tests/test_locomo_judge.py against fixed
fixtures.
"""

from __future__ import annotations

import re
import string
from collections import Counter

# --- normalization (matches benchmarks/locomo/task_eval/evaluation.py) ---

_ARTICLES_RE = re.compile(r"\b(a|an|the|and)\b")


def normalize_answer(s: str) -> str:
    s = s.replace(",", "")
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES_RE.sub(" ", s)
    s = " ".join(s.split())
    return s


# --- porter stemming (required for parity with official metric) ----------
# The upstream metric (benchmarks/locomo/task_eval/evaluation.py) uses
# nltk.PorterStemmer. nltk is declared in pyproject; importing here will
# fail fast if it is not installed rather than silently producing
# divergent F1 numbers.
from nltk.stem import PorterStemmer as _PorterStemmer  # noqa: E402

_stemmer = _PorterStemmer()


def _stem(w: str) -> str:
    return _stemmer.stem(w)


# --- F1 -------------------------------------------------------------------

def f1_score(prediction: str, ground_truth: str) -> float:
    """Token F1 with porter-stemmed normalization.

    Matches benchmarks/locomo/task_eval/evaluation.py:f1_score.
    """
    pred_tokens = [_stem(w) for w in normalize_answer(prediction).split()]
    gold_tokens = [_stem(w) for w in normalize_answer(ground_truth).split()]
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_multi_answer(prediction: str, ground_truth: str) -> float:
    """Cat 1 multi-hop F1: split both sides on ',', max-F1 across pred
    sub-answers per gold sub-answer, mean over gold sub-answers.

    Matches benchmarks/locomo/task_eval/evaluation.py:f1.
    """
    preds = [p.strip() for p in prediction.split(",") if p.strip()]
    golds = [g.strip() for g in ground_truth.split(",") if g.strip()]
    if not preds or not golds:
        return 0.0
    scores: list[float] = []
    for gt in golds:
        scores.append(max(f1_score(p, gt) for p in preds))
    return sum(scores) / len(scores)


# --- cat 5 adversarial ---------------------------------------------------

CAT5_ABSTAIN_MARKERS = ("no information available", "not mentioned")


def is_abstention(prediction: str) -> bool:
    """LoCoMo-official cat-5 detection: pass if the (resolved) model
    answer mentions one of the abstention markers (case-insensitive).
    """
    p = prediction.lower()
    return any(m in p for m in CAT5_ABSTAIN_MARKERS)


_CAT5_LEAD_RE = re.compile(
    r"^\s*(?:\(?\s*([ab])\s*\)?)\s*(?:[:.\-)\]]\s*|\s+|$)",
    re.IGNORECASE,
)


def decode_cat5_choice(prediction: str, answer_key: dict[str, str]) -> str:
    """Map the model's free-text (a)/(b) selection back to the option
    text. Generalizes benchmarks/locomo/task_eval/gpt_utils.py:get_cat_5_answer
    to also handle longer outputs that lead with an option marker
    (``"(a) Not mentioned ..."``, ``"a: explanation"``, ``"A. ..."``).

    When the prediction starts with a recognizable a/b marker we resolve
    to the option text; otherwise we return the raw prediction so the
    abstention check can fall back to substring matching.
    """
    p = prediction.strip()
    if not p:
        return prediction
    p_low = p.lower()
    # Exact short forms first (cheap, matches upstream get_cat_5_answer).
    if p_low == "a":
        return answer_key["a"]
    if p_low == "b":
        return answer_key["b"]
    if len(p_low) == 3 and p_low.startswith("(") and p_low.endswith(")"):
        if "a" in p_low:
            return answer_key["a"]
        if "b" in p_low:
            return answer_key["b"]
        return prediction
    # Leading marker in a longer answer: "(a) text" / "a) text" / "a: ..." / "A. ..."
    m = _CAT5_LEAD_RE.match(p)
    if m:
        choice = m.group(1).lower()
        if choice in answer_key:
            return answer_key[choice]
    return prediction


# --- public API ----------------------------------------------------------

def score_one(
    *,
    category: int,
    prediction: str,
    answer: str,
    answer_key: dict[str, str] | None = None,
) -> dict:
    """Score one prediction. Returns ``{f1, abstention_correct, resolved}``.

    ``f1`` is the LoCoMo metric for the given category (cat 5 returns a
    0/1 abstention flag in the f1 slot — official scripts do the same
    so that all five categories average to a single 'accuracy' number).
    ``resolved`` is the model output AFTER cat-5 a/b decoding (or the
    raw prediction otherwise) so callers can log what was scored.
    """
    if category in (2, 3, 4):
        return {"f1": f1_score(prediction, answer), "resolved": prediction}
    if category == 1:
        return {"f1": f1_multi_answer(prediction, answer), "resolved": prediction}
    if category == 5:
        if answer_key is not None:
            resolved = decode_cat5_choice(prediction, answer_key)
        else:
            resolved = prediction
        ok = is_abstention(resolved)
        return {"f1": 1.0 if ok else 0.0, "abstention_correct": ok,
                "resolved": resolved}
    raise ValueError(f"Unknown LoCoMo category: {category}")
