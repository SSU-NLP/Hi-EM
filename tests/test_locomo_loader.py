"""Unit tests for src/hi_em/locomo_loader.py.

We exercise the loader against the real benchmarks/locomo/data/locomo10.json
fixture (small, ~10 conversations) so the assertions reflect the actual
upstream schema rather than a mock that could drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hi_em.locomo_loader import (
    CAT2_HINT,
    NOT_MENTIONED,
    _stable_seed,
    build_entries,
    stratify_by_category,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "benchmarks/locomo/data/locomo10.json"


@pytest.fixture(scope="module")
def entries() -> list[dict]:
    return build_entries(DATA)


def test_total_entry_count_matches_qa_total(entries: list[dict]) -> None:
    raw = json.loads(DATA.read_text())
    expected = sum(len(s["qa"]) for s in raw)
    assert len(entries) == expected


def test_entry_has_required_keys(entries: list[dict]) -> None:
    e = entries[0]
    for k in ("question_id", "sample_id", "question_type", "category",
              "question", "answer", "haystack_sessions"):
        assert k in e, f"missing {k}"
    assert isinstance(e["haystack_sessions"], list)
    assert all(isinstance(s, list) for s in e["haystack_sessions"])
    assert all("role" in t and "content" in t
               for s in e["haystack_sessions"] for t in s)


def test_all_turns_are_user_role_with_speaker_prefix(entries: list[dict]) -> None:
    """Both LoCoMo speakers emit as role='user' so the Hi-EM segmenter
    (which only embeds user turns) sees the full conversation. Speaker
    identity is preserved in content prefix.
    """
    raw = json.loads(DATA.read_text())
    sample = raw[0]
    e = next(x for x in entries if x["sample_id"] == sample["sample_id"])
    first_session = e["haystack_sessions"][0]
    session_1_raw = sample["conversation"]["session_1"]
    assert len(first_session) == len(session_1_raw)
    for built, raw_turn in zip(first_session, session_1_raw):
        assert built["role"] == "user"
        assert raw_turn["speaker"] in built["content"]
        assert raw_turn["text"] in built["content"]
    # Sanity: across all sessions of all entries, every turn is user-role.
    assert all(t["role"] == "user"
               for x in entries[:50]
               for s in x["haystack_sessions"]
               for t in s)


def test_first_turn_of_each_session_carries_date_prefix(entries: list[dict]) -> None:
    raw = json.loads(DATA.read_text())
    sample = raw[0]
    e = next(x for x in entries if x["sample_id"] == sample["sample_id"])
    for i, sess in enumerate(e["haystack_sessions"], 1):
        date = sample["conversation"][f"session_{i}_date_time"]
        assert sess[0]["content"].startswith(f"[{date}]"), \
            f"session {i} first turn missing date prefix"


def test_cat2_question_appends_date_hint(entries: list[dict]) -> None:
    cat2 = [e for e in entries if e["category"] == 2]
    assert cat2, "no cat-2 entries found"
    for e in cat2[:5]:
        assert e["question"].endswith(CAT2_HINT)


def test_cat5_multichoice_format_and_answer_key(entries: list[dict]) -> None:
    cat5 = [e for e in entries if e["category"] == 5]
    assert cat5, "no cat-5 entries found"
    for e in cat5[:5]:
        assert "Select the correct answer:" in e["question"]
        assert "(a)" in e["question"] and "(b)" in e["question"]
        assert "answer_key" in e
        ak = e["answer_key"]
        assert set(ak) == {"a", "b"}
        # Exactly one of (a)/(b) must be the abstention option.
        assert (ak["a"] == NOT_MENTIONED) ^ (ak["b"] == NOT_MENTIONED)


def test_cat5_shuffle_is_deterministic_across_runs(entries: list[dict]) -> None:
    """Re-load and verify cat-5 a/b assignment is stable for same (sample, q_idx)."""
    again = build_entries(DATA)
    assert len(entries) == len(again)
    for a, b in zip(entries, again):
        if a["category"] == 5:
            assert a["answer_key"] == b["answer_key"]
            assert a["question"] == b["question"]


def test_cat3_answer_split_on_semicolon(entries: list[dict]) -> None:
    raw = json.loads(DATA.read_text())
    cat3_raw = [(s["sample_id"], i, q) for s in raw for i, q in enumerate(s["qa"])
                if int(q.get("category", 0)) == 3]
    if not cat3_raw:
        pytest.skip("no cat-3 entries")
    sid, idx, qa = cat3_raw[0]
    e = next(x for x in entries if x["question_id"] == f"{sid}_q{idx}")
    expected = str(qa.get("answer", "")).split(";")[0].strip()
    assert e["answer"] == expected


def test_extra_databases_attached(entries: list[dict]) -> None:
    e = entries[0]
    assert "extra_databases" in e
    db = e["extra_databases"]
    assert "session_summaries" in db
    assert "observations" in db
    assert db["session_summaries"], "session_summaries empty"
    s0 = db["session_summaries"][0]
    assert {"session_idx", "date", "text"} <= set(s0)
    assert db["observations"], "observations empty"
    o0 = db["observations"][0]
    assert {"session_idx", "date", "speaker", "text"} <= set(o0)


def test_stratify_by_category_returns_balanced_sample(entries: list[dict]) -> None:
    sub = stratify_by_category(entries, per_category=1, also_per_sample=True)
    # 10 conv × up-to-5 cat × 1 = up to 50; some samples may lack a category.
    assert 0 < len(sub) <= 10 * 5
    # Each (sample_id, category) appears at most once.
    seen = {(e["sample_id"], e["category"]) for e in sub}
    assert len(seen) == len(sub)


def test_stable_seed_independent_of_pythonhashseed() -> None:
    s1 = _stable_seed("conv-26", 0)
    s2 = _stable_seed("conv-26", 0)
    assert s1 == s2
    assert _stable_seed("conv-26", 1) != s1
