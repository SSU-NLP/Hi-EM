"""Unit tests for src/hi_em/locomo_judge.py — must match the numeric
behavior of benchmarks/locomo/task_eval/evaluation.py for the metrics we
expose (cat 1/2/3/4 F1, cat 5 abstention).
"""

from __future__ import annotations

import pytest

from hi_em.locomo_judge import (
    decode_cat5_choice,
    f1_multi_answer,
    f1_score,
    is_abstention,
    normalize_answer,
    score_one,
)


# --- normalization -------------------------------------------------------

def test_normalize_strips_punct_articles_lowercase() -> None:
    assert normalize_answer("The Quick, brown FOX!") == "quick brown fox"


def test_normalize_handles_extra_whitespace() -> None:
    assert normalize_answer("  hello   world  ") == "hello world"


# --- F1 ------------------------------------------------------------------

def test_f1_exact_match_is_one() -> None:
    assert f1_score("hello world", "Hello, world!") == pytest.approx(1.0)


def test_f1_zero_when_no_overlap() -> None:
    assert f1_score("blue sky", "loud music") == 0.0


def test_f1_partial_overlap() -> None:
    # 2 shared / 3 pred / 4 gold → P=2/3, R=2/4, F1=4/7≈0.571
    val = f1_score("blue sky cloud", "blue sky over there")
    assert 0.5 < val < 0.65


def test_f1_handles_empty() -> None:
    assert f1_score("", "anything") == 0.0
    assert f1_score("anything", "") == 0.0


# --- multi-answer F1 (cat 1) --------------------------------------------

def test_multi_answer_f1_full_match() -> None:
    val = f1_multi_answer("apple, banana", "banana, apple")
    assert val == pytest.approx(1.0)


def test_multi_answer_f1_partial() -> None:
    # gold = ["apple", "banana", "cherry"], pred = ["apple", "banana"]
    # per-gold: max F1 → apple=1, banana=1, cherry=0; mean = 2/3
    val = f1_multi_answer("apple, banana", "apple, banana, cherry")
    assert val == pytest.approx(2 / 3, abs=1e-3)


def test_multi_answer_f1_zero_no_overlap() -> None:
    assert f1_multi_answer("dog, cat", "fish, bird") == 0.0


# --- cat 5 abstention ----------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Not mentioned in the conversation", True),
    ("not mentioned anywhere", True),
    ("There is no information available", True),
    ("The answer is purple bananas", False),
    ("", False),
])
def test_is_abstention(text: str, expected: bool) -> None:
    assert is_abstention(text) is expected


def test_decode_cat5_choice_letter_only() -> None:
    key = {"a": "Not mentioned in the conversation", "b": "purple bananas"}
    assert decode_cat5_choice("a", key) == "Not mentioned in the conversation"
    assert decode_cat5_choice("B", key) == "purple bananas"


def test_decode_cat5_choice_paren_format() -> None:
    key = {"a": "Not mentioned in the conversation", "b": "purple bananas"}
    assert decode_cat5_choice("(a)", key) == "Not mentioned in the conversation"
    assert decode_cat5_choice("(B)", key) == "purple bananas"


def test_decode_cat5_choice_passthrough_freetext() -> None:
    key = {"a": "Not mentioned in the conversation", "b": "purple bananas"}
    free = "I don't see this information"
    assert decode_cat5_choice(free, key) == free


@pytest.mark.parametrize("text,expected_key", [
    ("(a) Not mentioned in the conversation", "a"),
    ("(b) purple bananas because reasons", "b"),
    ("a) some explanation", "a"),
    ("A: long form answer", "a"),
    ("B. another long answer", "b"),
    ("  (A)  trailing whitespace ok", "a"),
])
def test_decode_cat5_choice_leading_marker_in_long_answer(
    text: str, expected_key: str,
) -> None:
    """The official upstream maps single-char/paren outputs only. We
    extend that to longer outputs that lead with a recognizable a/b
    marker — a common Qwen-style answer shape ('(a) Not mentioned in
    the conversation') would otherwise drop to raw substring matching
    and miscount when the lure happens to live on the (a) side.
    """
    key = {"a": "Not mentioned in the conversation", "b": "purple bananas"}
    assert decode_cat5_choice(text, key) == key[expected_key]


def test_score_one_cat5_long_answer_with_marker_decodes() -> None:
    """Worst-case from the audit: model answers '(a) Not mentioned ...'
    when (a) is the abstention. Decode → resolve to abstention text →
    abstention detected → 1.0."""
    key = {"a": "Not mentioned in the conversation", "b": "purple bananas"}
    out = score_one(category=5, prediction="(a) Not mentioned in the conversation",
                    answer="purple bananas", answer_key=key)
    assert out["f1"] == 1.0
    assert out["abstention_correct"] is True
    # And the inverse — model picks the lure with a leading marker.
    out2 = score_one(category=5, prediction="(b) purple bananas, definitely",
                     answer="purple bananas", answer_key=key)
    assert out2["f1"] == 0.0
    assert out2["abstention_correct"] is False


# --- score_one router ---------------------------------------------------

def test_score_one_cat1_uses_multi_answer_f1() -> None:
    out = score_one(category=1, prediction="apple, banana",
                    answer="banana, apple")
    assert out["f1"] == pytest.approx(1.0)


@pytest.mark.parametrize("cat", [2, 3, 4])
def test_score_one_cat234_uses_single_f1(cat: int) -> None:
    out = score_one(category=cat, prediction="hello world",
                    answer="hello world")
    assert out["f1"] == pytest.approx(1.0)


def test_score_one_cat5_decodes_then_checks_abstention() -> None:
    key = {"a": "Not mentioned in the conversation", "b": "purple bananas"}
    out = score_one(category=5, prediction="(a)", answer="purple bananas",
                    answer_key=key)
    assert out["f1"] == 1.0
    assert out["abstention_correct"] is True

    out2 = score_one(category=5, prediction="b", answer="purple bananas",
                     answer_key=key)
    assert out2["f1"] == 0.0
    assert out2["abstention_correct"] is False


def test_score_one_unknown_category_raises() -> None:
    with pytest.raises(ValueError):
        score_one(category=99, prediction="x", answer="y")
