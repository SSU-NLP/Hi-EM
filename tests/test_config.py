"""Unit tests for ``hi_em.config`` (single hyperparameter source-of-truth)."""

from __future__ import annotations

import json
from pathlib import Path

from hi_em.config import get_section, load_config


def test_load_config_no_file_returns_module_defaults(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "missing.json")
    assert cfg["segmenter"] == {"alpha": 1.0, "lmda": 10.0, "sigma0_sq": 0.01}
    assert cfg["evaluation"] == {"sliding_k": 20, "rag_k": 10}
    assert cfg["round"] == {"turns_per_round": 20}


def test_load_config_with_file_overrides_section(tmp_path: Path) -> None:
    p = tmp_path / "h.json"
    p.write_text(json.dumps({
        "segmenter": {"alpha": 5.0, "lmda": 2.0},   # partial: sigma0_sq omitted
        "evaluation": {"rag_k": 7},                  # partial
    }))
    cfg = load_config(p)
    assert cfg["segmenter"]["alpha"] == 5.0
    assert cfg["segmenter"]["lmda"] == 2.0
    assert cfg["segmenter"]["sigma0_sq"] == 0.01     # default kept
    assert cfg["evaluation"]["rag_k"] == 7
    assert cfg["evaluation"]["sliding_k"] == 20      # default kept


def test_load_config_strips_underscore_keys(tmp_path: Path) -> None:
    """Documented JSON: keys starting with '_' are comments."""
    p = tmp_path / "h.json"
    p.write_text(json.dumps({
        "_comment": "top-level comment",
        "segmenter": {"_doc": "section comment", "alpha": 2.0},
    }))
    cfg = load_config(p)
    assert "_comment" not in cfg
    assert "_doc" not in cfg["segmenter"]
    assert cfg["segmenter"]["alpha"] == 2.0


def test_get_section_shortcut(tmp_path: Path) -> None:
    p = tmp_path / "h.json"
    p.write_text(json.dumps({"stm": {"max_topics": 99}}))
    sec = get_section("stm", p)
    assert sec["max_topics"] == 99
    # other stm defaults preserved
    assert sec["max_turns"] == 200


def test_get_section_unknown_returns_empty() -> None:
    assert get_section("does_not_exist") == {}


def test_repo_default_config_loadable() -> None:
    """The shipped configs/hiem.json must parse and have all expected sections."""
    cfg = load_config()
    for section in ("segmenter", "memory_window", "topic_importance",
                    "stm", "round", "evaluation"):
        assert section in cfg, f"missing section: {section}"
