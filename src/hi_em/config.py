"""HiEM 알고리즘 hyperparameter config loader.

Single source of truth: ``configs/hiem.json``. 자격증명/환경/model은
``.env`` (python-dotenv)이 담당.

Precedence (낮음 → 높음): module defaults < ``configs/hiem.json`` < CLI 인자.

Sections:
    segmenter         : sCRP + Gaussian (alpha, lmda, sigma0_sq)
    memory_window     : stateless baseline (k_topics, k_turns_per_topic)
    topic_importance  : Phase 2-Full 4 작용 (alpha[], lambda_r, lambda_freq, min_floor)
    stm               : Phase 2-Full STM (max_topics, max_turns, promotion_threshold)
    round             : Phase 2-Full round (turns_per_round)
    evaluation        : Phase 4 baseline budgets (sliding_k, rag_k)

JSON에서 ``_``로 시작하는 키는 주석 — loader가 자동 strip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "hiem.json"


_DEFAULTS: dict[str, dict[str, Any]] = {
    "segmenter": {"alpha": 1.0, "lmda": 10.0, "sigma0_sq": 0.01},
    "memory_window": {"k_topics": 3, "k_turns_per_topic": 5},
    "topic_importance": {
        "alpha": [1.0, 1.0, 1.0, 1.0],
        "lambda_r": 0.5,
        "lambda_freq": 0.5,
        "min_floor": 0.1,
    },
    "stm": {"max_topics": 10, "max_turns": 200, "promotion_threshold": 0.5},
    "round": {"turns_per_round": 20},
    "evaluation": {"sliding_k": 20, "rag_k": 10},
}


def _strip_comments(obj: Any) -> Any:
    """Recursively drop keys starting with '_' (documented-JSON sentinel)."""
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if not k.startswith("_")}
    return obj


def load_config(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """Return merged config: defaults overridden per-section by file values.

    File-missing or section-missing 모두 안전 — 누락된 항목은 default 사용.
    """
    out: dict[str, dict[str, Any]] = {k: dict(v) for k, v in _DEFAULTS.items()}
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if p.exists():
        raw = _strip_comments(json.loads(p.read_text()))
        for section, values in raw.items():
            if section in out and isinstance(values, dict):
                out[section].update(values)
            elif isinstance(values, dict):
                out[section] = dict(values)
    return out


def get_section(name: str, path: Path | str | None = None) -> dict[str, Any]:
    """Convenience accessor for a single section dict."""
    return load_config(path).get(name, {})
