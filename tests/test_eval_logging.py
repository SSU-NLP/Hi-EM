"""Unit tests for ``hi_em.eval_logging``."""

from __future__ import annotations

from hi_em.eval_logging import WandbRun, aggregate_summary, parse_judge_yes_no


def test_wandb_run_disabled_when_explicitly_off(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    wb = WandbRun(project="p", name="n", group="g", config={"a": 1},
                  sidecar_path=tmp_path / "sidecar")
    assert wb.enabled is False
    # All ops are no-ops; sidecar isn't written.
    wb.log({"x": 1}, step=1)
    wb.summary(y=2)
    wb.log_table("t", ["c"], [[1]])
    wb.finish()
    assert not (tmp_path / "sidecar").exists()


def test_wandb_run_falls_back_when_init_throws(monkeypatch, tmp_path) -> None:
    """Auth failure (no WANDB_API_KEY and no `wandb login`) → no-op, no crash."""
    monkeypatch.delenv("WANDB_MODE", raising=False)
    import hi_em.eval_logging as m
    orig = m.wandb
    if orig is None:
        return  # wandb not installed in this env

    class _StubWandb:
        @staticmethod
        def init(**_):
            raise RuntimeError("auth failed (test stub)")

    monkeypatch.setattr(m, "wandb", _StubWandb)
    wb = WandbRun(project="p", name="n", group="g", config={"a": 1},
                  sidecar_path=tmp_path / "sidecar")
    assert wb.enabled is False
    assert not (tmp_path / "sidecar").exists()


def test_aggregate_summary_empty_list_returns_empty() -> None:
    assert aggregate_summary([]) == {}


def test_aggregate_summary_basic_metrics() -> None:
    per_q = [
        {"accuracy": 1, "prefill_tokens": 100, "latency_sec": 1.0,
         "error": None, "question_type": "single-session-user"},
        {"accuracy": 0, "prefill_tokens": 200, "latency_sec": 2.0,
         "error": None, "question_type": "single-session-user"},
        {"accuracy": 1, "prefill_tokens": 300, "latency_sec": 3.0,
         "error": "RateLimit", "question_type": "multi-session"},
    ]
    s = aggregate_summary(per_q)
    assert s["n_questions"] == 3
    assert abs(s["accuracy_overall"] - (2 / 3)) < 1e-9
    assert s["prefill_tokens_avg"] == 200.0
    assert s["prefill_tokens_p50"] == 200.0
    assert s["latency_sec_avg"] == 2.0
    assert s["error_rate"] == 1 / 3
    assert s["accuracy_by_qtype/single-session-user"] == 0.5
    assert s["accuracy_by_qtype/multi-session"] == 1.0


def test_parse_judge_yes_no() -> None:
    # plain
    assert parse_judge_yes_no("yes") is True
    assert parse_judge_yes_no("No.") is False
    # closed think block stripped
    assert parse_judge_yes_no("<think>reasoning</think>yes") is True
    assert parse_judge_yes_no("<think>okay</think>\nThe answer is yes.") is True
    # unclosed think (max_tokens hit) — no extractable answer → safe False
    assert parse_judge_yes_no("<think>not closed thinking forever") is False
    # answer with explanation: take last yes/no
    assert parse_judge_yes_no("definitely no, the answer is wrong") is False
    assert parse_judge_yes_no("I think yes") is True
    # 'yes' inside closed think but no answer after → False
    assert parse_judge_yes_no("<think>so finally yes</think>") is False
    # garbage → False (safer than guessing)
    assert parse_judge_yes_no("???") is False
    assert parse_judge_yes_no("") is False


def test_aggregate_summary_topic_revisit_only_when_present() -> None:
    s_no = aggregate_summary([{"accuracy": 1}])
    assert "topic_revisit_hit_rate" not in s_no
    s_yes = aggregate_summary([
        {"accuracy": 1, "topic_revisit_hit": 1},
        {"accuracy": 0, "topic_revisit_hit": 0},
        {"accuracy": 1, "topic_revisit_hit": 1},
    ])
    assert s_yes["topic_revisit_hit_rate"] == 2 / 3
