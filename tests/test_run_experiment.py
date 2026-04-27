"""End-to-end resume invariant tests for ``scripts/run_experiment.py``.

Coverage spans both ``--method sliding`` (LLM-only path, fastest to set up) and
``--method hi-em`` (the path R-11 will actually run, exercising encoder +
segmenter + LTM jsonl). The Hi-EM tests are critical because they're the only
ones that catch:
    - LTM jsonl double-append after resume (preload_history corruption)
    - segmenter state leakage across questions
    - encoder thread-safety regressions

SKILL §10 #13: reference run vs interrupt+resume must produce identical
``primary_metric``. We use deterministic fakes (LLM / encoder / tokenizer) so
the assertion is exact, not statistical.

Test matrix:
    - full round cycle completes → completed.json + per-round checkpoints
    - mid-round crash → only completed rounds have checkpoint.json; resume runs
      from the next round
    - already-completed experiment is a no-op on re-run
    - reference vs resumed produce identical round 2 summary

Coverage: SKILL §1 (layout), §3 (atomic), §4 (round cycle ordering), §5
(resume), §9.7 (checkpoint as truth), §9.8 (sanity check helper).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# --- Fakes (deterministic so reference/resume comparisons are exact) -----

class FakeLLM:
    """Drop-in replacement for OpenAIChatLLM. All calls deterministic."""

    def chat(self, messages, model, **kwargs):  # noqa: D401, ARG002
        last = messages[-1]["content"]
        # Judge prompt → always "yes" so accuracy=1.0 (deterministic baseline).
        if "Is the model response correct? Answer yes or no only" in last:
            return "yes"
        # Hypothesis: deterministic on the user message.
        return f"fake-answer-for-{abs(hash(last)) % 1000}"


def _make_dataset(path: Path, n: int) -> None:
    """Tiny LongMemEval-shaped dataset; question_type spread across two values
    so stratify could be exercised, though tests use plain --limit."""
    qtypes = ["single-session-user", "multi-session"]
    data = [
        {
            "question_id": f"q{i}",
            "question": f"What is fact_{i}?",
            "answer": f"fact_{i}",
            "question_type": qtypes[i % 2],
            "haystack_sessions": [[
                {"role": "user", "content": f"fact_{i} matters"},
                {"role": "assistant", "content": "noted"},
            ]],
        }
        for i in range(n)
    ]
    path.write_text(json.dumps(data))


class FakeEncoder:
    """Deterministic encoder for Hi-EM path tests. Hash-based vectors, internal
    lock matches real ``QueryEncoder`` interface (orchestrator expects it)."""

    def __init__(self):
        import threading as _t
        self.dim = 4
        self.device = "cpu"
        self._lock = _t.Lock()

    def encode(self, texts):
        import numpy as _np
        with self._lock:
            return _np.array([
                [(hash(t + str(i)) % 1000) / 1000.0 for i in range(self.dim)]
                for t in texts
            ], dtype=_np.float32)


def _patch_heavy_deps(monkeypatch, with_encoder: bool = False):
    """Replace LLM / tokenizer / wandb / dotenv with no-ops or fakes."""
    import run_experiment as re_mod
    monkeypatch.setattr(re_mod, "OpenAIChatLLM", lambda: FakeLLM())
    monkeypatch.setattr(re_mod, "count_prefill_tokens", lambda msgs, model: 0)
    if with_encoder:
        encoder = FakeEncoder()
        monkeypatch.setattr(re_mod, "QueryEncoder", lambda **kw: encoder)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(re_mod, "load_dotenv", lambda: None)


def _run_main(monkeypatch, *cli_args: str) -> None:
    """Invoke ``run_experiment.main()`` with synthetic argv."""
    import run_experiment as re_mod
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", *cli_args])
    re_mod.main()


# --- Tests --------------------------------------------------------------

def test_full_round_cycle_writes_all_artifacts(tmp_path, monkeypatch):
    _patch_heavy_deps(monkeypatch)
    data = tmp_path / "q.json"
    _make_dataset(data, n=4)
    results = tmp_path / "results"

    _run_main(
        monkeypatch,
        "--method", "sliding",
        "--data", str(data),
        "--questions-per-round", "2",   # → 2 rounds
        "--workers", "1",
        "--results-root", str(results),
        "--no-token-count",
        "--exp-id", "full_cycle",
    )

    exp = results / "experiments" / "full_cycle"
    # SKILL §1 layout
    assert (exp / "experiment.json").exists()
    assert (exp / "completed.json").exists()
    assert (exp / "checkpoints" / "latest.json").exists()
    for r in (1, 2):
        rd = exp / "rounds" / f"round_{r:03d}"
        assert (rd / "checkpoint.json").exists(), f"round {r} checkpoint missing"
        assert (rd / "summary.json").exists()
        assert (rd / "hypothesis.jsonl").exists()
        assert (rd / "judged.jsonl").exists()

    # Experiment-level summary (cross-round aggregation)
    exp_summary_path = exp / "summary.json"
    assert exp_summary_path.exists(), "experiment-level summary.json must be saved"
    s = json.loads(exp_summary_path.read_text())
    assert s["n_questions"] == 4
    assert s["n_rounds"] == 2
    assert s["primary_metric"] == 1.0  # FakeLLM judge always says yes → 100%
    assert s["accuracy_overall"] == 1.0
    # By question_type: 2 qtypes seen in dataset (single-session-user, multi-session)
    assert s["accuracy_by_qtype/single-session-user"] == 1.0
    assert s["accuracy_by_qtype/multi-session"] == 1.0


def test_midround_crash_leaves_round1_complete_round2_incomplete(tmp_path, monkeypatch):
    _patch_heavy_deps(monkeypatch)
    data = tmp_path / "q.json"
    _make_dataset(data, n=4)
    results = tmp_path / "results"

    import run_experiment as re_mod
    orig = re_mod.phase_run
    counter = {"n": 0}

    def crash_on_second_call(*a, **kw):
        counter["n"] += 1
        if counter["n"] == 2:
            raise RuntimeError("simulated mid-round-2 crash")
        return orig(*a, **kw)

    monkeypatch.setattr(re_mod, "phase_run", crash_on_second_call)

    with pytest.raises(RuntimeError, match="simulated"):
        _run_main(
            monkeypatch,
            "--method", "sliding",
            "--data", str(data),
            "--questions-per-round", "2",
            "--workers", "1",
            "--results-root", str(results),
            "--no-token-count",
            "--exp-id", "crash_mid",
        )

    exp = results / "experiments" / "crash_mid"
    assert (exp / "rounds" / "round_001" / "checkpoint.json").exists()
    # Round 2 must NOT have a checkpoint — phase_run raised before mark_round.
    assert not (exp / "rounds" / "round_002" / "checkpoint.json").exists()
    assert not (exp / "completed.json").exists()


def test_resume_completes_remaining_rounds(tmp_path, monkeypatch):
    """After a crash, re-running the same exp_id picks up at round 2 and
    finishes the experiment cleanly."""
    _patch_heavy_deps(monkeypatch)
    data = tmp_path / "q.json"
    _make_dataset(data, n=4)
    results = tmp_path / "results"

    # 1) crash mid round 2
    import run_experiment as re_mod
    orig = re_mod.phase_run
    counter = {"n": 0}

    def crash_on_second(*a, **kw):
        counter["n"] += 1
        if counter["n"] == 2:
            raise RuntimeError("crash")
        return orig(*a, **kw)

    monkeypatch.setattr(re_mod, "phase_run", crash_on_second)
    with pytest.raises(RuntimeError):
        _run_main(
            monkeypatch,
            "--method", "sliding",
            "--data", str(data),
            "--questions-per-round", "2",
            "--workers", "1",
            "--results-root", str(results),
            "--no-token-count",
            "--exp-id", "resume_run",
        )

    # 2) un-patch + resume
    monkeypatch.setattr(re_mod, "phase_run", orig)
    _run_main(
        monkeypatch,
        "--method", "sliding",
        "--data", str(data),
        "--questions-per-round", "2",
        "--workers", "1",
        "--results-root", str(results),
        "--no-token-count",
        "--exp-id", "resume_run",
    )

    exp = results / "experiments" / "resume_run"
    assert (exp / "rounds" / "round_002" / "checkpoint.json").exists()
    assert (exp / "completed.json").exists()


def test_completed_experiment_is_idempotent_on_rerun(tmp_path, monkeypatch):
    _patch_heavy_deps(monkeypatch)
    data = tmp_path / "q.json"
    _make_dataset(data, n=4)
    results = tmp_path / "results"

    args = [
        "--method", "sliding",
        "--data", str(data),
        "--questions-per-round", "2",
        "--workers", "1",
        "--results-root", str(results),
        "--no-token-count",
        "--exp-id", "idempotent",
    ]
    _run_main(monkeypatch, *args)

    exp = results / "experiments" / "idempotent"
    r1 = json.loads((exp / "rounds" / "round_001" / "summary.json").read_text())
    completed_before = (exp / "completed.json").read_text()

    # Re-run: should detect completion and skip without overwriting.
    _run_main(monkeypatch, *args)

    r1_after = json.loads((exp / "rounds" / "round_001" / "summary.json").read_text())
    completed_after = (exp / "completed.json").read_text()
    assert r1 == r1_after
    # completed.json may rewrite timestamp; the key field "total_rounds" must match.
    assert json.loads(completed_before)["total_rounds"] == json.loads(completed_after)["total_rounds"]


def test_reference_run_equals_interrupted_then_resumed(tmp_path, monkeypatch):
    """The headline invariant (SKILL §10 #13): kill -9 mid-round-2, then resume,
    must yield the same round 2 summary as a single uninterrupted run."""
    _patch_heavy_deps(monkeypatch)
    data = tmp_path / "q.json"
    _make_dataset(data, n=4)

    # Reference: clean run.
    ref_root = tmp_path / "ref"
    _run_main(
        monkeypatch,
        "--method", "sliding",
        "--data", str(data),
        "--questions-per-round", "2",
        "--workers", "1",
        "--results-root", str(ref_root),
        "--no-token-count",
        "--exp-id", "ref",
    )
    ref_summary = json.loads(
        (ref_root / "experiments" / "ref" / "rounds" / "round_002" / "summary.json").read_text()
    )

    # Interrupt path.
    res_root = tmp_path / "res"
    import run_experiment as re_mod
    orig = re_mod.phase_run
    counter = {"n": 0}

    def crash_on_second(*a, **kw):
        counter["n"] += 1
        if counter["n"] == 2:
            raise RuntimeError("crash")
        return orig(*a, **kw)

    monkeypatch.setattr(re_mod, "phase_run", crash_on_second)
    with pytest.raises(RuntimeError):
        _run_main(
            monkeypatch,
            "--method", "sliding",
            "--data", str(data),
            "--questions-per-round", "2",
            "--workers", "1",
            "--results-root", str(res_root),
            "--no-token-count",
            "--exp-id", "res",
        )
    monkeypatch.setattr(re_mod, "phase_run", orig)
    _run_main(
        monkeypatch,
        "--method", "sliding",
        "--data", str(data),
        "--questions-per-round", "2",
        "--workers", "1",
        "--results-root", str(res_root),
        "--no-token-count",
        "--exp-id", "res",
    )
    res_summary = json.loads(
        (res_root / "experiments" / "res" / "rounds" / "round_002" / "summary.json").read_text()
    )

    # Determinism: primary_metric MUST match exactly.
    assert ref_summary["primary_metric"] == res_summary["primary_metric"]
    assert ref_summary["accuracy_overall"] == res_summary["accuracy_overall"]
    # n_processed identical too (round 2 sees the same 2 questions).
    assert ref_summary["n_processed"] == res_summary["n_processed"]


# --- Hi-EM specific path (R-11 actually runs this) ----------------------

def test_hi_em_method_full_cycle(tmp_path, monkeypatch):
    """Hi-EM end-to-end (Codex Q7): catches issues --method sliding can't see.

    The Hi-EM path exercises the encoder + segmenter + LTM jsonl write — none
    of which fire under sliding/full/rag. This is the path R-11 will run, so
    it must be e2e tested before paying for the 1-2h full-500 run.
    """
    _patch_heavy_deps(monkeypatch, with_encoder=True)
    data = tmp_path / "q.json"
    _make_dataset(data, n=4)
    results = tmp_path / "results"

    _run_main(
        monkeypatch,
        "--method", "hi-em",
        "--data", str(data),
        "--questions-per-round", "2",
        "--workers", "1",
        "--results-root", str(results),
        "--no-token-count",
        "--exp-id", "hiem_full",
    )

    exp = results / "experiments" / "hiem_full"
    assert (exp / "completed.json").exists()
    s = json.loads((exp / "summary.json").read_text())
    # FakeLLM judge always says yes → 100% with deterministic input.
    assert s["primary_metric"] == 1.0
    # Hi-EM-specific metric should be present in cross-round summary.
    assert "topic_revisit_hit_rate" in s
    # Per-question LTM dirs should be created under working_state.
    ltm_root = exp / "working_state" / "ltm"
    assert ltm_root.exists()
    assert any(ltm_root.iterdir()), "expected at least one conv_id LTM dir"


def test_hi_em_resume_no_stale_ltm_leak(tmp_path, monkeypatch):
    """Codex Q3 fix: resume must not double-append LTM jsonl.

    If ``run_hi_em`` doesn't wipe ``ltm_root / conv_id`` before reconstructing,
    a resumed round's first question will have its history written twice in
    the per-question LTM jsonl, contaminating prefill and topic state.
    Reference run vs interrupt+resume MUST produce byte-identical metrics.
    """
    _patch_heavy_deps(monkeypatch, with_encoder=True)
    data = tmp_path / "q.json"
    _make_dataset(data, n=4)

    # Reference (clean run)
    ref_root = tmp_path / "ref"
    _run_main(
        monkeypatch,
        "--method", "hi-em",
        "--data", str(data),
        "--questions-per-round", "2",
        "--workers", "1",
        "--results-root", str(ref_root),
        "--no-token-count",
        "--exp-id", "hiem_ref",
    )
    ref_summary = json.loads(
        (ref_root / "experiments" / "hiem_ref" / "summary.json").read_text()
    )

    # Interrupt + resume
    res_root = tmp_path / "res"
    import run_experiment as re_mod
    orig = re_mod.phase_run
    counter = {"n": 0}

    def crash_on_second(*a, **kw):
        counter["n"] += 1
        if counter["n"] == 2:
            raise RuntimeError("crash")
        return orig(*a, **kw)

    monkeypatch.setattr(re_mod, "phase_run", crash_on_second)
    with pytest.raises(RuntimeError):
        _run_main(
            monkeypatch,
            "--method", "hi-em",
            "--data", str(data),
            "--questions-per-round", "2",
            "--workers", "1",
            "--results-root", str(res_root),
            "--no-token-count",
            "--exp-id", "hiem_res",
        )
    monkeypatch.setattr(re_mod, "phase_run", orig)
    _run_main(
        monkeypatch,
        "--method", "hi-em",
        "--data", str(data),
        "--questions-per-round", "2",
        "--workers", "1",
        "--results-root", str(res_root),
        "--no-token-count",
        "--exp-id", "hiem_res",
    )
    res_summary = json.loads(
        (res_root / "experiments" / "hiem_res" / "summary.json").read_text()
    )

    # Hi-EM-specific assertions: any stale LTM leak shifts these.
    assert ref_summary["primary_metric"] == res_summary["primary_metric"]
    assert ref_summary["accuracy_overall"] == res_summary["accuracy_overall"]
    if "topic_revisit_hit_rate" in ref_summary:
        assert ref_summary["topic_revisit_hit_rate"] == res_summary["topic_revisit_hit_rate"]
