"""LoCoMo benchmark loader: official locomo10.json → entry stream
compatible with run_experiment.py.

Maps the per-conversation, multi-QA structure of LoCoMo onto the
per-question, embedded-history structure that run_experiment.py expects
(LongMemEval shape). All N questions of a conversation share the same
``haystack_sessions`` payload — preload cost is paid per-question (same
trade-off as the LongMemEval pipeline; we keep parity for now).

Conversation → entries:
    * One entry per QA pair.
    * ``haystack_sessions`` = ordered list of session turn-lists. Each
      session's first turn carries a ``[<datetime>] `` prefix so the LLM
      and the segmenter both see the temporal stamp without polluting
      role-alternation.
    * Speaker → role: ``speaker_a → user``, ``speaker_b → assistant``.
      Content is prefixed with ``"<Speaker>: "`` so name semantics are
      preserved even when consecutive same-speaker turns force
      consecutive same-role messages (LLMs handle this gracefully; the
      role label is purely structural for prompting).

Per-category question augmentation (matches official LoCoMo eval, see
benchmarks/locomo/task_eval/gpt_utils.py):
    * cat 2 (temporal): append date-hint suffix.
    * cat 5 (adversarial): wrap as multi-choice (a/b shuffled
      deterministically per (sample_id, q_idx)) and persist the
      ``answer_key`` mapping for downstream judging.

Extra databases (``session_summary``, ``observation``) ride along on
each entry under ``extra_databases`` for the rag-summary /
rag-observation methods. They are NOT used by sliding/full/rag/hi-em*
methods.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CAT2NAME = {
    1: "multi-hop",
    2: "temporal-reasoning",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

CAT2_HINT = " Use DATE of CONVERSATION to answer with an approximate date."

NOT_MENTIONED = "Not mentioned in the conversation"


def _stable_seed(*parts: Any) -> int:
    """PYTHONHASHSEED-independent seed for reproducible cat-5 shuffling."""
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:16], 16)


def _session_indices(conversation: dict) -> list[int]:
    """Return session numbers in chronological (numeric) order."""
    nums = []
    for k in conversation:
        if k.startswith("session_") and not k.endswith("_date_time"):
            try:
                nums.append(int(k.split("_", 1)[1]))
            except ValueError:
                continue
    return sorted(nums)


def _build_haystack_sessions(conversation: dict) -> list[list[dict]]:
    """LoCoMo conversation → list[list[{role, content}]] in session order.

    Both speakers are emitted with ``role="user"`` and a ``"<Speaker>: "``
    content prefix. The chat-role label is structural; the meaningful
    speaker identity rides in ``content``. We do NOT alternate
    user/assistant by speaker because LoCoMo conversations are
    two-person transcripts, not assistant-vs-user dialogues, and the
    Hi-EM segmenter only embeds turns whose role is ``user``
    (``orchestrator.preload_history``). Mapping ``speaker_b`` to
    ``assistant`` would silently exclude half the dialog from
    segmentation, biasing topic clustering and Hi-EM retrieval. With
    every turn as ``user`` the segmenter sees the full conversation;
    sliding/full/rag and the OpenAI/vLLM chat protocol accept
    consecutive same-role messages.

    Each session's first turn is prefixed with ``[<datetime>] `` so the
    date is reachable to encoder + LLM; subsequent turns are bare.
    """
    sessions: list[list[dict]] = []
    for i in _session_indices(conversation):
        date = conversation.get(f"session_{i}_date_time", "")
        turns_raw = conversation[f"session_{i}"]
        out: list[dict] = []
        for j, t in enumerate(turns_raw):
            speaker = t["speaker"]
            text = t["text"]
            content = f"{speaker}: {text}"
            if j == 0 and date:
                content = f"[{date}] {content}"
            out.append({"role": "user", "content": content})
        sessions.append(out)
    return sessions


def _extract_summaries(sample: dict) -> list[dict]:
    """Per-session summary records for rag-summary."""
    conv = sample["conversation"]
    summaries = sample.get("session_summary", {}) or {}
    out: list[dict] = []
    for i in _session_indices(conv):
        text = summaries.get(f"session_{i}_summary")
        if not text:
            continue
        out.append({
            "session_idx": i,
            "date": conv.get(f"session_{i}_date_time", ""),
            "text": text,
        })
    return out


def _extract_observations(sample: dict) -> list[dict]:
    """Per-(session, speaker, observation) records for rag-observation."""
    conv = sample["conversation"]
    obs_root = sample.get("observation", {}) or {}
    out: list[dict] = []
    for i in _session_indices(conv):
        date = conv.get(f"session_{i}_date_time", "")
        sess_obs = obs_root.get(f"session_{i}_observation", {}) or {}
        for speaker, items in sess_obs.items():
            for item in items:
                # item format: [text, dia_id] or [text, [dia_id, ...]]
                if isinstance(item, list) and item:
                    text = item[0]
                    evidence = item[1] if len(item) > 1 else None
                else:
                    text = str(item)
                    evidence = None
                out.append({
                    "session_idx": i,
                    "date": date,
                    "speaker": speaker,
                    "text": text,
                    "evidence": evidence,
                })
    return out


def _augment_cat5(question: str, adversarial_answer: str, seed: int) -> tuple[str, dict]:
    """Build LoCoMo-official multi-choice prompt for category 5.

    Half of the time (a) is "Not mentioned ...", the other half (b) is.
    The answer_key maps the model's eventual ``a``/``b`` selection back
    to the underlying option text, so the judge can detect the
    "not-mentioned" answer regardless of position.
    """
    import random
    rng = random.Random(seed)
    if rng.random() < 0.5:
        a, b = NOT_MENTIONED, adversarial_answer
    else:
        a, b = adversarial_answer, NOT_MENTIONED
    augmented = (
        f"{question} Select the correct answer: (a) {a} (b) {b}."
    )
    return augmented, {"a": a, "b": b}


def build_entries(
    data_path: str | Path,
    *,
    include_extra_databases: bool = True,
) -> list[dict]:
    """Load locomo10.json → list of per-QA entries.

    Each entry has the shape consumed by run_experiment.phase_run plus a
    few LoCoMo-only fields (``category``, ``evidence``, ``answer_key``,
    ``extra_databases``).
    """
    samples = json.loads(Path(data_path).read_text())
    entries: list[dict] = []
    for sample in samples:
        sample_id = sample["sample_id"]
        haystack_sessions = _build_haystack_sessions(sample["conversation"])
        if include_extra_databases:
            extras = {
                "session_summaries": _extract_summaries(sample),
                "observations": _extract_observations(sample),
                "speaker_a": sample["conversation"]["speaker_a"],
                "speaker_b": sample["conversation"]["speaker_b"],
            }
        else:
            extras = {}
        for q_idx, qa in enumerate(sample["qa"]):
            cat = int(qa["category"])
            qtype = CAT2NAME.get(cat, f"cat-{cat}")
            question = qa["question"]
            answer_key: dict | None = None
            if cat == 2:
                question = question + CAT2_HINT
            elif cat == 5:
                # ``adversarial_answer`` is the false plant; some samples
                # may use ``answer`` instead — accept either.
                lure = qa.get("adversarial_answer") or qa.get("answer") or ""
                seed = _stable_seed(sample_id, q_idx)
                question, answer_key = _augment_cat5(qa["question"], str(lure), seed)
            answer = qa.get("answer", "")
            if cat == 3 and isinstance(answer, str):
                # Official metric splits cat-3 ground truth on ';' and
                # uses only the first segment. Pre-split here so the
                # judge never needs to know about it.
                answer = answer.split(";")[0].strip()
            entry = {
                "question_id": f"{sample_id}_q{q_idx}",
                "sample_id": sample_id,
                "question_type": qtype,
                "category": cat,
                "question": question,
                "answer": "" if answer is None else str(answer),
                "evidence": qa.get("evidence", []),
                "haystack_sessions": haystack_sessions,
            }
            if answer_key is not None:
                entry["answer_key"] = answer_key
            if extras:
                entry["extra_databases"] = extras
            entries.append(entry)
    return entries


def stratify_by_category(
    entries: list[dict], per_category: int, *, also_per_sample: bool = True,
) -> list[dict]:
    """Sanity sampler: ``per_category`` entries per (sample_id, category) if
    ``also_per_sample`` else ``per_category`` per category overall.

    Default (``also_per_sample=True``): 10 conv × 5 cat × per_category Q.
    """
    from collections import defaultdict
    if also_per_sample:
        bucket: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for e in entries:
            bucket[(e["sample_id"], e["category"])].append(e)
        out: list[dict] = []
        for key in sorted(bucket):
            out.extend(bucket[key][:per_category])
        return out
    bucket2: dict[int, list[dict]] = defaultdict(list)
    for e in entries:
        bucket2[e["category"]].append(e)
    out2: list[dict] = []
    for cat in sorted(bucket2):
        out2.extend(bucket2[cat][:per_category])
    return out2
