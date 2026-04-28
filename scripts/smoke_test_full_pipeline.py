#!/usr/bin/env python3
"""Smoke test for Phase 2-Full STM + RoundProcessor pipeline.

Drives a 25-turn synthetic conversation with A→B→A topic switching, prints
the full trace (per-turn STM hit/miss + round trigger), and asserts a
handful of structural invariants:

* Topic atomicity — STM never holds a partial topic
* Round triggers fire exactly at every ``2*round_size`` jsonl rows
* After a round, STM topics ⊆ LTM topics (subset)
* Eviction obeys ``max_topics`` cap

Runs against the real vLLM endpoint configured via ``.env`` (no judge model;
this is a structural test, not a quality test). Use::

    uv run python scripts/smoke_test_full_pipeline.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hi_em.embedding import QueryEncoder  # noqa: E402
from hi_em.llm import OpenAIChatLLM  # noqa: E402
from hi_em.orchestrator import HiEM  # noqa: E402


TOPIC_A = [
    "What's the best way to roast brussels sprouts?",
    "How long should I roast them at 425F?",
    "Should I toss them with anything besides olive oil?",
    "Can I add bacon during roasting?",
    "What about adding balsamic glaze at the end?",
]

TOPIC_B = [
    "I'm planning a trip to Kyoto in fall.",
    "When does the maple foliage usually peak?",
    "Which neighborhoods are best for evening walks?",
    "Are temple gardens crowded on weekdays?",
    "Any food specialties I should not miss in Kyoto?",
]


def build_turn_schedule() -> list[str]:
    """A→B→A→B→A interleave so STM hit/miss path exercises both."""
    out: list[str] = []
    for i in range(5):
        out.append(TOPIC_A[i])
        out.append(TOPIC_B[i])
    # back-to-A revisit at end (5 more turns) to force STM hit
    for i in range(5):
        out.append(TOPIC_A[i])
    return out


def main() -> None:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("[fatal] OPENAI_API_KEY missing in .env"); sys.exit(1)

    print(f"[env] model={os.environ.get('HIEM_MODEL', 'Qwen/Qwen3-8B')}")
    print(f"[env] base_url={os.environ.get('OPENAI_BASE_URL', '(SDK default)')}")

    # ``.env`` may carry an inline-comment value (e.g. ``HIEM_DEVICE=  # auto``);
    # treat anything that doesn't look like a real device string as auto.
    raw_device = os.environ.get("HIEM_DEVICE", "").strip()
    device = raw_device if raw_device in {"cuda", "mps", "cpu"} else None
    encoder = QueryEncoder(device=device)
    llm = OpenAIChatLLM()

    ltm_root = REPO_ROOT / "outputs" / "_smoke_p2f_ltm"
    if ltm_root.exists():
        import shutil
        shutil.rmtree(ltm_root)

    hi = HiEM(
        conv_id="smoke",
        encoder=encoder,
        llm=llm,
        model=os.environ.get("HIEM_MODEL", "Qwen/Qwen3-8B"),
        ltm_root=ltm_root,
        use_stm=True,
        round_size=5,                # round every 5 user turns (= 10 jsonl rows)
        stm_max_topics=3,
        stm_max_turns=200,
        promotion_threshold=0.0,     # promote everything for visibility
        round_async=False,           # sync rounds for deterministic trace
        temperature=0.7,
        max_tokens=80,               # short replies — speed matters here
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    schedule = build_turn_schedule()
    print(f"[plan] {len(schedule)} turns, round_size=5 (trigger every 10 jsonl rows)")
    print()

    round_triggers: list[bool] = []
    stm_hits: list[bool] = []
    for i, text in enumerate(schedule):
        _, debug = hi.handle_turn(text, return_debug=True)
        topic_id = debug["topic_id"]
        stm_hit = debug["stm_hit"]
        triggered = debug["round_triggered"]
        stm_topics = sorted(hi.stm.current_topics())
        stm_turns = hi.stm.total_turns()
        marker = "TRIG" if triggered else "    "
        print(
            f"[turn {i:2d}] topic={topic_id} stm_hit={int(stm_hit)} "
            f"prefill_msgs={len(debug['messages'])} "
            f"STM={stm_topics}({stm_turns}t) {marker}"
        )
        round_triggers.append(triggered)
        stm_hits.append(stm_hit)

    print()
    print("[invariants]")

    # 1. round triggers fire at expected positions (turn idx 4, 9, 14, 19, 24)
    expected = [(i + 1) % 5 == 0 for i in range(len(schedule))]
    assert round_triggers == expected, (
        f"round trigger schedule mismatch:\n  expected={expected}\n  got     ={round_triggers}"
    )
    print(f"  ✓ round triggers fire at every 5 user turns (got {sum(round_triggers)} fires)")

    # 2. topic atomicity — every STM-resident topic has same turn count as LTM
    final_state = hi._ltm.load_state("smoke")
    ltm_topic_ids = {t["topic_id"] for t in final_state["topics"]}
    stm_ids = hi.stm.current_topics()
    assert stm_ids.issubset(ltm_topic_ids), (
        f"STM has unknown topics: {stm_ids - ltm_topic_ids}"
    )
    for tid in stm_ids:
        stm_turns_for_topic = hi.stm.get(tid)
        ltm_for_topic = hi._ltm.load_turns("smoke", topic_id=tid)
        assert len(stm_turns_for_topic) == len(ltm_for_topic), (
            f"topic {tid} atomicity violation: STM={len(stm_turns_for_topic)} vs LTM={len(ltm_for_topic)}"
        )
    print(f"  ✓ topic atomicity: every topic in STM matches LTM full count")

    # 3. capacity cap respected
    assert len(stm_ids) <= 3, f"max_topics=3 violated: got {len(stm_ids)}"
    print(f"  ✓ max_topics=3 cap respected ({len(stm_ids)} in STM)")

    # 4. revisit hits — last 5 turns are topic A again, STM should hit by then
    revisit_hits = sum(stm_hits[-5:])
    assert revisit_hits >= 3, (
        f"expected ≥3 STM hits in final A-revisit window, got {revisit_hits}"
    )
    print(f"  ✓ {revisit_hits}/5 STM hits in final A-revisit (≥3 expected)")

    print()
    print(f"[final] STM topics={sorted(stm_ids)} / LTM topics={sorted(ltm_topic_ids)}")
    print(f"[final] round_idx={hi.round_processor.round_idx}")
    print("[done] all invariants pass")


if __name__ == "__main__":
    main()
