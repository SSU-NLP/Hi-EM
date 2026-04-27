#!/usr/bin/env python3
"""Phase 3-3 smoke test for ``HiEM.handle_turn`` against a real LLM.

Scenario (A → B → A):
    Turn 1: 토픽 A (Kyoto 가을 여행)
    Turn 2: 토픽 B (파스타 레시피)
    Turn 3: 토픽 A로 복귀 (Kyoto 추가 질문)

기대 동작:
    - Turn 1, 3이 같은 ``topic_id``로 배정 (centroid cosine)
    - Turn 3의 prefill에 Turn 1 (user + assistant)가 포함
    - Turn 3의 Assistant 응답이 Turn 1의 정보(가을 여행 맥락)를 자연스럽게 사용

Usage:
    uv run python scripts/smoke_test_orchestrator.py --model openai/gpt-4o-mini
    uv run python scripts/smoke_test_orchestrator.py --model anthropic/claude-3.5-haiku
    uv run python scripts/smoke_test_orchestrator.py --model my-local-vllm-model

``.env`` (gitignored)에서 ``OPENAI_API_KEY`` + ``OPENAI_BASE_URL`` 자동 로드.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Silence "fork after parallelism" noise from HF tokenizers (must be set
# before sentence-transformers / transformers are imported).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hi_em.embedding import QueryEncoder  # noqa: E402
from hi_em.llm import OpenAIChatLLM  # noqa: E402
from hi_em.orchestrator import HiEM  # noqa: E402


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Drop ``<think>...</think>`` blocks from reasoning-style models (Qwen, etc.)."""
    return _THINK_RE.sub("", text).strip()


SCENARIO = [
    "I'm planning a trip to Kyoto next month. When is the best time to see autumn leaves?",
    "By the way, can you give me a quick recipe for tomato pasta?",
    "Going back to Kyoto — which neighborhoods are best for foliage walks?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="OpenRouter slug or vLLM model name")
    parser.add_argument("--system", default=None, help="Optional system prompt")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--k-topics", type=int, default=3)
    parser.add_argument("--k-turns-per-topic", type=int, default=5)
    parser.add_argument(
        "--ltm-root", default=str(REPO_ROOT / "data" / "ltm" / "smoke"),
        help="(Wiped before run for repeatability)",
    )
    parser.add_argument("--output", default=str(REPO_ROOT / "outputs" / "phase-3-smoke.md"))
    parser.add_argument(
        "--no-strip-think", action="store_true",
        help="Default strips <think>...</think> from LTM-stored assistant text "
             "(reduces next-turn prefill tokens for reasoning models like Qwen3).",
    )
    args = parser.parse_args()

    load_dotenv()
    base_url = os.environ.get("OPENAI_BASE_URL", "(SDK default = api.openai.com)")
    api_key_set = bool(os.environ.get("OPENAI_API_KEY"))
    print(f"[env] OPENAI_BASE_URL = {base_url}")
    print(f"[env] OPENAI_API_KEY  = {'set' if api_key_set else 'MISSING — abort'}")
    if not api_key_set:
        sys.exit(1)
    print(f"[args] model={args.model}  system={args.system!r}  max_tokens={args.max_tokens}")

    print("[encoder] loading bge-base-en-v1.5 ...")
    encoder = QueryEncoder()
    print(f"  device = {encoder.device}")

    # fresh LTM root for repeatability
    ltm_root = Path(args.ltm_root)
    if ltm_root.exists():
        shutil.rmtree(ltm_root)

    hi = HiEM(
        conv_id="smoke",
        encoder=encoder,
        llm=OpenAIChatLLM(),
        model=args.model,
        ltm_root=ltm_root,
        system_prompt=args.system,
        response_filter=None if args.no_strip_think else strip_think_tags,
        k_topics=args.k_topics,
        k_turns_per_topic=args.k_turns_per_topic,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    trace: list[str] = [
        f"# Phase 3-3 Smoke Test",
        "",
        f"- 시각: {datetime.now(timezone.utc).isoformat()}",
        f"- model: `{args.model}`",
        f"- base_url: `{base_url}`",
        f"- system_prompt: `{args.system!r}`",
        f"- HP: k_topics={args.k_topics}, k_turns_per_topic={args.k_turns_per_topic}, "
        f"temperature={args.temperature}, max_tokens={args.max_tokens}",
        "",
        "**Scenario**: A(Kyoto) → B(pasta) → A(Kyoto 복귀). "
        "Turn 3에서 Turn 1이 같은 topic으로 배정 + prefill에 포함되면 Hi-EM 정상 동작.",
        "",
    ]

    for i, user_text in enumerate(SCENARIO, 1):
        print(f"\n--- Turn {i} ---")
        print(f"USER: {user_text}")
        response = hi.handle_turn(user_text)
        snippet = response if len(response) <= 300 else response[:300] + "..."
        print(f"ASSISTANT: {snippet}")

        turns = hi._ltm.load_turns("smoke")
        last_user = turns[-2]
        state = hi._ltm.load_state("smoke")
        n_topics = len(state["topics"])
        counts = [t["count"] for t in state["topics"]]
        print(
            f"  → topic_id={last_user['topic_id']} "
            f"is_boundary={last_user['is_boundary']} "
            f"n_topics={n_topics} counts={counts}"
        )

        trace.extend([
            f"## Turn {i}",
            "",
            f"**User**: {user_text}",
            "",
            f"**Assistant**: {response}",
            "",
            f"**State**: topic_id=`{last_user['topic_id']}` "
            f"is_boundary=`{last_user['is_boundary']}` "
            f"n_topics={n_topics} counts={counts}",
            "",
        ])

    # Hi-EM 핵심 검증
    turns = hi._ltm.load_turns("smoke")
    user_turn_indices = [i for i, t in enumerate(turns) if t["role"] == "user"]
    t1_topic = turns[user_turn_indices[0]]["topic_id"]
    t3_topic = turns[user_turn_indices[2]]["topic_id"]
    revisit_ok = (t1_topic == t3_topic)

    print("\n=== Hi-EM core check ===")
    print(f"Turn 1 (Kyoto)        → topic_id = {t1_topic}")
    print(f"Turn 3 (Kyoto 복귀)   → topic_id = {t3_topic}")
    print(f"같은 topic으로 묶임: {revisit_ok} {'✓ PASS' if revisit_ok else '✗ FAIL'}")

    trace.extend([
        "## Hi-EM Core Verification",
        "",
        f"- Turn 1 (Kyoto) topic_id = **{t1_topic}**",
        f"- Turn 3 (Kyoto 복귀) topic_id = **{t3_topic}**",
        f"- 토픽 복귀 같은 ID: **{'✓ PASS' if revisit_ok else '✗ FAIL'}**",
        "",
        "**합격 기준**: Turn 1·3이 같은 topic_id로 묶이고, Turn 3 응답이 Turn 1의 Kyoto 맥락 (가을·시기)을 자연스럽게 이어받으면 통과. (응답 품질 평가는 사용자 육안 검증.)",
        "",
    ])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(trace))
    print(f"\ntrace → {out_path}")


if __name__ == "__main__":
    main()
