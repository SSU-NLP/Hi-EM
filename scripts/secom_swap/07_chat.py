"""Chat (response generation) given retrieved memory.

Per SeCom's eval contract: for each (question, retrieved_text) pair, ask
the chat LLM to answer using the retrieved context. Writes
``sample["predictions"]: List[str]`` aligned with ``sample["questions"]``.

LLM = ``openai/gpt-4o-mini`` via Crts (configurable).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]


CHAT_PROMPT = """Below is a conversation history excerpt that may be relevant to the question. Use it to answer briefly and accurately.

[Conversation history]
{context}

[Question]
{question}

Answer:"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_path", required=True)
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not key or not base_url:
        sys.exit("OPENAI_API_KEY / OPENAI_BASE_URL missing in .env")

    import openai
    client = openai.OpenAI(api_key=key, base_url=base_url)

    data = []
    with open(args.load_path) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"n_conv: {len(data)}, model: {args.model}", flush=True)

    def call(prompt: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                content = r.choices[0].message.content or ""
                return content, time.perf_counter() - t0
            except Exception as e:
                if attempt == 2:
                    return f"[ERROR] {str(e)[:200]}", time.perf_counter() - t0
                time.sleep(2 * (attempt + 1))
        return "", 0.0

    total_calls = sum(len(s["questions"]) for s in data)
    print(f"total chat calls: {total_calls}", flush=True)

    results = []
    pbar = tqdm(total=total_calls, desc="chat")
    chat_times = []
    for sample in data:
        questions = sample["questions"]
        contexts = sample["retrieved_texts"]
        assert len(questions) == len(contexts)
        prompts = [
            CHAT_PROMPT.format(context=ctx, question=q)
            for ctx, q in zip(contexts, questions)
        ]
        preds = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(call, p): i for i, p in enumerate(prompts)}
            for fut in as_completed(futures):
                i = futures[fut]
                content, dt = fut.result()
                preds[i] = content
                chat_times.append(dt)
                pbar.update(1)
        sample["predictions"] = preds
        results.append(sample)
        with open(args.save_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pbar.close()

    print(f"\nchat done. avg latency = {sum(chat_times)/len(chat_times):.2f}s", flush=True)


if __name__ == "__main__":
    main()
