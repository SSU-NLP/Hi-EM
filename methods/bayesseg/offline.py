#!/usr/bin/env python3
"""BayesSeg **offline** (원본 SuperDialseg BayesSegmenter, whole-dialogue).

원본 = benchmarks/superdialseg .../bayesseg 의 modeling_bayesseg
.BayesSegmenter 동작 그대로(코드 복사 없이 빌드된 `segment` 스크립트
호출). 대화 **전체**를 stdin 으로 주고 `segment config/dp.config`
(SuperDialseg 가 박은 `-num-segs 7` 포함) 실행 → JSON 세그먼트 경계
인덱스. JVM cold-start 는 대화당 1회(offline 이라 무해).

online 판(methods/bayesseg/online.py = persistent JVM, native-K,
prefix-causal)과 **같은 데이터(Def-DTS 번들)·같은 metric(autoseg
Pk/WD+F1, Score)** → Hi-EM 내 offline↔online 직접 비교.
※ 원 SuperDialseg 논문 보고치(tiage .419/superseg .463/dialseg711
.614)는 그쪽 데이터·공식 metric — 본 표는 Hi-EM harness, 정확
일치 아님(방향·정상동작 검증용).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
DEFDTS = REPO / "benchmarks" / "Def-DTS"
BSEG = (REPO / "benchmarks" / "superdialseg" / "src" / "super_dialseg"
        / "models" / "bayesseg")
DATASETS = ["tiage", "dialseg711", "superseg"]


def _stub_anthropic() -> None:
    m = types.ModuleType("anthropic")
    m.Anthropic = lambda **kw: None
    sys.modules["anthropic"] = m


def _utts(dialogue: str) -> list[str]:
    return [s for s in dialogue.split("[NEWLINE]")
            if s.strip() not in ("[BOUNDARY]", "")]


def _clean(u: str) -> str:
    return u.replace("\t", " ").replace("\n", " ").strip() or "."


def _bayes_offline(utts, tmp_ref):
    """modeling_bayesseg.BayesSegmenter.forward 동일: 전체 발화 →
    `segment config/dp.config` → JSON 경계 인덱스(1-based, 세그먼트
    끝). 반환 per-utterance 0/1 (1=경계 AFTER, 마지막 0)."""
    with open(tmp_ref, "w") as fh:
        fh.writelines(_clean(u) + "\n" for u in utts)
    out = subprocess.run(
        ["bash", "-c",
         f'cat "{tmp_ref}" | ./segment config/dp.config'],
        cwd=str(BSEG), capture_output=True, text=True, timeout=120)
    raw = out.stdout.strip()
    try:
        idx = json.loads(raw)            # e.g. [2, 4, 6, 7, ...]
    except Exception:
        return None
    pred = [0] * len(utts)
    for i in idx:
        if 1 <= i <= len(utts):
            pred[i - 1] = 1
    if pred:
        pred[-1] = 0
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-05-20_bayesseg_offline")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--limit", type=int, default=0,
                    help="0=full test set; N=앞 N 대화")
    args = ap.parse_args()

    if not (BSEG / "classes" / "edu" / "mit" / "nlp" / "segmenter"
            / "SegTester.class").exists():
        sys.exit("bayesseg 미빌드 — ant build 먼저 (lib/+build.xml 필요)")

    _stub_anthropic()
    sys.path.insert(0, str(DEFDTS))
    os.chdir(DEFDTS)
    import src.autoseg as A  # noqa: E402

    exp = REPO / "outputs" / "experiments" / args.name
    exp.mkdir(parents=True, exist_ok=True)
    tmp_ref = "/tmp/_bayes_offline.ref"
    rows = []
    for ds in args.datasets:
        data = list(A.alternative_load_dataset(ds, "test"))
        if args.limit:
            data = data[: args.limit]
        preds, labels, miss, lat = [], [], 0, []
        for j, d in enumerate(data):
            utts = _utts(d["dialogue"])
            if len(utts) < 2:
                continue
            t0 = time.perf_counter()
            pf = _bayes_offline(utts, tmp_ref)
            lat.append((time.perf_counter() - t0) * 1000.0)
            if pf is None:
                miss += 1
                uttr = [False] * len(utts)
            else:
                uttr = [False] * len(utts)
                for i in range(len(utts) - 1):
                    if pf[i] == 1:
                        uttr[i + 1] = True
            uttr[0] = False
            pred = A.extract_pred(uttr)
            lbl, _ = A.extract_label(d["dialogue"].split("[NEWLINE]"), True)
            pred, lbl = A.align_pred_label(pred, lbl)
            preds.append(pred)
            labels.append(lbl)
            if (j + 1) % 50 == 0:
                print(f"  {ds} {j+1}/{len(data)} …")
        m = A.compute_metrics(preds, labels)
        score = 0.5 * m["f1"] + 0.25 * (1 - m["pk"]) + 0.25 * (1 - m["wd"])
        rows.append(dict(ds=ds, n=len(data), pk=m["pk"], wd=m["wd"],
                         f1=m["f1"], score=score, miss=miss,
                         lat_ms=float(np.mean(lat)) if lat else float("nan")))
        r = rows[-1]
        print(f"{ds:11s} n={r['n']:4d} Pk={r['pk']:.4f} WD={r['wd']:.4f} "
              f"F1={r['f1']:.4f} Score={r['score']:.4f} "
              f"lat/dial={r['lat_ms']:.0f}ms miss={miss}")

    L = ["# BayesSeg **offline** (원본 SuperDialseg BayesSegmenter)",
         "",
         "원본 modeling_bayesseg.BayesSegmenter 동작(`segment "
         "config/dp.config`, `-num-segs 7`, 대화 전체) 호출(코드 복사 X). "
         "data=Def-DTS 번들, metric=autoseg Pk/WD+F1, "
         "Score=0.5F1+0.25(1-Pk)+0.25(1-WD). online 판과 동일 harness.",
         f"limit={args.limit or 'full'}", "",
         "| dataset | n_dial | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | "
         "lat/dial(ms) | miss |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['ds']} | {r['n']} | {r['pk']:.4f} | {r['wd']:.4f} "
                 f"| {r['f1']:.4f} | {r['score']:.4f} | {r['lat_ms']:.0f} "
                 f"| {r['miss']} |")
    L += ["",
          "## 한계",
          "- 원 SuperDialseg 보고치(tiage .419/superseg .463/dialseg711 "
          ".614)는 그쪽 데이터·공식 metric — 본 표는 Hi-EM harness, "
          "정확 일치 아님(방향·정상동작 검증용).",
          "- `-num-segs 7` 은 SuperDialseg 가 박은 원본 설정 그대로(항상 "
          "7분할). 대화 8문장 미만이면 segment 크래시→miss 처리.",
          "- offline = 전체대화(미래 포함, JVM 대화당 1회). online 판 = "
          "methods/bayesseg/online.py (persistent JVM, native-K, prefix, "
          "AUXILIARY).",
          "- non-LLM CPU(+JVM), calls/turn=0 tok/turn=0."]
    (exp / "REPORT.md").write_text("\n".join(L) + "\n")
    print("report →", exp / "REPORT.md")


if __name__ == "__main__":
    main()
