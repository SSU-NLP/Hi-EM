# Archive — 2026-04-26 Baseline

**아카이브 시점**: 2026-04-27
**보존 사유**: Phase 4-Re (research-experiment-infrastructure 적용) 환경 전환 직전의 모든 raw 결과 + LTM state 영구 보존.

## 1. 디렉토리 구조

```
archive/2026-04-26-baseline/
├── README.md          (이 파일)
├── outputs/           기존 outputs/ 통째 이동 (Phase 0~4 모든 산출물)
└── ltm/               기존 data/ltm/ 통째 이동 (Phase 4 평가 시점의 conv별 LTM state)
```

## 2. 포함된 결과 요약 (Phase 별)

### Phase 0
- `outputs/benchmark-analysis.md` — 4 벤치마크 분석

### Phase 1 (Topic 경계 감지)
- `outputs/phase-1-topiocqa.md` — Phase 1-3/1-4 메인 결과 (Hi-EM F1=0.471 PASS marginal)
- `outputs/phase-1-topiocqa-{sweep,variants,multisignal,anchors,encoder,contextualized}.json` — 7-iter 탐색 raw
- `outputs/phase-1-tiage.md` — Phase 1-5 (Hi-EM 두 HP 모두 패배, Gate FAIL)
- `outputs/phase-1-tiage-sweep.json` — TIAGE 108 configs sweep (모두 cosine baseline 미달)
- `outputs/phase-1-clustering-quality.{json,md}` — V-measure/NMI/ARI 분석 (boundary F1 ↔ ARI trade-off 발견)

### Phase 3 (orchestrator smoke test)
- `outputs/phase-3-smoke.md` — vLLM + Qwen3-8B A→B→A 시나리오 PASS

### Phase 4 (downstream QA, oracle 30 sanity + 500 full)
- `outputs/phase-4-sanity{,-nothink}-{sliding,full,rag,hi-em}.jsonl{,.judged.jsonl,.wandb-run-id}`
- `outputs/phase-4-full-{sliding,full,rag,hi-em}.jsonl{,.judged.jsonl,.wandb-run-id}`
- `outputs/phase-4-sanity{,-nothink}.md` — 분석 리포트

## 3. Phase 4 결과 표 (보존)

### 3.1 Thinking ON (α=1, λ=10, σ²=0.01, sanity 30)

| Method | Overall | knowledge | multi | ssa | ssp | ssu | temporal |
|---|---|---|---|---|---|---|---|
| sliding | 0.700 | 0.80 | 0.20 | 1.00 | 0.80 | 0.80 | 0.60 |
| full | 0.833 | 0.80 | 0.60 | 1.00 | 1.00 | 0.80 | 0.80 |
| rag | 0.733 | 0.60 | 0.20 | 1.00 | 0.80 | 0.80 | 1.00 |
| **Hi-EM (1/10/0.01 + sanity)** | 0.633 | 0.20 | 0.00 | 1.00 | 0.80 | 1.00 | 0.80 |

### 3.2 Thinking OFF — sanity 30 / full 500

| Method | Overall | knowledge | multi | ssa | ssp | ssu | temporal |
|---|---|---|---|---|---|---|---|
| sliding (sanity / full) | 0.867 / 0.658 | 0.80 / 0.75 | 0.60 / 0.36 | 1.00 / 0.93 | 1.00 / 1.00 | 1.00 / 0.94 | 0.80 / 0.54 |
| full (sanity / full) | 0.900 / 0.712 | 0.80 / 0.81 | 0.60 / 0.55 | 1.00 / 0.98 | 1.00 / 0.93 | 1.00 / 0.88 | 1.00 / 0.54 |
| rag (sanity / full) | 0.867 / 0.692 | 0.80 / 0.83 | 0.60 / 0.56 | 1.00 / 0.89 | 1.00 / 0.83 | 1.00 / 0.84 | 0.80 / 0.52 |
| **Hi-EM (1/10/0.01 + full)** | **0.562** | 0.51 | **0.23** | 0.91 | 0.97 | 0.81 | 0.46 |
| **Hi-EM (10/1/0.01 + sanity)** | 0.767 | 0.60 | 0.20 | 1.00 | 1.00 | 0.80 | 1.00 |
| **Hi-EM (10/1/0.1 + sanity)** | 0.700 | 0.40 | 0.20 | 0.80 | 1.00 | 1.00 | 0.80 |
| **Hi-EM (1/10/0.1 + sanity)** | 0.700 | 0.40 | 0.20 | 1.00 | 1.00 | 0.80 | 0.80 |
| **Hi-EM (1/10/0.01 + sanity, 재측정)** | 0.733 | 0.40 | 0.20 | 1.00 | 1.00 | 1.00 | 0.80 |

## 4. 결정적 관찰

1. **Hi-EM은 4 method 중 일관 꼴찌** (full 500 기준 0.562 vs baseline 0.658~0.712)
2. **multi-session에서 모든 HP 0.20~0.23 그대로** — sCRP HP 4 regime sweep으로 풀리지 않음
3. **`gpt4_59c863d7` (history 24 turn)이 어떤 σ²/α/λ 조합에서도 n_topics=1로 collapse** — embedding 자체 한계 (HP 문제 아님)
4. **Sample noise (sanity 30)**: 같은 HP 두 측정에서 ±0.033~0.133 변동 (`temperature=0.7` 영향)
5. **Hi-EM 강점**: ssp 0.97 (full 500, 4 method 중 1위), ssu/ssa는 baseline 동등

## 5. Lost from disk (chat history values만 보존)

prefix `phase-4-sanity-*` 가 여러 번 덮어써져 다음 hi-em 측정 raw jsonl은 lost:
- `Hi-EM (10/1/0.01 + sanity)`: 0.767 / 0.60 / 0.20 / 1.00 / 1.00 / 0.80 / 1.00
- `Hi-EM (10/1/0.1 + sanity)`: 0.700 / 0.40 / 0.20 / 0.80 / 1.00 / 1.00 / 0.80
- `Hi-EM (1/10/0.1 + sanity)`: 0.700 / 0.40 / 0.20 / 1.00 / 1.00 / 0.80 / 0.80

→ Phase 4-Re 환경에서는 `experiment_id` 별 dir 격리로 덮어쓰기 방지.

## 6. 환경 정보

- Model: `Qwen/Qwen3-8B` (vLLM 로컬 endpoint `http://210.222.65.89:50200/v1`)
- Encoder: `BAAI/bge-base-en-v1.5` (Apple Silicon MPS)
- Judge: 같은 vLLM endpoint + Qwen3-8B (LongMemEval 6 prompt template, MIT)
- Dataset: `benchmarks/LongMemEval/data/longmemeval_oracle.json` (500 questions, 6 type)
- W&B project: `hi-em-phase4` (chosh04089-soongsil-university)

## 7. Phase 4-Re 적용 후 변경

- 새 결과는 `results/experiments/{exp_id}/` 에 저장 — 본 archive 와 격리
- session.json (HP sweep / 4-method 비교) 도입
- atomic save / round / resume / replay 인프라 활성화
- 본 archive 는 **read-only reference** — 새 실험과의 비교 baseline
