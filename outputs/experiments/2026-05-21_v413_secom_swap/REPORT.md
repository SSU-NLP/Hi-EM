# v4.1.3 → SeCom segmentation backend swap

`outputs/experiments/2026-05-21_v413_secom_swap/` · 2026-05-21 · in progress

## 한 줄

SeCom (Pan et al., ICLR 2025) 의 LLM 기반 segmentation backend (`gpt-4o-mini`) 를
Hi-EM v4.1.3 (online, O(1)/turn) 으로 drop-in 교체. **downstream QA 품질이 유지/소폭변동
되며, segmentation latency 는 [TBD]× 감소** 한다는 paper 의 핵심 claim 의 실측 근거.

## 실험 setup

**Dataset**: Long-MT-Bench+ (`panzs19/Long-MT-Bench-Plus`)
- n_conv = 11, n_sessions ≈ 55 (5/conv), avg n_turns/session = 13.7, n_questions = 27/conv

**Pipeline (SeCom 원본 5-stage 그대로, segment 만 swap)**:
1. segment → topic 단위 chunking
2. compress (LLMLingua-2 xlm-roberta-large-meetingbank, rate=0.75)
3. retrieve (multi-qa-mpnet-base-dot-v1 + FAISS, top-k=1)
4. chat (`openai/gpt-4o-mini` via Crts)
5. eval (QA F1, subspan EM, ROUGE-L, BERTScore-F1)

**비교 row**:

| 표기 | Retriever | Segmentation | Response gen |
|---|---|---|---|
| (paper) SeCom (BM25, GPT4-Seg) | BM25 | GPT-4-0125 | GPT-3.5-Turbo |
| (paper) SeCom (MPNet, GPT4-Seg) | MPNet | GPT-4-0125 | GPT-3.5-Turbo |
| (paper) SeCom (MPNet, Mistral-7B-Seg) | MPNet | Mistral-7B-Instruct-v0.3 | GPT-3.5-Turbo |
| (paper) SeCom (MPNet, RoBERTa-Seg) | MPNet | RoBERTa (SuperDialSeg-FT) | GPT-3.5-Turbo |
| **(ours) Control: gpt-4o-mini** | MPNet | `openai/gpt-4o-mini` (Crts) | `openai/gpt-4o-mini` |
| **(ours) Ours: v4.1.3** | MPNet | **Hi-EM v4.1.3 (online, O(1)/turn)** | `openai/gpt-4o-mini` |

설계 noteset:
- SeCom 의 4 paper variant 는 **Table 1 / Table 3 보고치 인용** (Mistral-7B local + RoBERTa
  fine-tuned ckpt 재현 비현실적). 우리는 2개 ours row 만 직접 실행.
- 두 ours row 는 chat 모델 (gpt-4o-mini) 과 retriever (MPNet) 통일 → **유일한 차이가
  segmentation method**. 공정한 swap 비교.

**v4.1.3 segmentation 파라미터**:
- Encoder: `sentence-transformers/multi-qa-mpnet-base-dot-v1` (L2-normalized)
- δ\* re-calibrated for mpnet (TIAGE train δ* = 0.5557 은 bge 기준이라 부적합):
  see `delta_star_calibration.json` — 권장값 = δ_prev p80
- 기타 v4.1.1 default (α=1, λ=10, β=0.25, pe_threshold=1.0, ctx_window=3, ctx_decay=0.7,
  ctx_blend_a=0.5, η=1.0, f0_min_starts=2)
- Per-session fresh segmenter (SeCom 의 LLM call 도 session 단위 → fair compare)

## 결과 (TBD — running)

### Latency

| method | encode (s/conv) | segment (s/conv) | total (s/conv) | **ms/exchange** |
|---|---:|---:|---:|---:|
| baseline (gpt-4o-mini) | — | — | TBD | TBD |
| **ours (v4.1.3)** | TBD | TBD | TBD | TBD |

**Speedup**: TBD ×

### Downstream QA

| method | QA F1 ↑ | Subspan EM ↑ | ROUGE-L ↑ | BERTScore-F1 ↑ |
|---|---:|---:|---:|---:|
| baseline (gpt-4o-mini seg) | TBD | TBD | TBD | TBD |
| **ours (v4.1.3 seg)** | TBD | TBD | TBD | TBD |
| **Δ (ours − baseline)** | TBD | TBD | TBD | TBD |

### Segment statistics

| method | n_segments | avg exchanges/segment | very_weak/weak/normal/strong band |
|---|---:|---:|---|
| baseline | TBD | TBD | — |
| ours | TBD | TBD | TBD |

## 해석 (TBD)

## 판정 (TBD)

## 한계 / 검증 미해결

- **n_conv = 11**: Long-MT-Bench+ test split 전체. 통계적 power 제한 (단일 run).
  multi-seed (생성 LLM temperature=0 이라 seed 효과 없음, segmentation 도 deterministic).
- **mpnet δ\* 는 휴리스틱**: paper 의 F1-supervised δ\* (TIAGE train) 가 아닌
  MTB+ 의 δ_prev 분포 p80. 다른 quantile (p70/p85/p90) 의 sensitivity 미측정.
- **SeCom 의 4 paper row 는 재현 안 함** — 인용치라 우리 환경 (Crts gpt-4o-mini chat)
  과 chat LLM 다름 (paper = GPT-3.5-Turbo). 절대값 비교 시 disclaimer 필요.
- **mpnet retriever CPU 실행** (WSL2 GPU 미가용). retrieval 자체 결과는 같지만
  retrieval 시간은 paper 값과 직접 비교 불가.
- **LLMLingua-2 compression**: 동일 rate=0.75 사용. 두 method 가 input segment 가
  달라서 compressed token 수도 다를 수 있음 → fair 한지 검토 필요.

## 산출

- `src/hi_em/secom_adapter.py` — HiEMSecomSegmenter (mpnet → v4.1.3 wrap)
- `scripts/secom_swap/01_prepare_data.py` — MTB+ → SeCom JSONL
- `scripts/secom_swap/02_calibrate_delta_star.py` — mpnet δ* 추정
- `scripts/secom_swap/03_segment_v413.py` — v4.1.3 segmentation runner
- `scripts/secom_swap/04_segment_baseline.py` — gpt-4o-mini segmentation runner
- `scripts/secom_swap/05_compress.py` / `06_retrieve.py` / `07_chat.py` / `08_eval.py`
- `scripts/secom_swap/run_pipeline.sh` — orchestrator
- `delta_star_calibration.json` — mpnet δ_prev 분포 + 권장값
- `latency_ours.json` / `latency_baseline.json` — per-conv timing
- `metrics_ours.json` / `metrics_baseline.json` — downstream eval

## 변경 이력

- **2026-05-21 초안**: 인프라/스크립트 작성, paper variants 표 + 우리 row 정의
- **2026-05-21 (실행)**: δ* calibration → segment → compress → retrieve → chat → eval
