# Hi-OnTop — Hi-EM Dialogue Topic Segmenter (main model)

`src/hi_em/hi_ontop.py` · `class HiOnTop` · 2026-05-22 · current main model

## 한 줄

v4.1.3 의 dead-code 를 전부 제거한 reduced form. v4.1.3 와 **matched HP 에서
byte-identical** (TIAGE/Dialseg711/SuperSeg 38,242 turn 0 mismatch 검증).
online, O(m)/turn, past-only causal-window cosine-distance threshold segmenter.

## 왜 만들었나

v4.1.3 (`HiEMSegmenterV413(HiEMSegmenterV411)`) 의 HP audit (2026-05-22)
에서 SEM2 machinery 가 default/canonical setting 에서 출력에 0 영향임이 실증됨:

- EventRNN — `eta_prev=1` default → δ_model weight 0, 학습/forward 안 함
- f0/restart/re-entry — `f0_min_starts≥2` circular deadlock 으로 영구 봉인
- SEM2 variance (per-topic σ²_k, scaled-inv-χ² posterior) — dead f0 로만 흘러감
- sticky-CRP `alpha` 및 canonical/default `lmda=10` — `_fresh_baseline_for_prev`
  의 prior-cancel 설계로 repeat-vs-fresh argmax 에서 상쇄. 단 `lmda=1`
  같은 낮은 stickiness 는 archived full form 에서 non-prev f0 fallback 과
  상호작용해 일부 출력 차이를 낼 수 있으므로 전역 dead 로 쓰지 않는다.
- `sigma_delta_c` / `var_likelihood_weight` / `pe_prior` / `cos_threshold` /
  `pe_threshold` / `hard_pe_fallback` / `min_transitions_for_pe` — 전부 dead
- RNN 구성 HP(`rnn_hidden_dim`, `rnn_lr`, `rnn_n_epochs`,
  `rnn_ready_min_transitions`, `rnn_max_history`, `seed`) 와 SEM2 variance
  세부 HP(`pe_var_sigma0_sq`, `pe_var_df0`, `pe_var_min_sq`,
  `pe_var_max_sq`, `pe_var_window`) 도 default `eta_prev=1` /
  f0-dead 경로에서는 출력 dead

→ v4.1.3 의 4단계 SEM2 파이프라인 (sCRP prior / RNN PE / σ²_k likelihood /
Bayes posterior argmax) 이 코드로는 실행되나, 수학적으로 단일 threshold 로 환원.
Hi-OnTop = 그 환원형을 dead code 없이 정직하게 구현.

## 알고리즘

L2-normalized 발화 임베딩 스트림 `s_1, s_2, …` 에 대해:

```
c_{t-1} = normalize( Σ_{i=1..min(m,t-1)} ρ^{i-1} · s_{t-i} )   # causal window
δ_prev  = 1 − cos(s_{t-1}, s_t)
δ_ctx   = 1 − cos(c_{t-1}, s_t)
δ_eff   = a · δ_prev + (1−a) · δ_ctx
g_t     = δ_eff / δ*                          # graded boundary score
boundary(t) ⟺ g_t ≥ 1  ⟺  δ_eff ≥ δ*
```

`topic_id` = monotonic segment counter (boundary 마다 ++; 재진입 없음).

### Edge cases (v4.1.3 parity)

- turn 0: `_prev_s` 없음 → `δ_eff = 0` → graded 0, boundary False, topic 0 생성.
- turn 1: `_recent=[s_0]` → `c_0 = s_0` → `δ_ctx = δ_prev` → `δ_eff = δ_prev`.
- `_recent` 는 ctx_window 크기로 capped (FIFO).

## HP (4개 전부 live)

| HP | 의미 | default |
|---|---|---|
| `delta_star` (δ*) | boundary threshold | 0.5594 (encoder/dataset calibration 필요) |
| `ctx_window` (m) | causal window 크기 | 2 |
| `ctx_decay` (ρ) | window geometric decay | 0.7 |
| `ctx_blend_a` (a) | δ_prev vs δ_ctx blend | 0.5 |

`beta` 등 v4.1.x 의 나머지 HP 는 Hi-OnTop 에 존재하지 않음 (dead 였음).

### δ* calibration

encoder-dependent. 측정값:
- bge/mpnet 계열 TIAGE-train prev-cos: ≈ 0.5594
- SeCom-swap (multi-qa-mpnet, MTB+ δ_prev p80, ctx_window=3): 0.6194
- SeCom-swap (multi-qa-mpnet, MTB+ **δ_eff p80, ctx_window=2**): 0.5983
  — δ* 는 δ_prev 가 아니라 δ_eff 에 적용되므로 ctx_window 변경 시 δ_eff 기준
  재보정이 원칙. ctx_window 3→2 default 변경에 따라 m=2 δ_eff 로 재산출한 값.
  `02_calibrate_delta_star.py --mode delta_eff --ctx_window 2` ·
  `delta_star_calibration_hiontop_m2.json`.

HP sweep (2026-05-22, `outputs/experiments/2026-05-22_v413_hp_sweep/`):
train/val+tune split 튜닝 → held-out/test 평균에서 swept config 가 canonical
default 를 못 이김 (swept −0.006 mean-3). **default HP 가 robust** 하다는
negative result 로 보고 가능. 단 dialseg711 은 official train split 이 없어
seeded tune/held-out split 을 썼으므로 full-test literature number 로 직접
인용하지 않는다.

## graded boundary score

`graded_score = δ_eff / δ*` 를 매 turn 노출 (Ben-Yakov & Henson 2018 의
graded hippocampal boundary response 매핑). bands:

| band | graded_score | downstream 권고 |
|---|---|---|
| very_weak | < 0.7 | 보류 |
| weak | 0.7 ~ 1.0 | within-segment |
| normal | 1.0 ~ 1.3 | 경계 |
| strong | ≥ 1.3 | 즉시 commit |

## SEM 계승 (정직한 서술)

Hi-OnTop 는 **full SEM2 구현이 아니다**. SEM 의 핵심 직관 — "event boundary 는
다음 관측이 최근 event context 로 잘 예측되지 않을 때 발생" — 의 *minimal online
realization*. SEM2-style RNN dynamics / sticky-CRP prior / f0-restart / variance
calibration 은 구현·audit 했으나 v4.1.3 default 에서 argmax 결정에 영향 없음
(→ `archive/legacy_sem_ablation/sem_core_v413.py` 에 ablation 증거물로 보존).
paper 는 이 audit 을 disclosure 로 명시.

## paper main claim

graded boundary score (calibrated) + online O(m)/turn latency +
SeCom LLM-segmentation-backend drop-in 교체 + segmentation latency 대폭 감소.

## API

```python
seg = HiOnTop(dim=768, delta_star=0.5594, ctx_window=2, ctx_decay=0.7, ctx_blend_a=0.5)
for s in scene_vectors:
    topic_id, is_boundary = seg.assign(s)
    g = seg.last_graded_score
seg.history()            # per-turn: turn/topic_id/is_boundary/delta_eff/graded_score
seg.graded_scores()
seg.boundary_strength()  # {very_weak/weak/normal/strong: count}
```

## 한계 / 검증 미해결

- 알고리즘이 causal-window cosine-distance threshold — TextTiling 류 lexical
  heuristic 과 *구조* 가 같음. 차이는 signal (contextual embedding cosine) +
  graded score + online latency. paper 에서 이 점 정직히 서술 필요.
- 신경과학 5-요소 (예측 event model / 경계 reset / LTM snapshot / re-entry /
  adaptive timescale) 중 "예측오류→경계" 1개만, degenerate 형태로 구현.
  biology revival 설계는 codex 스레드에 보존 (후속 버전 후보).
- δ* 외 HP 가 robust 하나, 이는 *현 데이터셋* 한정 — 다른 도메인 미검증.

## SeCom-swap 인코더 backend

`HiEMSecomSegmenter` (`secom_adapter.py`) 의 인코더는 두 contract 를 모두
지원한다 — `sentence-transformers` (`encode(list, batch_size=, normalize_
embeddings=, ...)`) 와 `hi_em.embedding` 의 `QueryEncoder`/`APIEncoder`
(`encode(list) -> L2-normalized`). `_encode()` 가 TypeError fallback 으로
분기.

SeCom-swap 의 segment 단계는 default 로 **Crts `/v1/embeddings` API**
(`APIEncoder`, `03_segment_v413.py --encoder_backend api`) 를 쓴다. Crts 의
`multi-qa-mpnet-base-dot-v1` 출력은 로컬 sentence-transformers 와 bit-identical
(cos=1.0 검증) → δ* calibration·segmentation 결과 불변, 로컬 CPU forward 만
API 로 offload (encode ~0.9 s/turn CPU → ~0.11 s/turn API). retrieve 단계는
SeCom 내부 langchain `HuggingFaceEmbeddings` (로컬) 그대로 — `benchmarks/` 는
읽기 전용. Crts≡로컬 이므로 retrieval 결과 동일.

## 변경 이력

- **2026-05-22 신설**: v4.1.3 dead-code audit → reduced form 추출. v4.1.3 와
  byte-identical 검증. decision-log 2026-05-22 참조.
- **2026-05-22 후속**: `ctx_window` default 를 canonical reported config 에
  맞춰 `2` 로 정정. v4.1.x archive 와의 parity 는 matched HP 기준으로 검증.
- **2026-05-22 SeCom-swap**: segment 인코더를 Crts API backend 로 전환
  (위 § 참조). ctx_window 3→2 변경에 맞춰 δ* 를 m=2 δ_eff p80 = 0.5983 으로
  재보정.
