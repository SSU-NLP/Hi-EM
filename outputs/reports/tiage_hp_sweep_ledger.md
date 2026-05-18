# Segmentation eval & HP-sweep coverage ledger (TIAGE + SuperDialseg + Dialseg711)

**목적**: 이미 평가한 (벤치 × method × HP grid × split) 조합을 한곳에
기록 → 재실행 낭비 방지. 새 sweep/벤치-eval 전 **여기부터 확인**. 새로
돌리면 본 ledger 에 한 행 append (CLAUDE.md: cross-experiment 인덱스는
`outputs/reports/`; 개별 결과는 각 `outputs/experiments/<name>/REPORT.md`).
(2026-05-18: TIAGE 전용 → 전 segmentation 벤치로 범위 확장.)

**판정 지표 regime 주의**: 2026-05-18 부터 **target = WD/F1/Pk**
(ARI/n_topics/collapse = guard). 그 이전 sweep 의 "best" 라벨은
*ARI-primary* 기준이라 regime 이 달라 best 행을 가로 비교 금지 — 수치
자체(WD/F1/Pk/ARI)는 각 REPORT 에 다 있으니 재해석은 가능.

데이터: TIAGE `benchmarks/tiage/.../anno/<split>/`. test = 100 conv /
1564 turn / 315 shift. train = 300 conv / 4392 turn / 887 shift.
seed-invariant 확정(std=0.000) → seed 0 단독 = 3-seed 동일.

---

## 평가 완료 (재실행 금지 — 결과는 해당 REPORT 참조)

| exp name (`outputs/experiments/`) | method | swept grid | split·seeds | 상태 | headline |
|---|---|---|---|---|---|
| `2026-05-17_tiage_v33x_compare` (+`_a1`) | 13-method | (기본 HP, sweep 아님) α=1 λ=10 등 | test · 0/1/2 | 완료 | ARI-primary regime. v3.3.9 미포함 시점 표. |
| `2026-05-17_tiage_all_hiem_full` | 12-method | 기본 HP | test · 0/1/2 | 완료(수기분석) | ARI best v3.3.6 0.359; 해석·판정 본문 |
| `2026-05-17_tiage_hpsweep_stageA` | **v3.3.6, v3.3.5** | **alpha{0.5,1,2} × lmda{5,10,20} × pe_var_sigma0_sq{0.01,0.04,0.1}** (27 combo/method = 54) | test · 0/1/2 | **부분 35/54** (BrokenPipe 중단, partial-safe 보존) | ARI-primary. best ARI v3.3.6{α0.5,λ5,σ0²0.04}=0.383; best WD v3.3.5{α0.5,λ10,σ0²0.1}=0.502(ARI0.313 nt2.6 degenerate 경향) |
| `2026-05-18_tiage_v339_full` | 13-method (v3.3.9 포함) | 기본 HP | test · 0/1/2 | 완료 | **WD/F1/Pk target**. v3.3.9 best F1 0.437 / Pk 0.415 / ARI 0.408 / WD 0.605; best WD v3.3.5 0.598 |
| `2026-05-18_tiage_v339_eta_ablation` | **v3.3.9** | **eta_prev{1.0,0.7,0.5,0.0}** (4) | test · 0 | 완료 | target. η 단조: η=1 best 전지표(F1 0.437/WD 0.605/Pk 0.415/ARI 0.408); η=0(RNN-only) worst(0.421/0.636/0.437/0.369). RNN 유해 확정 |

### prior (pre-DTS, LoCoMo-era — TIAGE DTS 와 무관, 참고만)
`2026-05-08_v331_hpsweep`, `2026-05-08_v332_pesweep`,
`2026-05-12_v3342_sweep_full` — LoCoMo QA task 시절. grid·결과는 각
REPORT 참조. DTS target 과 metric·benchmark 다름 → 재사용 불가, 재실행
대상도 아님.

---

## train-calibration (segmenter 무관, threshold-only — 저비용)

| script / out | grid | split | 상태 |
|---|---|---|---|
| `scripts/calibrate_v3310_delta_star.py` → `outputs/runs/_misc/v3310_delta_star_train.md` | v3.3.10 δ_eff: m{2,3,4} × ρ{0.5,0.7,0.9} × a{0,0.5,1.0} (27), F1-최적 δ* | **train** | **완료(2026-05-18)**. train-best m=2 ρ=0.7 a=0.5 δ*=0.5594 F1 **0.441** vs a=1.0(=v3.3.9 등가) F1 **0.435** → causal-window 이득 **+0.006 = 노이즈**. a=0 더 나쁨. **결론: v3.3.10 full test-sweep 폐기**; train-best+a=1.0 만 test 1회로 WD 각도만 확인. |

---

## 표준 벤치 평가 — 공식 SuperDialseg metric (Pk/WD/F1/Score)

`scripts/run_superdialseg_eval.py` · 데이터 `benchmarks/superdialseg_data/`
(coldog2333/SuperDialseg, gitignored). metric = 공식 verbatim
(window='auto'), Score=0.5·F1+0.25·(1−Pk)+0.25·(1−WD). 인코더 =
multi-qa-mpnet (우리 기본; 문헌 인코더 정합은 미확정 caveat).

| exp | 벤치/split | method | Score↑ | F1↑ | Pk↓ | WD↓ | predr |
|---|---|---|---:|---:|---:|---:|---:|
| `2026-05-18_superseg_test_v3310` | superseg test (1322d/17328t, bnd 0.232) | prev-cos@oracleθ (천장) | 0.447 | 0.458 | 0.467 | 0.660 | 0.542 |
| ″ | ″ | v3.3.9 @TIAGE-δ* (zero-shot) | 0.460 | **0.449** | 0.468 | 0.589 | 0.418 |
| ″ | ″ | v3.3.9 @oracle-δ* | 0.447 | 0.458 | 0.467 | 0.660 | 0.542 |
| ″ | ″ | **v3.3.10 @TIAGE-cfg (zero-shot)** | **0.463** | 0.432 | 0.471 | **0.541** | 0.316 |
| ″ | ″ | v3.3.10 @oracle-δ* | 0.446 | 0.458 | 0.470 | 0.663 | 0.545 |

**판정**: v3.3.10 > v3.3.9 on **Score·WD·predrate**(과분절 억제, codex
causal-window 예측 실증 — TIAGE +0.006 노이즈는 잡담 특유 가림이었음);
F1 은 v3.3.9 우위(경계 덜 찍어 recall↓ ↔ WD↑). 인접-cosine unsupervised
**천장 ≈ F1 0.46 / Score 0.45**(전 oracle 행 수렴) → 문헌 0.55~0.65 는
unsupervised 도달 불가 = supervised regime 시사. 직접 우열주장 금지
(인코더·supervised 미확정).

---

## 아직 안 한 / 갭 (다음 후보)

- **Stage A 잔여 19/54** (v3.3.6/5 미완 combo) — codex 진단상 구조문제로
  재개 critical-path 아님. v3.3.9 로 대체됨.
- **v3.3.9 HP sweep** (σδ_c·α·λ) — eta_prev 만 ablation. (δ* 는
  train-calibration 대상, test-sweep 금지 — leakage.)
- **v3.3.10 TIAGE test** — train +0.006(노이즈)라 full sweep 폐기.
  단 SuperDialseg 에서 Score/WD 이득 확인됨 → TIAGE test 1회는
  cross-check 가치만(낮음).
- **Dialseg711 test** (711d/19350t, zero-shot 표준벤치) — 어댑터
  준비됨(`run_superdialseg_eval.py --dataset dialseg711`), 미실행.
- **superseg δ* val-calibration** — 현재 v3.3.x 는 TIAGE-δ* zero-shot
  전이. superseg validation 으로 δ* 재calibration 시 향상 여지(미실행).
- **인코더 정합 ablation** (all-mpnet-base-v2 등 문헌 정합) — 미실행.
- **strong-similarity 천장(codex C)**: TextTiling-SBERT depth — 부분
  대체됨(oracle 행이 천장 ≈0.46 확정). 정밀 depth-method 는 미실행.
- TopiOCQA segmentation GT — 미착수.

---

_갱신 규칙_: 새 sweep/ablation/comparison 실행 시 위 "평가 완료" 표에
한 행 추가(exp name·method·정확한 grid·split·seeds·상태·headline).
부분중단도 "부분 N/M" 으로 기록. 2026-05-18 생성.
