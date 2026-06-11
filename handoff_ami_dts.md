# Handoff — AMI/DTS 화제분절 신호 탐색 (Hi-OnTop)

**기간**: 2026-06-09 ~ 2026-06-11 · **대상**: Hi-OnTop 화제분절 *신호/판정* 개선
**한 줄**: δ_eff(직전 점프 임계치)의 한계를 진단하고, **de-neutralized prototype + run-length 적응 β** 라는
새 신호를 발견·검증. 신호(oracle) 차원에서 진짜 성과(아무도 못 깬 superseg 벽 돌파, cross-domain 검증),
배포(deploy) 차원에서는 modest(+0.024 Score) — **online reset 부트스트랩**이 미해결 천장.

---

## 0. 배경 / 목표
- Hi-OnTop 기존 신호: `δ_eff = a·δ_prev + (1−a)·δ_ctx` (직전 발화 + 2턴 윈도우 cosine 거리), 적응임계치 μ+cσ.
- 두 도메인: **AMI**(긴 회의, drift, 잡음 多, gold 극sparse) vs **DTS**(tiage/dialseg711/superseg; 짧은 대화
  concat, sharp seam, 잡음 거의 無).
- 목표: 두 도메인 *모두* δ_eff를 이기는 신호. online·0-look-ahead·인코더 고정(MiniLM-int8)·무학습 유지.
- metric: AMI = ±2 tolerance F1 / Score(=0.5F1+0.25(1-Pk)+0.25(1-WD)); DTS = exact F1(concat-seam라 sharp).
  oracle = per-meeting/dialog 최적 임계치(신호 천장). deploy = 완전 online.

## 1. 진단 — δ_eff·V_rel·prototype 형태의 한계
- **magnitude 단독 불가**: 가장 큰 cosine spike가 경계가 아니라 noise(화자전환·단발 이상치). top-K/threshold로
  LLM 경계 일치 ~0.11 고착.
- **V_rel = r_active − λ·r_global** (active prototype 거리 − 0.6·global 거리): drift(AMI)에 강함
  (oracle 천장 0.687 > LLM 0.543, 2-fold·제약 OK). **그러나 DTS에선 회귀** — concat-seam은 prototype(평균)
  보다 직전 점프(δ_prev)가 sharp. λ(global) 항이 원인 아님 — λ=0(순수 prototype)도 superseg<δ_eff.
- **prototype 형태 변형 다 실패**(superseg 벽): mean/nn/medoid/varnorm/subspace/info-gate 전부 superseg
  0.42~0.44 < 0.467. 이유: **평균(prototype)은 짧은 segment에 원리적으로 불리**(직전 점프가 최선).
- trace로 확인(`outputs/reports/vrel_*_trace.md`): prototype은 텍스트 요약 아님(임베딩 평균 벡터). 길게
  누적하면 추임새 같은 **중립점으로 흐려짐**(추임새 cos 0.7, 내용 0.3). gold/LLM 없이 "요약"은 불가.

## 2. 돌파 — de-neutralized prototype
**핵심**: prototype·발화에서 **중립(global) 성분을 제거**한 뒤 비교 → "화제의 *변별적* 방향"만.
```
m_c = normalize(m − β·(m·g)·g)   # prototype에서 global 성분 β만큼 제거
x_c = normalize(x − β·(x·g)·g)
r_active = 1 − cos(x_c, m_c)
V = r_active − 0.6·(1 − cos(x, g))
```
- **β=1 (full de-neut)**: superseg **0.506 > 0.467 — 처음으로 벽 깸**, DTS 3개 다 δ_eff 넘음. 단 AMI 0.222(짐).
- **β=0 = V_rel**(AMI 0.659, DTS 짐). 두 도메인이 β에 대해 정반대.

## 3. 적응 β (자기검출) — 도메인 라벨 없이
`β_t = clip(A − B·log(1+l/L0), 0, 1)`, l = (현)segment 길이.
- 짧은 segment(DTS) → β→1 (de-neut) / 긴 segment(AMI) → β→낮음 (V_rel). **run-length가 sharp/drift를
  자동 판별** (R̄=global 집중도는 방향이 **거꾸로**라 실패 — superseg R̄가 오히려 제일 높음).
- **best (A,B)=(2.0,1.0)** oracle: tiage 0.462 / dialseg711 0.384 / superseg 0.506 / AMI 0.341 — **4개 다
  δ_eff 초과**. β평균 = DTS 1.00 / AMI 0.39.

## 4. 검증 (정직)
| 검증 | tiage | dialseg711 | superseg | AMI | 결론 |
|---|--:|--:|--:|--:|---|
| oracle (full) | 0.462 | 0.384 | 0.506 | 0.341 | 4개 strict (δ_eff .452/.313/.467/.235) |
| **2-fold (도메인 내)** | even **FAIL** / odd ✓ | ✓✓ | ✓✓ | ✓✓ | dialseg/superseg/AMI robust, **tiage tie** |
| **LOO (cross-domain)** | +0.010 PASS | +0.071 PASS | +0.039 PASS | +0.072 PASS | (A,B) 전이됨; AMI는 grid 어떤 (A,B)든 >δ_eff |
| **deploy (calib-c, held-out test)** | — | — | — | Score 0.367 vs δ_eff 0.343 (±2F1 동률 0.106) | **+0.024 modest, localization 동률** |

## 5. Deploy 현실 (★ 가장 중요한 한계)
- 공정 비교(c를 calib에서 정하고 test 보고; 둘 다 같은 적응임계치 μ+cσ): **adaptive-deneut 0.367 vs
  δ_eff 0.343 Score (+0.024)**, **±2F1 둘 다 0.106 (동률)**.
- 즉 **oracle의 큰 우위가 deploy로 거의 안 넘어옴.** Score 이득은 Pk/WD(개수·간격), localization(±2F1)은 동률.
- 원인 = **online reset 부트스트랩**: 깨끗한 prototype은 *정답 경계에서 reset*해야 하는데(oracle), online은
  경계를 몰라 추측 reset → 틀리면 prototype 오염 → 악순환. hard-reset·robust·peak·EM 반복정제·
  BOCPD particle filter·lagged changepoint emission **모두 격차 못 메움**(deploy ±2F1 0.07~0.15 vs clean+μcσ
  oracle 0.554).

## 5b. Calibration 불필요 (c 안 골라도 됨) — 2026-06-11 추가
deploy 비교 시 c(적응임계치 σ-배수)를 calib/test split으로 고르는 게 번거로움. 검증 결과 **불필요**:
- **AMI Score-vs-c**: deneut Score가 *모든* c에서 δ_eff ≥ (c=2.5 .356/.329, c=2.0 .364/.274, c=1.5 .372/.203,
  c=1.2 .329/.176, c=1.0 .267/.170). 우열이 c에 안 뒤집힘 → **고정 c(또는 Otsu)면 충분, calib 생략 가능.**
  (`scripts/ami_dts_score_vs_c.py`)
- **AUC (threshold-free 신호 비교, c 자체 없음)**: ±2 AUC — AMI deneut 0.621 vs δ_eff 0.497(+0.123, 큼),
  tiage +0.047, superseg +0.025, **dialseg711 −0.045(혼재)**. δ_eff ±2 AUC ~0.50 = 거의 random; 둘 다 절대값
  약함(0.5~0.62, task 난이도). (`scripts/ami_dts_auc_eval.py`)
- **★ caveat (지표 비대칭)**: `Score`는 deneut 압승이나 `±2F1`만 보면 δ_eff가 약간 나음(δ_eff ±2F1 ~0.168 vs
  deneut ~0.138). deneut의 Score 우위는 Pk/WD(개수·간격)에서 옴. 지표를 Score로 두면 deneut 승.
- 앞서 "공정 calib +0.024"는 δ_eff가 자기 best-c(과소분절)로 끌어올린 것 — 고정 c에선 격차 +0.04~+0.17.

## 6. 미해결 / 다음 후보
1. **online reset 부트스트랩** (최우선) — clean prototype을 online으로 유지. codex 자문: BOCPD lagged
   changepoint-start emission(구현했으나 미흡), commit-and-refine(bounded lag), soft/multi-hypothesis.
2. **(A,B) 자기보정** — 고정값 대신 데이터 통계(평균 segment 길이 등)에서 유도 → tuning 자체 제거.
3. **3번 predictive prototype** (transformer dynamics, RNN보다 선호) — 미시도. 단 무학습 제약 충돌·짧은
   segment 데이터 부족 우려.
4. tiage tie는 정직하게 인정 (작은 데이터 noise + δ_ctx가 짧은 tiage에 잘 맞음).

## 7. 산출물
- **스크립트**(`scripts/`): `ami_vrel_eval.py`, `ami_vrel2_eval.py`, `ami_bocpd_eval.py`,
  `ami_bocpd_lag_eval.py`, `ami_localmap_eval.py`, `ami_adaptive_deneut_deploy.py`, `dts_vrel_check.py`,
  `trace_vrel_segment.py`, `trace_compare3.py`, `trace_summary_build.py`, `trace_proto_content.py`.
  `/tmp/` 탐색 스크립트(oracle/β/overfit/LOO): proto_smart2, proto_beta, beta_runlen2, beta_overfit,
  beta_loo, deploy_calib — 재현 시 `scripts/`로 승격 필요.
- **trace/리포트**(`outputs/reports/`): `vrel_segment_trace.md`, `vrel_compare3_trace.md`,
  `vrel_summary_build_trace.md`, `vrel_proto_content_c2.md`.
- **REPORT**: `outputs/experiments/2026-06-10_ami_vrel_localmap/REPORT.md` (V_rel 단계까지; de-neut/adaptive-β는
  본 handoff가 최신).
- **decision-log**: 2026-06-10 entry (V_rel). **de-neut/adaptive-β 결정은 아직 decision-log 미기록 — 추가 필요.**

## 8. 핵심 수치 한눈에
```
                          tiage  dialseg  superseg   AMI(±2)   AMI(Score,deploy)
δ_eff (baseline)          0.452   0.313    0.467     0.235      0.343(best-c)
adaptive-deneut (2.0,1.0) 0.462   0.384    0.506     0.341      0.367(calib-c, test)
                          tie     ✓        ✓벽깸     ✓          +0.024
(참조: AMI V_rel oracle 0.687, clean+μcσ deploy-천장 0.554, LLM full-context ±2 0.543/Score 0.640)
```

## 부록 A — 전체 시도 ledger (빠짐 없이)
**진단**
- δ_eff z-score (LLM/gold/random turn): 신호는 LLM 경계에 솟으나(z+0.545) magnitude 최대값은 noise → 분리 불가.
- picking(top-K, global z-threshold): LLM 경계 일치 ~0.11 고착 — picking으론 못 풀어.
- **gap 분해**(`decompose_gap`): clean(gold-reset)+per-meeting임계 0.687 / clean+단순μ+cσ **0.554** /
  detected-reset deploy 0.15 → **임계치 충분, 병목은 reset**임을 분해로 증명.

**codex 설계 (3개, 전부 구현·실측)**
- **local-MAP A1~A5**(`ami_localmap_eval`): active-event LLR(가우시안) + sCRP prior + coherence/speaker/singleton
  penalty. → **실패** (prior가 1-D LLR 압도 → 과분절 12000개; per-particle μ 수정 후도 Score 0.305/±2F1 0.069).
- **BOCPD top-K particle filter**(`ami_bocpd_eval`): 개수 맞춤(Pk/WD↑) but localization 나쁨(±2F1 0.069).
- **BOCPD lagged changepoint-start emission**(`ami_bocpd_lag_eval`): localization 0.069→0.10 개선, 그래도 deploy 0.10 ≪ 0.554.

**신호 탐색 (prototype/blend)**
- V_rel = active − λ·global → AMI oracle 0.687. prototype 형태 mean/centroid/nn/medoid/window/content-weight/
  robust/subspace/varnorm/info-gate → **전부 superseg 벽(0.467) 못 넘음**(0.42~0.44).
- **de-neutralize**(중립성분 제거) → superseg 0.506 **첫 돌파**. fixed-a combo(δ_prev+V_rel) → strict 없음.
- **adaptive a_t reliability blend**(codex 설계, δ_prev/V_rel z-score blend; `adaptive_probe`,`grid_adaptive`) →
  **실패**(양 극단보다 못한 어중간한 중간; tiage 간신히, dialseg/superseg 회귀, AMI 0.46≪0.66).
- **부분 de-neut β sweep** → β=0.70 고정 strict(단 AMI 0.29로 희생).
- **adaptive β = R̄(global 집중도)** → **실패**(R̄ 방향 거꾸로: superseg R̄ 최고, AMI 최저).
- **adaptive β = run-length**(`ami_dts_adaptive_beta`) → **성공**, best (2.0,1.0).

**deploy 시도**
- V_rel 적응임계치 Score 0.358. robust/peak/anchor/refractory(`ami_vrel2_eval`) → 무효(±2F1 0.15 고착).
- EM 반복정제(`em_refine`) → 나쁜 고정점 0.13 수렴 안 함. de-neut deploy(`ami_adaptive_deneut_deploy`) Score 0.372.

**검증**: overfit 2-fold(`ami_dts_beta_overfit`), LOO(`ami_dts_beta_loo`), deploy calib(`ami_dts_deploy_calib`),
Score-vs-c·Otsu(`ami_dts_score_vs_c`), AUC(`ami_dts_auc_eval`).

**이 handoff scope 밖 (별도 문서)**: LLM 버퍼-지연 곡선(`outputs/experiments/2026-06-09_llm_buffer_curve/REPORT.md`,
figure_R), filler forward-merge/geometry-backchannel/info-content/GraphSeg(`outputs/experiments/2026-06-09_ami_
filler_prototype/`, `outputs/reports/ami_*_view.md`, `scripts/run_graphseg_ami.py`). → AMI robustness 진단 단계.

## 9. 정직한 한 줄 결론
**de-neut + run-length 적응 β는 신호(oracle) 차원에서 진짜 발견 — superseg 벽을 cross-domain robust하게 깸.
그러나 deploy로는 modest(+0.024 Score, localization 동률) — online reset 부트스트랩이 남은 천장.**
정식 hi_ontop 버전 승격은 deploy가 이 격차를 메운 뒤로 보류.
