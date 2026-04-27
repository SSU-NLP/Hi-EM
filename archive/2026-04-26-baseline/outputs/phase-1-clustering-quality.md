# Phase 1-6 옵션 5: Clustering Quality (V-measure / NMI / ARI)

**날짜**: 2026-04-25
**스크립트**: `scripts/run_clustering_quality.py`
**Raw 결과**: `outputs/phase-1-clustering-quality.json`

## 목적

Boundary F1만으로는 Hi-EM의 차별점(토픽 ID 부여 + centroid 누적)을 측정할 수 없다. Cosine baseline은 "boundary 지나면 다른 cluster"라 **돌아온 토픽을 같은 cluster로 묶지 못함**. Hi-EM은 centroid 비교라 같은 토픽 복귀 시 같은 ID 부여 가능하다는 가설을 직접 검증.

## 결과

| Benchmark | Method | V-meas | NMI | **ARI** | Hom | Comp |
|---|---|---|---|---|---|---|
| TopiOCQA dev | cosine(θ=0.525) | **0.926** | **0.926** | **0.488** | 0.950 | 0.902 |
| TopiOCQA dev | Hi-EM freq-shift (α=10, λ=1, σ₀²=0.1) | 0.910 | 0.910 | 0.187 | 0.985 | 0.845 |
| TopiOCQA dev | Hi-EM persistence (α=1, λ=10, σ₀²=0.01) | 0.919 | 0.919 | 0.398 | 0.949 | 0.891 |
| TIAGE test | cosine(θ=0.5) | **0.928** | **0.928** | **0.568** | 0.935 | 0.921 |
| TIAGE test | Hi-EM sweep-best (α=10, λ=3, σ₀²=0.1) | 0.909 | 0.909 | 0.314 | 0.979 | 0.848 |
| TIAGE test | Hi-EM persistence (α=1, λ=10, σ₀²=0.01) | 0.914 | 0.914 | 0.397 | 0.955 | 0.875 |

## 해석

### 1. 원래 가설 반박

"Hi-EM은 boundary F1에 약해도 토픽 ID 부여에서 우위"라는 가설은 **모든 metric에서 반박**됨:
- V-measure / NMI: cosine 우위 (격차 작음, ~0.01)
- **ARI**: cosine 우위 (격차 큼, 0.1+)
- 어떤 HP regime에서도 Hi-EM이 cosine을 못 넘음

원인은 sweep top 10 패턴과 일치: Hi-EM은 **homogeneity↑ / completeness↓** → **over-segmentation** 경향. ARI는 chance-corrected 쌍별 metric이라 over-cluster에 가장 민감.

### 2. 새 발견 — Boundary F1 ↔ ARI trade-off

| HP regime | Boundary F1 (TopiOCQA) | ARI (TopiOCQA) |
|---|---|---|
| freq-shift (α=10) | **0.471** (best) | 0.187 (worst) |
| persistence (α=1) | 0.378 | **0.398** (best Hi-EM) |

α↑ → boundary 정확도↑ + over-cluster → ARI↓
α↓ → boundary 둔감 + 묶기 보존 → ARI↑

**메모리 시스템 관점에선 persistence HP가 적합**: 같은 토픽 돌아올 때 같은 메모리 호출하려면 cluster 보존성(completeness/ARI)이 boundary 정확도보다 중요. → **Phase 2 LTM/Memory window 설계 시 persistence HP 채택 근거**.

### 3. Cosine baseline이 ARI에서 강한 이유

Cosine은 sequential clustering이라 같은 주제 turn이 인접해 있으면 같은 cluster, 멀리 떨어져 다시 등장하면 다른 cluster로 분리. TopiOCQA·TIAGE 모두 **같은 conversation 안에서 토픽 복귀가 드물고**, 인접 turn 간 cohesion이 강해 cosine sequential이 충분히 잘 동작. Hi-EM의 centroid 비교 우위가 발현될 시나리오(긴 대화에서 토픽 복귀)는 LongMemEval 같은 벤치마크가 필요 — 본 결과는 한계 명시.

## 결론

1. **원래 가설(Hi-EM의 unsupervised clustering 우위)은 두 벤치마크에서 모두 반박**됨.
2. **Boundary F1 ↔ ARI trade-off 발견** — Phase 2에서 persistence HP 채택 근거.
3. **옵션 3 (Phase 2 reframing) 강화**: "어떤 unsupervised segmentation metric으로도 Hi-EM의 가치 증명 불가. 진짜 가치는 Phase 4 downstream QA에서만 검증 가능."

## 한계

- **TopiOCQA/TIAGE는 평균 12~15턴, 토픽 복귀 거의 없음** — Hi-EM centroid 비교의 우위 시나리오(긴 대화 + 토픽 복귀)가 데이터에 없음. LongMemEval/LoCoMo (Phase 4)에서만 검증 가능.
- TIAGE GT cluster ID는 binary shift label에서 derive — sequential 가정 (한 번 shift하면 새 cluster, 같은 토픽 복귀 처리 못 함). cosine baseline 형식과 동일한 가정이라 baseline에 유리할 가능성 (단, Hi-EM도 같은 GT 기준이라 비교 자체는 fair).

## Cascade

- `context/06-decision-log.md` — 2026-04-25 entry append
- `plan.md` — 옵션 5 [x] 처리, 메모: persistence HP가 Phase 2에 적합 근거
- `handoff.md` — 다음 할 일 → 옵션 3 (Phase 2 reframing 진입)
- `report.md` — §7.1 옵션 E에 결과 추가
