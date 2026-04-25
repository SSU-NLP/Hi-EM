# Hi-EM 프로젝트 종합 보고서

**작성 시점**: 2026-04-25 (Phase 1-5 TIAGE 평가 직후, Phase 1-6 종합 Gate FAIL 시점)
**목적**: 사용자가 처음부터 끝까지 다시 보고 다음 방향을 판단하기 위한 입력 문서.

---

## 0. 프로젝트 요약

**Hi-EM (Human-inspired Episodic Memory)** — Transformer 기반 LLM에 fine-tuning 없이 붙는 실시간 대화 메모리 관리 시스템.

**핵심 아이디어**: 인지과학 모델 SEM(Franklin et al. 2020, *Psychological Review*)의 **sticky Chinese Restaurant Process** 기반 event segmentation을 쿼리-토픽 구조로 재해석. 사용자 발화 스트림을 토픽 단위로 자동 분할하고, **LTM(SSD 영속 저장) ↔ Memory window(STM, 현재 라운드 prefill 대상) 계층**을 통해 LLM 호출 시 관련 턴만 효율적으로 전달.

**설계 제약** (`brief.md`)
- **No fine-tuning** — 어떤 Transformer LLM에도 plug-in
- **실시간** — 턴당 추가 latency +10~20% 이내
- **PyTorch only** (TensorFlow/Keras 금지)
- **SEM2 코드 직접 복사 금지** (알고리즘 참조만)

**차별점 가설** (검증 대상)
- vs Sliding window: 오래된 정보 보존
- vs Full context: context length·비용 폭발 회피
- vs RAG (flat retrieval): topic 구조로 조직화 → 더 관련 있는 턴 prefill

---

## 1. 아키텍처

### 1.1 데이터 흐름

```
매 턴:
  user query → bge 임베딩 → HiEMSegmenter.assign() → topic_id
  → LTM에 (turn 원문 + topic_id + 메타) 영속 기록

매 라운드 (LLM 호출 직전):
  current query → 관련 topic 선별 → LTM에서 해당 topic의 턴들
  → Memory window(STM)에 promotion → LLM의 prefill prefix로 전달
  → LLM 응답 → 새 턴 자동 segmentation 루프
```

### 1.2 모듈 구조 (확정 + Phase 1-1 구현 완료)

```
src/hi_em/
├── embedding.py      # bge-base-en-v1.5 wrapper (L2 norm, 768dim)
├── topic.py          # Topic 클래스: μ + diag σ² + Welford 업데이트
├── scrp.py           # sticky_crp_unnormed(counts, prev_k, α, λ)
└── sem_core.py       # HiEMSegmenter.assign(s) → (topic_id, is_boundary)

(Phase 2 이후 추가 예정)
├── ltm.py             # SSD 영속 저장
├── memory_window.py   # 현재 라운드 prefill 대상 선별
├── importance.py
├── merge.py
└── orchestrator.py    # 매 턴/라운드 파이프라인
```

**용어**: 초기 설계엔 "KV cache paging"이었으나 LLM 런타임 API에 종속되어 책임 경계가 흐려져 **"LTM → Memory window promotion"으로 전환** (`context/06-decision-log.md` 2026-04-23). KV cache 재사용은 prefill prefix 사용의 부산물이며 실제 paging은 downstream(vLLM/SGLang) 담당.

---

## 2. 수학 모형 (옵션 A 확정)

### 2.1 Scene 임베딩
$$\mathbf{s}_n = \text{normalize}(\text{bge\_base\_en\_v1.5}(\text{query}_n)) \in \mathbb{R}^{768}$$
**입력은 사용자 쿼리만** (응답 미포함, design rule).

### 2.2 sCRP Prior (SEM 식 1, 동치 구현)
$$\Pr(e_n = k \mid e_{1:n-1}) \propto
\begin{cases}
C_k + \lambda\,\mathbb{I}[e_{n-1}=k] & k \le K \\
\alpha & k = K+1
\end{cases}$$

- $C_k$: 지금까지 topic $k$에 할당된 턴 수
- $\alpha$: concentration (새 토픽 prior)
- $\lambda$: stickiness (이전 토픽 보너스)
- **Hi-EM 기본값** (persistence regime): $\alpha=1.0$, $\lambda=10.0$, $\sigma_0^2=0.01$
  → 대화는 topic persistence가 기본이라는 가정

### 2.3 Likelihood (옵션 A — Centroid + diag variance)
$$P(\mathbf{s}_n \mid e_n=k, \mu_k, \sigma_k^2) = \mathcal{N}(\mathbf{s}_n;\, \mu_k,\, \mathrm{diag}(\sigma_k^2))$$

- Cold start ($n_e < 3$): $\sigma_k^2 = \sigma_0^2 \cdot \mathbf{1}$
- 이후: Welford 온라인 업데이트, $\sigma_\min^2$ floor 적용

### 2.4 MAP 배정 (online, SEM 식 8-9)
$$\hat{e}_n = \arg\max_k\,\big[\log \Pr(e_n=k \mid e_{1:n-1}) + \log P(\mathbf{s}_n \mid e_n=k)\big]$$

### 2.5 Welford 업데이트
```
n_e += 1
delta = s - μ
μ    += delta / n_e
M2   += delta * (s - μ)         # μ 갱신 후 다시 계산
σ²    = max(M2 / n_e, σ_min²)   # n_e ≥ 3 이후에만
```

### 2.6 Markov 확장 — **철회**
초기 설계에 $P(e_n \mid e_{n-1}, \mathbf{s}_{n-1})$ 형태 확장 검토했으나 옵션 A에선 likelihood가 이미 $\mathbf{s}_n$을 사용 → prior에 $\mathbf{s}_{n-1}$ 추가는 **double counting**. 철회.

### 2.7 SEM2에서 의도적으로 미포팅한 부분
- **restart vs repeat 분기** (`run()` 내부): 옵션 A에서 centroid가 $\mathbf{s}_{n-1}$에 의존 안 함 → 두 likelihood 동일, 분기 무의미
- **`prior[k_prev] -= λ/2` halving**: 논문에 explicit 유도 없음(`context/00-sem-paper.md §7` 검증 미해결), 보수적 생략

---

## 3. Phase 0 — 자료 분석 및 설계 확정

### 3.1 Step 0-1: SEM 논문 정독 ✓

**산출물**: `context/00-sem-paper.md` (11,940자)
- SEM-paper.pdf 35페이지 정독 (`pdftotext` 산문 + `pdftoppm` 이미지 직독으로 수식 페이지 6,7,8,9,10,11,19,20,34,35 처리)
- SEM2 코드(`SEM/sem/sem.py`) `_calculate_unnormed_sCRP` (L.144-159), `run()` (L.161-354) verbatim pseudocode 작성
- 식 1~24 전체를 **계승/대체/폐기** 카테고리로 분류

| 분류 | 식 | 의미 |
|---|---|---|
| 유지 | 1, 4, 5, 6, 8, 9 | sticky-CRP prior + Bayes posterior + likelihood factorization + log-lik = −PE² + local MAP |
| 대체 | 2, 3 | $f$ (GRU)와 $f_0$ → 옵션 A의 centroid Gaussian으로 교체 |
| 폐기 | 7 | full posterior — Bell number 폭발로 intractable |
| 폐기 | 10–15 | memory corruption (Z-channel, uniform time) — 실서비스 불필요 |
| 폐기 | 18–23 | Gibbs sampling reconstruction — LTM 원문 저장으로 대체 |
| 폐기 | 16, 17, 24 | 인간 인지 실험 시뮬레이션 전용 |

**검증 미해결 3건** (§7에 명시):
1. 식 2 `diag(β)` vs 식 11 `τI` covariance 형태 차이의 이론적 유도 부재
2. SEM2 `prior[k_prev] -= λ/2` halving 유도 부재 → Hi-EM 미포팅, 추후 sanity test 필요
3. Markov 확장 형태 — Step 0-3에서 철회

### 3.2 Step 0-2: 벤치마크 데이터 분석 ✓

**산출물**: `outputs/benchmark-analysis.md` (8,239자)

| 벤치마크 | 단위 | 단위 수 | 턴/단위 | turn 길이 | Topic GT | Claude-유사도 |
|---|---|---|---|---|---|---|
| LoCoMo | 대화 | 10 | 588 | 108~142자 | 없음 (session=날짜) | 중 |
| TopiOCQA dev | 대화 | 205 | 12.3 | 37자 (factoid) | **있음** (Wiki doc) | 낮음 |
| LongMemEval oracle | 질문 | 500 | 22 (1.9 sess × 11.6 turns) | 1206자 (긴 chat) | 없음 (semantic tag) | **높음** |

**핵심 관찰**:
- TopiOCQA만 **명시적 turn-level topic ground truth** 보유 (Wiki document 단위)
- LongMemEval session ≠ topic — 한 세션 안에 여러 subtopic 공존 가능. **session 경계를 topic GT proxy로 쓰면 안 됨** (이 인식이 Phase 2.5 폐기로 이어짐)
- LongMemEval은 Hi-EM 주 타깃 시나리오(Claude-style chat)와 가장 유사 → **Phase 4 downstream QA 평가용**
- Vivid 차이: turn 길이 **30배 분포**(37 vs 1206자) → 임베딩 분포·variance 사정이 벤치마크별로 완전히 다름. $\sigma_0^2$ 등 hyperparameter regime split 가능성 시사

**TIAGE는 0-2 시점에선 Tier 2 옵션이었으나** Phase 2.5 폐기 후 Tier 1으로 승격 (아래 §6.4 참조)

### 3.3 Step 0-3: 사건 모델 옵션 A 확정 ✓

**옵션 A (Centroid + diag variance)** 선택. 후보 6개(A 단순 centroid / B + momentum / C + entity set / D multi-signal ensemble / E linear predictor / F 새 제안) 중.

**선택 근거**:
1. **Incremental 설계 원칙**: 실험 근거 없는 multi-signal 가중치 튜닝 회피, 단순 baseline 먼저 깔고 병목 보면 D로 확장
2. **SEM2 자연 대응**: `log_likelihood_next`/`log_likelihood_f0` Gaussian centroid로 직접 매핑
3. **Welford 적합**: cold start prior + 온라인 업데이트
4. **벤치마크 한계 명시 후 진입**: LongMemEval 긴 content에서 centroid 단독 한계 예상 → Phase 4 실험 후 확장 여지

**부수 결정**: Markov 확장 철회 (위 §2.6).

**기각 옵션 (벤치마크 근거)**:
- B (momentum): 대화 비순차로 효과 약함
- C (entity set): TopiOCQA bias 위험
- D (multi-signal): 가중치 근거 없음
- E (linear predictor): 작은 topic 과적합 / cold start 약함

---

## 4. Phase 1 — Topic 경계 감지 코어 + 평가

### 4.1 Step 1-1, 1-2: 구현 + 단위 테스트 ✓

**구현** (`src/hi_em/`): embedding.py / topic.py / scrp.py / sem_core.py
**테스트** (`tests/`): test_scrp.py (7) / test_topic.py (6) / test_sem_core.py (5) — **18/18 passing in 0.89s**

테스트 항목:
- sCRP: 첫 턴, stickiness 적용, Hi-EM α/λ 반전, SEM2 default, 용량 가득, 입력 불변
- Topic: Welford vs `np.mean`/`np.var` 일치, cold start 복귀값, σ_min² floor, log-lik 최대 at centroid, PE zero, 입력 불변
- HiEMSegmenter: 첫 턴 topic 0, stickiness 유지, 3 cluster 복원, boundary flag, 재현성

### 4.2 Step 1-3: TopiOCQA dev 측정 — 7회 iteration

**Gate 기준** (plan.md): Hi-EM F1 > cosine baseline F1 AND F1 > 0.4 AND 턴당 overhead ≤ 200ms

**초기값** (Hi-EM persistence HP α=1, λ=10, σ₀²=0.01): F1=0.378 → **FAIL**

**탐색 7-iteration 종합**:

| Iter | 시도 | 최고 Hi-EM F1 | Cosine ref | 결론 |
|---|---|---|---|---|
| 1 | HP grid 108 configs | 0.471 | 0.467 | **HP만으로 marginal PASS** (α=10, λ=1, σ₀²=0.1 = SEM2 defaults) |
| 2 | 구조 변형 5종 (gauss-origin/global/self, vMF-origin/const) | 0.471 | 0.467 | 구조 변경 효과 없음 |
| 3 | Multi-signal (cos + jaccard + entity) 564 configs | 0.451 | 0.467 | 가산 multi-signal 오히려 퇴보 |
| 4 | Contextualized (K∈{0..all} prior turn 연결) | 0.439 | 0.464 | **context 추가가 오히려 해** (인접 임베딩 동질화) |
| 5 | Anchor 4종 (centroid/EMA/last-turn/max) | 0.468 | 0.468 | anchor 차이 거의 없음 |
| 6 | bge-large (1024 dim, 3× params) | 0.463 | 0.464 | 더 큰 임베딩도 개선 못함 |

**결정적 발견**:
1. **F1 ~0.47가 bge + similarity-based segmentation의 intrinsic ceiling** — 7가지 서로 다른 합리적 접근 모두 동일 천장
2. **HP regime split 필요**: TopiOCQA(빈번한 shift, factoid)는 **SEM2 defaults**(α=10, λ=1, σ₀²=0.1) 유리. Hi-EM persistence 기본값(α=1, λ=10, σ₀²=0.01)은 LongMemEval 같은 긴 chat 가정용
3. **TopiOCQA shift rate 28%**가 precision cap을 ~0.32로 강제 → 어떤 method도 F1 ~0.47 못 넘음

### 4.3 Step 1-4: TopiOCQA Gate ✓ marginal PASS

| 조건 | 값 | 판정 |
|---|---|---|
| Hi-EM F1 > cosine | 0.471 > 0.467 | ✓ |
| Hi-EM F1 > 0.4 | 0.471 | ✓ |
| 턴당 overhead ≤ 200ms | ~20 ms (embed 19.6 + assign 0.29) | ✓ |

**Gate: PASS (marginal)**. TopiOCQA는 Hi-EM 주 타깃 아님 → "최소 동작 sanity check" 의미.

### 4.4 Step 1-5: TIAGE test 측정 ✗ FAIL

**데이터**: PersonaChat 기반 chit-chat, 인간 annotated. test 100 conv / 1564 turns / 315 shifts (shift rate 21.5%/transition, Cohen's Kappa 0.48)

| Method | Precision | Recall | F1 |
|---|---|---|---|
| (a) all-boundary | 0.215 | 1.000 | 0.354 |
| (b) **cosine (θ=0.525)** | 0.332 | 0.575 | **0.421** |
| (c) Hi-EM persistence | 0.245 | 0.451 | 0.317 |
| (c') Hi-EM freq-shift | 0.239 | 0.895 | 0.377 |

**Hi-EM 두 HP 모두 cosine baseline에 명확히 패배**.
- Persistence: 너무 sticky → recall 0.451로 shift 놓침
- Freq-shift: 너무 자주 분할 → precision 0.239로 노이즈

**Gate: FAIL** (Hi-EM F1 < cosine, F1 < 0.4)

Latency만 통과: 0.73 ms/turn (overhead) — A100에서 매우 빠름.

### 4.5 Step 1-6: 종합 Gate ✗ FAIL

| 벤치마크 | Gate |
|---|---|
| TopiOCQA (Phase 1-4) | PASS (marginal) |
| TIAGE (Phase 1-5) | FAIL |

**종합: FAIL** — 두 벤치마크 모두 PASS여야 Phase 2 진입.

---

## 5. 폐기된 작업의 기록 (실패에서 얻은 것)

### Phase 2.5 — LongMemEval session-as-topic-boundary smoke test (폐기)

**발생 → 진단 → 폐기 경위**:
1. Phase 1-3 후 LongMemEval oracle에서 옵션 A 감도 조기 검증을 위해 smoke test 설계
2. Session 경계를 topic 경계 GT proxy로 사용
3. 결과: persistence HP에서 session-boundary recall 0.734 (PASS), within-session purity 0.542 / avg topics/sess 1.96 (둘 다 FAIL — 3.3배 과분할)
4. HP grid sweep (100 configs) 시도 → σ₀² monotonic trade-off만 확인. 어떤 config도 PASS 불가
5. **사용자 지적으로 개념적 오류 발견**: LongMemEval은 turn-level topic label이 없음. 한 세션 내 subtopic 공존이 정상적인 데이터 → Hi-EM의 정상 분할이 FP로 잘못 처벌됨
6. **결론**: LongMemEval = downstream QA 평가용 (Phase 4 4-baseline 비교). topic 경계 감지 평가에 쓰면 안 됨.
7. Phase 2.5 smoke test **폐기**, 산출물·스크립트 삭제 (`context/06-decision-log.md` 2026-04-24 entry로 기록 보존)

**얻은 것**: 평가 축의 명확한 분리 — 토픽 경계 감지(TopiOCQA + TIAGE) vs downstream QA(LongMemEval, LoCoMo).

---

## 6. 핵심 발견 (Findings)

### 6.1 bge + similarity 방법론의 F1 ceiling
TopiOCQA에서 7가지 서로 다른 접근(HP / 구조 / multi-signal / context / anchor / encoder upgrade) 모두 F1 ~0.47에 막힘. **이는 hyperparameter나 알고리즘 선택의 문제가 아니라 임베딩 자체의 intrinsic discriminative ceiling**.
- 다른 topic 간 cos 유사도 ~0.5 (낮지만 충분한 구분 안 됨)
- TopiOCQA shift rate 28% → precision 상한 ~0.32

### 6.2 Hi-EM HP는 benchmark regime별 분기
TopiOCQA(frequent shift, factoid)는 **SEM2 defaults**(α=10, λ=1, σ₀²=0.1) 유리.
Hi-EM persistence 기본값(α=1, λ=10, σ₀²=0.01)은 **persistence regime**(긴 chat, LongMemEval 같은) 가정용.
이건 `02-math-model.md` 하이퍼파라미터 표에 regime split으로 명시.

### 6.3 Hi-EM의 centroid abstraction이 짧은 chit-chat에선 손해
TIAGE(50자 짧은 PersonaChat)에서 **두 HP 모두 cosine baseline에 패배** (0.317~0.377 vs 0.421).
가설: 짧은 utterance는 인접 턴 cosine 직접 비교가 충분히 discriminative한데, Hi-EM의 centroid average + sCRP 추상화는 신호를 smooth out.

### 6.4 두 벤치마크 합쳐 보면 일관 패턴
| | Hi-EM 최고 F1 | Cosine F1 | 격차 |
|---|---|---|---|
| TopiOCQA | 0.471 | 0.467 | +0.004 (실질 동일) |
| TIAGE | 0.377 | 0.421 | −0.044 (Hi-EM 패배) |

→ **Hi-EM의 topic boundary detection 성능은 cosine baseline 대비 차별점 없음** — TopiOCQA는 같은 수준, TIAGE는 더 나쁨.

### 6.5 그렇다면 Hi-EM의 가치는?
**Topic boundary F1은 Hi-EM의 핵심 contribution이 아닐 가능성**. 진짜 가치 가설:
- Online unsupervised **clustering 구조** (cosine threshold는 binary judgment뿐)
- **Topic 표현(centroid)** — downstream retrieval/importance에 활용
- **Memory hierarchy** (LTM/Memory window) — topic ID로 인덱싱
- **다운스트림 QA 효율** — Phase 4에서 4-baseline (Sliding window / Full context / RAG / Hi-EM) 비교로 **유일하게 검증 가능**

### 6.6 검증 미해결 3건 (Phase 0-1에서 식별, 여전히 open)
1. **식 2 `diag(β)` vs 식 11 `τI` covariance 차이의 이론적 유도 부재** — 현재 Hi-EM 옵션 A는 두 분포 모두 동일하게 다루므로 영향 없음. Phase 1+에서 향후 설계엔 영향 가능
2. **SEM2 `prior[k_prev] -= λ/2` halving 유도 부재** — Hi-EM 옵션 A에선 restart-vs-repeat 분기 자체 미포팅이라 무관. 옵션 D 확장 시 재검토 필요
3. **TIAGE 12턴 평균 / TopiOCQA 12턴 평균 → variance 학습 기회 거의 없음** — 본 평가는 사실상 **centroid 부분만 측정**. 옵션 A의 variance 효과는 Phase 4 LongMemEval QA에서 간접 측정하기로 (이건 Phase 1-3에 한계로 명시됨)

---

## 7. 미해결 / 의사결정 대기

### 7.1 Phase 1-6 종합 Gate FAIL → 어디로?

**옵션 A**: TIAGE HP sweep
- TopiOCQA처럼 5분 짧은 실험으로 천장 확인
- 작은 효과만 있어도 PASS 가능성 (현재 cosine 0.421 vs Hi-EM persistence 0.317, 격차 0.1)
- 그러나 TopiOCQA에서 본 패턴 — sweep 후도 cosine 천장과 동등할 가능성 큼

**옵션 B**: Topic boundary F1 ≠ Hi-EM 핵심 가치라는 reframing 인정
- 두 벤치마크 일관 패턴 → Hi-EM의 segmentation F1 우위는 없다는 결론
- decision-log에 정직히 기록 후 Phase 2 진입
- 진짜 가치 검증은 Phase 4 downstream QA에서

**옵션 C**: 옵션 D escalation (multi-signal)
- plan.md FAIL 경로에 명시된 정공법
- 단 TopiOCQA에서 multi-signal 효과 없던 전례 — 비용 대비 이득 낮음 가능성

**옵션 D** (새 제안): Hi-EM의 **likelihood를 cosine-vs-prev-turn**으로 변경
- 현재: log P(s | e=k) = log N(s; μ_k, σ²) — centroid 비교
- 대안: cosine(s, last_turn_in_topic) 기반 — pairwise 비교 (cosine baseline이 강한 이유와 같음)
- sCRP prior는 유지, likelihood만 교체
- 옵션 A의 변형이라 옵션 D보다 작은 변경

**옵션 E** (재평가): TopiOCQA marginal PASS는 그냥 noise 수준 차이일 수도. 두 벤치마크 합쳐 "Hi-EM segmentation 우위 없음"으로 결론짓고 옵션 B 진행

### 7.2 Phase 4 QA accuracy 평가 (미수행)
- Hi-EM의 진짜 차별점이 측정되는 유일한 곳
- Sliding window / Full context / RAG / Hi-EM 4-baseline 비교
- 평가 전 Phase 2 (LTM + Memory window) 구현 필요
- LongMemEval 5개 능력 별 + LoCoMo 모두 측정

### 7.3 Phase 2 LTM 저장 포맷 결정 미완
- JSON / SQLite / Parquet 중 선택 (`context/01-hi-em-design.md`에 미확정 위임)
- Memory window 크기 정책 $K_\text{window}$ 고정 vs 적응적
- Topic importance 공식 구체화

---

## 8. 한계 / 주의

### 8.1 Embedding 의존성
- bge-base-en-v1.5에 강하게 의존. 임베딩 모델 교체 시 σ₀² 등 재튜닝 필요
- bge-large 시도해 봤으나 TopiOCQA에서 개선 없음 (오히려 F1 0.471 → 0.463 하락)
- **언어 한정**: en-v1.5 — 다국어/한국어 적용 시 별도 검토

### 8.2 Latency 측정의 제약
- Phase 1-3/1-5 latency는 **Hi-EM 자체 overhead**만 (embedding + assign)
- LLM 호출 오버헤드와의 상대 비교는 Phase 4에서만 가능
- 현재 측정값: ~20 ms/turn (CPU·GPU 차이 큼). brief.md "+10~20%" 제약 검증은 LLM 1000ms 가정 하에서만 PASS 주장

### 8.3 벤치마크 미스매치
- TopiOCQA(factoid Wiki QA)와 TIAGE(chit-chat) 모두 Hi-EM 주 타깃(Claude-style 긴 chat)과 다른 chat 성격
- LongMemEval은 가장 가깝지만 boundary GT 없음 → 직접 segmentation 평가 불가
- 결과적으로 **Hi-EM의 segmentation 우수성을 main 시나리오에서 정량 검증할 직접 수단 부재**

### 8.4 검증 미해결 (Phase 0-1 식별)
§6.6 참조. 모두 현재 옵션 A 구현엔 영향 없으나 향후 옵션 D 확장 시 재검토 필요.

### 8.5 Hi-EM의 "옵션 A는 잠정적" 성격
Step 0-3 결정문에서도 "단순 baseline 먼저 → Phase 4 실험 후 확장 여지"라 명시. 현재 결과는 그 계약 안에 있음. 옵션 A 영구 확정이 아니라 Phase 1-2 실험을 위한 출발점.

---

## 9. 환경 / 인프라 / 협업 정책

### 9.1 코드 환경
- 로컬: WSL2 Ubuntu + anaconda Python 3.9.12
- Colab: A100 GPU runtime + VSCode remote kernel 연동 (`Ctrl+Shift+P → Jupyter: Specify Server`)
- 의존성: `requirements.txt`에 핀 (transformers<4.50, sentence-transformers<4 — Python 3.9 호환)

### 9.2 파일 구조
```
Hi-EM/
├── brief.md              프로젝트 한 줄 요약
├── plan.md               Phase 0~5 로드맵
├── handoff.md            세션 진입점
├── CLAUDE.md             코딩 규칙 + Step 완료 프로토콜
├── README.md             외부 안내
├── report.md             ← 이 파일
├── context/              설계 문서 (검증된 결정)
├── src/hi_em/            Phase 1-1 구현
├── tests/                pytest 테스트
├── scripts/              실험·분석 스크립트
├── notebooks/            Colab 실험 notebook (setup_colab 선행 가정)
├── outputs/              실험 결과 .md / .json
├── benchmarks/*          외부 benchmark repo (gitignored, 각자 clone)
├── SEM/                  SEM2 참조 (nicktfranklin/SEM2)
└── setup_colab.ipynb     Colab 환경 셋업 (gitignored)
```

### 9.3 협업 정책 (CLAUDE.md 영구 규칙)
- **Claude는 git add/commit/push 직접 실행 금지** — 명령어만 제시, 사용자가 실행
- **모든 실험 notebook은 setup_colab.ipynb 선행 가정** — 환경 셋업 로직 중복 금지
- **Step 완료 직전 3-angle self-audit** (구조/동작/설계 각도 최소 3 Q&A) → 그 다음 `python scripts/check_step_done.py` exit 0 받아야 [x]
- **모든 설계 결정은 `context/06-decision-log.md`에 append-only 기록** (번복도 새 entry로)

### 9.4 Phase 진행 추적
- 검증 게이트: `scripts/check_step_done.py` (Step별 산출물 존재·내용 길이·키워드 검증)
- 현재 자동 감지: Step 1-5 (TIAGE 평가) — Phase 1-6 종합 Gate에서 멈춤 상태

---

## 10. 결정 로그 시간순 요약 (`context/06-decision-log.md` 압축)

| 날짜 | 결정 | 근거 / 결과 |
|---|---|---|
| 2026-04-23 | 프로젝트 초기 설계 방향 | SEM 논문 이해 + no fine-tuning 제약 + 실시간 가정 |
| 2026-04-23 | 사건 모델 옵션 A 확정 + Markov 확장 철회 | 벤치마크 분석 + double counting 회피 |
| 2026-04-23 | Phase 1 범위 재정의 + KV cache→Memory window 용어 전환 | topic 분할 검증을 LTM 설계 이전에 |
| 2026-04-24 | Phase 1-3 TopiOCQA HP regime split 확인, marginal PASS | 7-iteration 탐색으로 천장 식별 |
| 2026-04-24 | LongMemEval-as-boundary-GT 폐기, TIAGE 추가 | 평가 축 분리 (topic 경계 vs downstream QA) |
| **2026-04-25** | **Phase 1-5 TIAGE Gate FAIL — 종합 FAIL** | **Hi-EM segmentation 우위 부재 의심** |

---

## 11. 다음 결정에 도움이 되는 질문

1. **"Topic boundary F1 우위가 Hi-EM의 정의된 contribution인가?"**
   - Yes → 옵션 D escalation 또는 옵션 A 구조 변경 필요 (옵션 D, E, 또는 likelihood 변경)
   - No → 옵션 B (reframing + Phase 2 진입)로 진행. F1 결과는 "옵션 A는 cosine baseline 수준 segmentation"으로 정직히 기록

2. **"Phase 4 downstream QA에서 Hi-EM이 RAG를 이기는 게 증명 가능한가?"**
   - Phase 1-3/1-5 결과가 "그렇지 않을 수 있다"고 시사
   - 그럼에도 시도할 가치 있는가? (구현 비용 vs 학습 가치)
   - **이 질문 답이 "예"면 옵션 B로 즉시 진행, "예측 모름"이면 옵션 A/D로 Phase 1 더 파보기**

3. **"sCRP + topic clustering 자체가 downstream에서 의미 있는 구조를 제공하는가?"**
   - boundary F1이 아니라 **clustering 품질** 측정도 가능 (V-measure, ARI vs session 라벨)
   - 현재 측정 안 했음. Phase 1-7로 추가 가능

4. **"Hi-EM 설계의 어떤 부분을 진짜 검증했고, 어떤 부분은 가정으로 남았는가?"**
   - 검증됨: sCRP 구현 정확성, Welford 정확성, MAP 루프 동작, Latency 충족
   - 가정만: $\sigma^2$ 효과 (cold start로 인해 학습 기회 거의 없었음), Memory window 효율, downstream QA 기여, 벤치마크별 HP 적응성

---

## 12. 다음 행동 후보 (사용자 결정 대기)

| # | 행동 | 비용 | 기대 |
|---|---|---|---|
| 1 | TIAGE HP sweep 5분 실험 | 작음 | 본질 한계 vs HP 문제 구별 |
| 2 | Hi-EM likelihood를 cosine-vs-last-turn으로 교체 후 재측정 | 작음 (옵션 A 변형) | TIAGE에서 cosine baseline 따라잡기 가능 |
| 3 | Phase 1 종결 + 옵션 B 채택 → Phase 2 진입 | 중 (LTM/Memory window 설계) | downstream QA 정직 검증으로 가치 판정 |
| 4 | 옵션 D 본격 escalation (multi-signal 재설계) | 큼 | 새 가설 검증 필요 |
| 5 | Hi-EM의 clustering 품질 (V-measure/ARI) 측정 추가 | 작음 | boundary F1 외 segmentation 가치 측면 |
| 6 | Phase 0의 검증 미해결 3건 중 일부 재검토 | 작음~중 | 옵션 A 구조 정당성 강화 |

---

## 부록 A. 결정 로그 위치

- `context/06-decision-log.md` — 모든 설계 결정 append-only (이 보고서 §10이 압축본)
- `context/00-sem-paper.md` — SEM 논문 정독 + 검증 미해결 3건 (Hi-EM 관점 정리)
- **`context/sem-equations.md`** — SEM 식 1~24 원본 정의 reference (LaTeX + 페이지 + Hi-EM 처리 매핑)
- `context/01-hi-em-design.md` — Hi-EM 설계 확정사항 + Phase 위임 사항
- `context/02-math-model.md` — 수식 + HP regime split 표
- `context/04-benchmarks.md` — 벤치마크 메타정보 + 평가 축 분리 명시

## 부록 B. 산출물 위치

- `outputs/benchmark-analysis.md` — Phase 0-2 벤치마크 분석
- `outputs/phase-1-topiocqa.md` — TopiOCQA F1 + 7-iteration 탐색 이력
- `outputs/phase-1-topiocqa-{sweep,variants,multisignal,anchors,encoder,contextualized}.json` — 각 iteration raw 결과
- `outputs/phase-1-tiage.md` — TIAGE F1 + Gate FAIL

## 부록 C. 탐색 스크립트 위치

- `scripts/run_topiocqa_segmentation.py` — main TopiOCQA eval
- `scripts/run_topiocqa_{sweep,variants,multisignal,anchors,bigencoder,contextualized}.py` — 7-iteration 보조 탐색
- `scripts/run_tiage_segmentation.py` — TIAGE eval
- `scripts/check_step_done.py` — Step 완료 검증 게이트

---

**문서 종료.** 사용자 판단 대기.
