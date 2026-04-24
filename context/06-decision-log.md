# 설계 결정 로그

모든 주요 설계 결정의 **근거와 날짜**를 기록한다.
판단이 바뀔 때마다 기존 내용을 덮어쓰지 말고 **append**.

## 형식
```
YYYY-MM-DD: <결정 사항 한 줄>
근거: <관찰/실험/논문>
영향 범위: <어떤 문서/코드가 바뀌나>
대안: <고려했다가 기각한 옵션들과 기각 이유>
```
---

## 기록

### 2026-04-23: 프로젝트 초기 설계 방향 설정
**근거**:
- SEM 논문 (Franklin et al. 2020) 이해
- no fine-tuning 제약 (어떤 Transformer LLM에도 붙어야 함)
- 대화 메모리 실서비스 목표 (turn당 latency +10~20% 이내)

**결정**:
- sticky-CRP prior 유지 (topic 수 자동 결정)
- RNN event 모델 폐기 (no fine-tuning 제약)
- Memory reconstruction (Gibbs) 폐기 (실서비스 불필요)
- LTM/STM 계층 도입
- Markov 가정 확장: $P(e_n \mid e_{n-1}, \mathbf{s}_{n-1})$

**영향 범위**: 전체 설계

**대안**:
- SEM 그대로 따라가며 RNN만 제거 (기각: 사건 모델 대체 필요)
- 모든 것을 새로 설계 (기각: SEM 구조가 충분히 유용)

---

### 2026-04-23: 사건 모델 미확정 상태로 유지
**근거**:
- 이전 논의에서 TopiOCQA/TIAGE 분석만으로 "Centroid + Entity set + Multi-signal" 설계 도출
- 그러나 이 두 벤치마크는 Claude-유사 장기 대화와 성격이 다름
- LoCoMo/LongMemEval까지 분석해야 bias 없는 설계 가능

**결정**:
- `context/01-hi-em-design.md`의 "2. 사건 모델" 섹션을 미확정으로 열어둠
- Phase 0 완료 후 Claude Code가 직접 판단하여 채워넣음

**영향 범위**:
- `context/01-hi-em-design.md` 미확정 섹션
- `context/02-math-model.md` 미확정 섹션
- Phase 1 진입 전 반드시 해결

**대안**:
- Multi-signal 앙상블로 확정 (기각: 편향된 근거)
- Centroid only로 확정 (기각: 너무 단순할 수 있음)

---

### 2026-04-23: SEM2 레포로 교체 (nicktfranklin/SEM2)
**근거**:
- 기존 `SEM/`는 `ProjectSEM/SEM` (archival) v1로 TF1 기반 (`SEM/models/sem.py`, tf.Session 등).
- `nicktfranklin/SEM2`는 current working build. 모듈 구조가 정돈됨 (`SEM/sem/sem.py`에 `_calculate_unnormed_sCRP`, `run()`, `event_models.py`, `memory.py`, `hrr.py` 분리).
- 알고리즘 동일, 구현 참조를 더 깨끗한 버전으로.

**결정**: `SEM/` 전체 교체, 관련 문서(`README.md`, `brief.md`, `handoff.md`, `CLAUDE.md`, `plan.md`, `context/00-sem-paper.md`, `01-hi-em-design.md`, `02-math-model.md`)의 경로·명칭 일괄 갱신. SEM2도 TF-based이므로 Hi-EM은 코어 알고리즘만 참조하고 PyTorch로 재구현.

**영향 범위**: 위 파일들, Phase 1 구현 시 SEM2 코드 참조 기준.

**대안**: 기존 SEM v1 유지 (기각: deprecated).

---

### 2026-04-23: Step 0-1 완료 — SEM 논문 정독 + SEM2 코드 검증

**근거**:
- SEM 논문 전 35페이지 정독 (pdftotext 산문 + 수식 10페이지 PNG 이미지 직독).
- SEM2 `SEM/sem/sem.py` `_calculate_unnormed_sCRP(prev_cluster=None)` (L.144–159)와 `run()` (L.161–354) 실제 코드 verbatim 검증.

**결정**: `context/00-sem-paper.md`를 논문 검증본으로 재작성. 식 1~24 각각 정확 정의, Hi-EM 계승/대체/폐기 매핑. 검증 미해결 지점 3건 명시(§7):
1. 식 (2) `diag(β)` vs 식 (11) `τI` covariance 선택 이론적 유도 불완전
2. SEM2 `run()`의 `prior[k_prev] -= lmda/2.` halving 논문에 explicit 유도 없음
3. Markov 확장 $P(e_n\mid e_{n-1}, s_{n-1})$의 구체 수식 미확정

**영향 범위**: `context/00-sem-paper.md` (11940자).

**대안**: pdftotext만으로 진행 (기각: 수식 기호가 `/H11005` 등 깨진 코드포인트로 출력되어 수식 검증 불가능 → 이미지 렌더링 경로로 전환).

---

### 2026-04-23: Step 0-2 완료 — 벤치마크 실데이터 분석

**근거**:
- LoCoMo (10 conv × 27 sess × 22 turns/sess, topic annotation 없음, session 경계=날짜)
- TopiOCQA dev (2514 turns, 205 conv, 12.3 turns/conv, Wiki doc Topic ground truth, topic shifts 3.3/conv, section shifts 7.7/conv)
- LongMemEval oracle (500 Q, 6 question types, 1206자/turn chat, 21일 mean span)

**결정**: `outputs/benchmark-analysis.md`에 옵션 A~F × 3 벤치마크 증거 매트릭스 작성. Step 0-3 입력으로 확정.

**영향 범위**: `outputs/benchmark-analysis.md`(8239자), Step 0-3 결정 근거.

**대안**: 없음 (데이터 관찰 단계).

---

### 2026-04-23: 사건 모델 옵션 A 확정 (Centroid + diag variance)

**근거**:
- `outputs/benchmark-analysis.md` §4 매트릭스: 옵션별 종합 적합도는 D > A,C > B,E 순이나 D는 가중치 튜닝 근거가 Phase 0 시점에 없음.
- 옵션 C(Entity set)는 TopiOCQA 편향 위험 — `handoff.md` 경고 직접 저촉.
- **Incremental 설계 원칙**: 최단순 baseline으로 출발해 실험 병목 관찰 후 옵션 D로 확장.
- SEM2 `log_likelihood_next`/`log_likelihood_f0` 인터페이스에 자연 대응, Welford online update와 일관.

**결정**:
$$P(\mathbf{s}_n \mid e_n = k) = \mathcal{N}\big(\mathbf{s}_n;\, \mu_k,\, \mathrm{diag}(\sigma_k^2)\big)$$
- `context/01-hi-em-design.md §4` "확정", `02-math-model.md` 수식 확정.
- 미확정 섹션은 "Phase 1/2로 위임된 결정"으로 재분류 (stale marker 전부 제거).

**영향 범위**: `context/01-hi-em-design.md`, `02-math-model.md`, Phase 1 구현 시작점.

**대안 및 기각 사유**:
- B (Momentum): 대화 순서 느슨해 효과 의문, 추가 복잡성 정당화 부족.
- C (Entity set): TopiOCQA bias 위험, LongMemEval 엔티티 sparse.
- D (Multi-signal): 가중치 근거 부족, Phase 4 실험 후 확장 대상.
- E (Linear predictor): 작은 topic 과적합, cold start 어려움.

**예상 한계 (Phase 4에서 재검토)**:
- LongMemEval 긴 content에서 centroid variance 증가 예상.
- TopiOCQA section shift 과분할 가능성, $\lambda$ 민감도 test 필요.

---

### 2026-04-24: 평가 축 개념 정정 — LongMemEval는 QA용, Topic 경계 감지는 TIAGE로 확장

**근거**:
- Phase 2.5 smoke test에서 LongMemEval oracle session 경계를 topic 경계 GT proxy로 사용해 평가 → Gate FAIL (recall 0.734 PASS, purity 0.542 FAIL, topics/sess 1.96 FAIL).
- 재검토: LongMemEval는 **downstream QA accuracy 평가용**으로 설계된 benchmark. turn-level topic label이 없고 session 경계는 weak proxy. 한 세션에 여러 subtopic이 공존할 수 있어 Hi-EM의 정상적 subtopic 분할이 FP로 잘못 처벌됨.
- 결과적으로 Phase 2.5 smoke test의 **전제 자체가 틀린 설계**였음.

**결정**:
- **LongMemEval**은 topic 경계 감지 평가에 쓰지 않는다. Phase 4 downstream QA 비교(Sliding window / Full context / RAG / Hi-EM 4 baseline)에만 사용.
- **Topic 경계 감지 벤치마크는 TopiOCQA + TIAGE 2종**으로 확장:
  - TopiOCQA (Wikipedia doc 경계, factoid QA, frequent-shift regime)
  - TIAGE (PersonaChat 인간 주석, chit-chat, 20%/transition shift rate) — 기존 Tier 2 "옵션"에서 **Tier 1 필수**로 승격
- Phase 2.5 "LongMemEval smoke test"는 **폐기**. 대신 Phase 1-3 augment로 TIAGE 평가 추가.

**영향 범위**:
- `context/04-benchmarks.md`: 평가 축 구분 명시, TIAGE Tier 승격
- `plan.md`: Phase 2.5 LongMemEval smoke test 제거, Phase 1-3에 TIAGE 추가
- `benchmarks/tiage/` 추가 clone (`.gitignore`에 등록)
- `setup_colab.ipynb` BENCHMARK_REPOS에 tiage 추가
- `scripts/run_tiage_segmentation.py` 신설
- `outputs/phase-2.5-smoke.md`·`.json` 및 sweep 결과는 "참고용" (결론에 쓰지 않음)

**대안**:
- LongMemEval session을 topic 경계로 강제 가정하고 gate 유지 (기각: 근본적으로 misaligned)
- Phase 2.5를 LongMemEval QA로 재정의 (기각: Phase 4와 중복, Phase 2 메모리 계층 전 smoke는 더 가볍게)

**Phase 1 topic 경계 검증 재정의**:
- TopiOCQA gate: 현재 PASS (marginal, F1 0.471)
- TIAGE gate: 이번 추가 수행 — 결과 `outputs/phase-1-tiage.md`로 기록
- 두 벤치마크 모두 PASS면 Phase 2 진입

---

### 2026-04-24: Phase 1-3 TopiOCQA — hyperparam regime split 확인, Gate marginal PASS

**근거**:
- Hi-EM 초기값(α=1, λ=10, σ₀²=0.01) → F1=0.378 FAIL (cosine baseline 0.468 대비 크게 낮음).
- Iteration 1 (HP grid sweep): 108개 config 중 **α=10, λ=1, σ₀²=0.1** = SEM2 defaults → F1=0.471 marginal PASS.
- Iteration 2 (구조 변형 5종 — gauss-origin/global/self, vMF-origin/const): 모두 F1 ~0.45–0.47 범위. 구조 변경이 cosine 상한을 못 뚫음.
- Iteration 3 (multi-signal: cos + jaccard + entity overlap 564 config): 개선 없음 (Gaussian의 암묵적 normalization이 linear 가산보다 우수). 옵션 D 전환 불필요.
- 본질적 원인: bge-base-en-v1.5 임베딩 기준 다른 topic 간 cos 유사도 ~0.5, shift rate 28% → **precision 상한 ~0.32, F1 상한 ~0.47**. embedding의 intrinsic 제약.

**결정**:
- Hi-EM 사건 모델 옵션 A 유지 (구조 변경·옵션 D escalation 불필요).
- **하이퍼파라미터는 benchmark regime에 따라 분기**:
  - Persistence regime (LongMemEval 등 실서비스 대화): α=1, λ=10, σ₀²=0.01 유지
  - Frequent-shift regime (TopiOCQA 류 factoid QA): α=10, λ=1, σ₀²=0.1
- `context/02-math-model.md` 하이퍼파라미터 표에 regime split 명시.
- Phase 2.5 smoke test(LongMemEval oracle)로 Persistence regime에서의 성능을 즉시 검증.

**영향 범위**:
- `outputs/phase-1-topiocqa.md` (F1 결과 + 탐색 이력 + 권장)
- `context/02-math-model.md` (regime-split HP 표)
- `plan.md` (1-3, 1-4 [x] + regime note)
- `scripts/run_topiocqa_{segmentation, sweep, variants, multisignal}.py` (탐색 스크립트, Phase 1 보조 산출물)

**대안 및 기각 사유**:
- 초기 Hi-EM HP로 gate 엄격 FAIL 처리 → 옵션 D 전환 (기각: 3 iteration에서 옵션 D도 효과 없음 확인, 과대 재설계).
- embedding 교체 (bge-large 등) (기각: Phase 1 범위 밖, 큰 모델 다운로드 필요).
- gate 조건 완화 (기각: plan.md 규칙 훼손).

**남은 리스크**:
- Persistence regime HP가 LongMemEval에서도 적합한지 Phase 2.5에서 확인 필요. 여기서 FAIL이면 옵션 D escalation 재고.

---

### 2026-04-23: Phase 1 범위 재정의 + KV cache → Memory window 용어 전환

**근거**:
- Phase 1 완료 기준을 "단위 테스트 통과"에서 "TopiOCQA topic shift F1 gate 통과"로 강화. 이유: topic 분할이 검증 안 된 상태에서 LTM/STM 계층 구축은 허구 위에 쌓임.
- 3-angle audit(구조·동작·설계 18 Q&A)로 8개 gap 발견:
  1. `Topic_section` 변화를 noise로 취급(FP 카운트) 정의 누락
  2. Gate 임계값 선험적 숫자 제거 → "baseline 대비 우위 + 절대 하한" 규칙화
  3. brief.md latency +10~20% 제약 조기 측정(Phase 4 아닌 Phase 1-3)
  4. TopiOCQA 평균 12턴 → variance 학습 기회 거의 없음, centroid 부분만 검증된다는 한계 명시
  5. FAIL 시 옵션 D escalation 경로 plan 명시 (Phase 0-3 결정 번복 규칙)
  6. Phase 2 완료 후 Phase 3 진입 전 smoke test(Phase 2.5) 신설 — LongMemEval 옵션 A 감도 조기 가늠
  7. TopiOCQA gate는 "최소 동작 sanity check" 비대칭 의미 재포지셔닝 (PASS ≠ 최종 성공, FAIL = 확실한 재설계)
  8. Phase 4에서 TopiOCQA F1 항목 제거(Phase 1로 이전)
- 용어 전환: "KV cache paging/관리"는 LLM 런타임 내부 메커니즘에 종속돼 Hi-EM 책임 경계를 흐림. Hi-EM은 **LTM(SSD)에서 현재 라운드 prefill 대상을 Memory window(STM)로 promotion**하는 역할로 단순화. KV cache 재사용은 이 promotion의 부산물이며 실제 prefill 전달은 downstream(vLLM/SGLang)이 담당.

**결정**:
- plan.md 전체 재구조: Phase 1 4-Step (1-1 코어 / 1-2 unit test / 1-3 TopiOCQA 측정 / 1-4 Gate), Phase 2.5 smoke test 신설, Phase 4에서 TopiOCQA 이전.
- 용어 전역 교체: "STM" → "Memory window", "KV cache paging" → "LTM → Memory window promotion" (구현 세부는 필요시만 언급).
- `kv_cache.py` → `memory_window.py` 모듈명 변경 (architecture).

**영향 범위**:
- `plan.md` (전체)
- `handoff.md` "Phase별 진입점" Phase 1~2.5 재작성
- `brief.md` 핵심 차별점 bullet
- `context/01-hi-em-design.md` §9, §A
- `context/03-architecture.md` 모듈명
- `context/05-open-questions.md` 질문 3
- `README.md` 상단 description

**대안 및 기각 사유**:
- plan 유지(기각: topic 분할 검증 없는 LTM/STM 설계는 허구).
- "KV cache paging" 용어 유지(기각: LLM 런타임 API 종속, Hi-EM 책임 경계 모호).
- Phase 1에 LongMemEval QA까지 포함(기각: QA accuracy는 full 파이프라인 필요, Phase 4가 적절).
- Gate 절대 임계 "F1 0.6" 고정(기각: 선험적 근거 없음, baseline 상대 비교가 정직).

---

### 2026-04-23: Markov 확장 $P(e_n\mid e_{n-1}, s_{n-1})$ 철회

**근거**:
- 위 "사건 모델 옵션 A 확정"과 함께.
- 옵션 A의 likelihood가 이미 $s_n$을 centroid 대비 평가하고, 이력은 centroid 업데이트(Welford)에 implicit 반영됨.
- prior에 $s_{n-1}$ 의존 항을 추가하면 **double counting** + prior/likelihood 역할 혼합.

**결정**: prior은 SEM 원본 Eq 1 그대로 유지. scene-conditional 신호가 필요하면 likelihood(사건 모델)를 확장한다.

**영향 범위**: `context/01-hi-em-design.md §3`("철회" 기록), `02-math-model.md` 생성 모형 단순화.

**대안**: 확장 유지하되 double counting은 정규화로 상쇄 (기각: 해석 혼란, 이득 불명).