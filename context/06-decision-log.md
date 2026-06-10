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

### 2026-04-25: Notebook ↔ Script 분리 원칙 명문화 + 파일 cascade 검사 규칙 추가

**근거**:
- `notebooks/phase-1-tiage.ipynb` commit 후 사용자 확인: 원래 의도는 "notebooks/ 디렉토리 통째로 지워도 로컬에서 동작 가능"이었음. setup_colab.ipynb만 gitignored였던 것은 그 정책의 일부였으나, 그 외 ipynb 처리에 대한 명시 규칙 부재로 혼동 발생.
- 검증 결과: 현재 `notebooks/phase-1-tiage.ipynb`는 `subprocess.run([sys.executable, 'scripts/run_tiage_segmentation.py', ...])`로 실제 로직을 scripts에 위임하는 **얇은 wrapper**. 즉 portability는 이미 자연스럽게 달성된 상태.
- 별개로 사용자 지적: Claude가 한 파일 수정 시 다른 파일들의 stale 가능성을 즉시 검사하지 않아 `plan.md` / `handoff.md` / `README.md`가 outdated 상태로 commit되는 문제가 반복됨 (Phase 1-5 직후 재발견).

**결정** (CLAUDE.md 영구 규칙으로 추가):
1. **파일 수정 시 최신성 cascade 검사**: 파일 1개 수정·생성·삭제마다 README/plan/handoff/04-benchmarks/03-architecture/06-decision-log/sem-equations/report/.gitignore 등에 영향 가능성 즉시 검사 → 발견 시 사용자에게 일괄 업데이트 여부 확인.
2. **Notebook ↔ Script 분리 원칙**: 모든 실험 로직은 `scripts/*.py`에 있고 `notebooks/*.ipynb`는 그것을 호출하는 얇은 wrapper. notebooks/ 삭제해도 프로젝트 동작 보장. portability 원칙 위반 시 무효 → script로 분리.
3. **`.ipynb` tracking 정책 확정**:
   - `setup_colab.ipynb` → **gitignored** (Colab 전용, 일회성)
   - 그 외 `notebooks/*.ipynb` → **git tracked** (연구 기록, 협업자 공유). 단 portability 원칙 충족 시에만.

**영향 범위**:
- `CLAUDE.md`: "파일 수정 시 최신성 cascade 검사" 섹션 신설 + "Notebook 실행 정책"에 portability 원칙 + tracking 정책 sub-sections 추가
- `report.md` §9.2 디렉토리 구조: setup_colab.ipynb 위치 정정(`notebooks/` 안), portability 원칙 명시
- `report.md` §9.3 협업 정책: cascade 규칙 + .ipynb tracking 정책 항목 추가
- `README.md` 디렉토리 트리: notebooks/에 portability 코멘트 + setup_colab 위치 표시 정정

**대안 (기각)**:
- 모든 `.ipynb`를 gitignore 확장 (사용자 검토 후 기각: phase-1-tiage 같은 연구 기록은 협업자에게 공유 가치 있음)
- 모든 `.ipynb`를 별도 외부 레포로 분리 (기각: 중첩 git 관리 부담)

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

---

### 2026-04-25: TIAGE HP sweep 종료 + 옵션 1 폐기 → 옵션 5+3 권장 경로 채택

**근거**:
- Phase 1-6 종합 Gate FAIL 후 5 후보(`report.md §12`) 중 옵션 1(TIAGE HP sweep)을 가장 먼저 처리. TopiOCQA만 sweep best로 보고된 비대칭 해소 + reframing 논증의 결정적 증거 확보 목적.
- `scripts/run_tiage_sweep.py` 작성 (TopiOCQA sweep mirror, 108 configs: 3α × 6λ × 6σ²) → TIAGE test 실행 (4.3s, embedding 캐시).
- 결과: **best Hi-EM F1 = 0.383 (α=10, λ=3, σ₀²=0.1) vs cosine baseline 0.421**. Gate 두 조건 모두 FAIL (Hi-EM > cosine: False, -0.038; Hi-EM > 0.4: False).
- Top 10 패턴 분석: 모두 α=10 (새 토픽 자주 생성), λ 다양(0.0~3.0, stickiness 영향 약함), σ₀² 큰 값 선호(0.05~0.1, likelihood gating 약화) → high recall(0.7~0.96) / low precision(0.23~0.26) → **over-segmentation 패턴**. TIAGE chit-chat의 boundary 신호 자체가 흐려서 sticky-CRP 정밀 분할이 본질적으로 불가능.

**결정**:
- 옵션 1 폐기 (Gate FAIL 확정, 추가 시도 무의미).
- 권장 경로 = **옵션 5 → 옵션 3** 묶음. 옵션 5(V-measure/ARI)로 boundary F1 외 unsupervised metric 보완 → 옵션 3(Phase 2 reframing 진입)에서 "boundary F1은 sanity check일 뿐, 진짜 가치는 downstream QA"라는 논증을 정직하게 기록.
- 옵션 2(likelihood 교체)·4(multi-signal 옵션 D)는 보류. Phase 4 결과 후 필요 시 재고.

**영향 범위**:
- `handoff.md` 현재 상태 + 다음 할 일 갱신
- `plan.md` Phase 1-6 결정 분기 #1 [x] 처리
- `report.md §7.1` 옵션 A에 sweep 결과 + 패턴 분석 추가
- `scripts/run_tiage_sweep.py` (신규)
- `outputs/phase-1-tiage-sweep.json` (신규, sweep 자동 생성)
- `context/03-architecture.md` scripts 목록에 sweep 추가

**대안 및 기각 사유**:
- 옵션 1을 Phase 5(논문 실험) 직전으로 미루기 (기각: 옵션 3 reframing 논증의 핵심 증거가 sweep 결과인데, 그게 없으면 reframing이 약한 주장이 됨. 지금 처리해야 옵션 3 진행 가능).
- 옵션 4(multi-signal) 우선 (기각: TopiOCQA에서 multi-signal 효과 약함 전례, scope creep 위험).
- TIAGE에 likelihood 교체(옵션 2)부터 시도 (기각: HP가 본질 한계인지 likelihood 형식이 본질 한계인지 분리 측정 어려움. 먼저 HP 천장 확인 = 옵션 1이 정공법).

---

### 2026-04-25: 옵션 5 (clustering quality) 완료 — 가설 반박 + Phase 2 HP 선택 근거 확보

**근거**:
- `scripts/run_clustering_quality.py` 작성, sklearn V-measure/NMI/ARI/homogeneity/completeness 측정. TopiOCQA dev + TIAGE test 두 벤치마크 × cosine baseline (V-measure best θ) + Hi-EM 두 HP regime (freq-shift 또는 sweep-best, persistence).
- 결과 (`outputs/phase-1-clustering-quality.md` 표 참조):
  - **모든 metric에서 cosine 우위** (TopiOCQA cosine ARI=0.488 vs Hi-EM best 0.398; TIAGE cosine ARI=0.568 vs Hi-EM best 0.397).
  - V-measure / NMI 차이는 작음(~0.01-0.02), ARI 차이는 큼(0.1+).
  - Hi-EM 패턴: homogeneity↑ / completeness↓ → over-segmentation (sweep top 10 패턴과 일치).
- **새 발견**: Boundary F1 ↔ ARI **trade-off**:
  - freq-shift HP (α=10): TopiOCQA F1=0.471 (best) / ARI=0.187 (worst)
  - persistence HP (α=1): F1=0.378 / ARI=0.398 (Hi-EM best)
  - α↑ → boundary 정확도↑ + over-cluster → ARI↓; α↓ → boundary 둔감 + 묶기 보존 → ARI↑

**결정**:
- 원래 가설("Hi-EM은 boundary F1에 약해도 토픽 ID 부여 우위") **반박** 기록.
- **Phase 2 LTM/Memory window 설계 시 persistence HP (α=1, λ=10, σ₀²=0.01) 채택** — 메모리 시스템 관점에선 cluster 보존성(completeness/ARI)이 boundary 정확도보다 중요. 같은 토픽 복귀 시 같은 메모리 호출이 핵심 가치.
- 옵션 3 (Phase 2 reframing 진입) **강화된 논증**: "어떤 unsupervised segmentation metric으로도 Hi-EM의 가치 증명 불가. **그러나 메모리 시스템 관점에선 persistence HP가 cluster 보존성 우위**. 진짜 가치는 Phase 4 downstream QA에서만 검증 가능."

**한계**:
- TopiOCQA/TIAGE는 평균 12~15턴, 토픽 복귀 거의 없음 → Hi-EM centroid 비교 우위 시나리오(긴 대화 + 복귀)가 데이터에 없음. Phase 4 LongMemEval/LoCoMo에서만 검증 가능.
- TIAGE GT cluster ID는 binary shift label에서 derive (sequential 가정) — cosine baseline 형식과 동일하므로 baseline에 형식적 유리. Hi-EM도 같은 GT 기준이라 비교 자체는 fair.

**영향 범위**:
- `scripts/run_clustering_quality.py` (신규)
- `outputs/phase-1-clustering-quality.json` (raw)
- `outputs/phase-1-clustering-quality.md` (해석)
- `handoff.md` 현재 상태 + 다음 할 일 (옵션 5 ✅, 옵션 3 권장)
- `plan.md` 옵션 5 [x] + persistence HP Phase 2 채택 메모
- `report.md §12` 옵션 5 결과 한 줄
- `context/03-architecture.md` scripts 추가
- `README.md` 디렉토리 + 현재 상태

**대안 및 기각 사유**:
- TopiOCQA만 측정 (기각: TIAGE도 함께 봐야 두 벤치마크 일관성 확인 가능. 1분 추가만 필요).
- HP 단일 측정 (freq-shift만) (기각: persistence HP 추가 측정으로 trade-off 발견 — 큰 발견).
- 다른 cluster baseline (k-means on embeddings 등) 추가 (기각: scope creep, cosine sequential이 baseline으로 충분).

---

### 2026-04-25: Phase 2 진입 (옵션 3 reframing) + Step 2-1 LTM 저장 포맷 확정

**근거**:
- Phase 1-6 옵션 1·5 종료 후 옵션 3(Phase 2 reframing 진입) 채택. boundary F1·ARI 모두 unsupervised metric으로는 Hi-EM 가치 증명 불가 → **진짜 가치는 Phase 4 downstream QA에서만 검증 가능**. Phase 2 LTM/Memory window 구현 후 Phase 4 진입이 정공법.
- LTM 저장 포맷 후보: JSON / SQLite / Parquet / Hybrid(JSONL+npy memmap). 결정 요인: 매 턴 append (write), centroid cosine top-k (read), 10k turns 미만 스케일, 단일 process (no concurrency), debug 우선.

**결정**:
- **포맷: per-conversation JSONL (embedding inline, append-only) + `<conv_id>.state.json` (topic 상태 latest snapshot, overwrite)**
- **디렉토리**: `data/ltm/` (gitignored)
- **Turn 스키마**: `{turn_id, ts, role, text, embedding[768], topic_id, is_boundary}`
- **Topic state**: `{conv_id, n_turns, topics: [{topic_id, centroid, variance, count, first_turn_id, last_turn_id}]}`
- **Topic 분할 HP**: persistence (α=1, λ=10, σ₀²=0.01) 채택 — 옵션 5에서 ARI/completeness 우위 (cluster 보존성)가 메모리 시스템 가치와 정합.

**영향 범위**:
- `context/01-hi-em-design.md §9.1` 신규 섹션 + `§A` 한 줄 [확정] 처리
- `plan.md` Phase 2 Step 2-1 [x] + 2-2~ 명시
- `.gitignore` `data/` 추가
- `handoff.md` 다음 할 일 → Step 2-2 (`src/hi_em/ltm.py` 구현)
- `context/03-architecture.md` 향후 추가 모듈 표시 (`ltm.py`, `memory_window.py`)

**대안 및 기각 사유**:
- SQLite (기각: index 이점은 over-engineered, cat/grep 디버깅 손실, sqlite3 CLI 추가 학습 비용).
- Parquet (기각: append 비효율 — rewrite, pyarrow 의존성 추가, 바이너리라 디버깅 어려움).
- Hybrid (JSONL+npy memmap) (기각: embedding 30% 절약하나 두 파일 동기화 부담 + idx 매핑 복잡도. 50MB 절약 가치 없음).
- 전역 1 file (기각: multi-conversation lock/seek 부담, 손상 risk 전파).
- Topic state도 append-only (`.topics.jsonl`) (기각: centroid는 매 턴 변하므로 history 가치 낮음, 디버깅 필요 시 future 변경).
- HP freq-shift (α=10) 채택 (기각: ARI 0.187·0.314로 메모리 보존성 약함. 옵션 5 trade-off에서 명확).

**Phase 5 직전 재검토 트리거**:
- Phase 4 read 병목 시 → Hybrid/SQLite 교체.
- 100k turns 이상 시 → Parquet 검토.

---

### 2026-04-25: Phase 3-1 LLM 백엔드 = OpenAI-compatible (OpenRouter + vLLM 기본)

**근거**:
- Phase 3 orchestrator가 LLM을 호출해야 하므로 어댑터 결정 필요. CLAUDE.md "no fine-tuning" 원칙 + 모델 변수 비교(GPT-4o / Claude / Llama) 위해 외부 API 추상화.
- 사용자 결정 (2026-04-25): OpenAI Chat Completion API 템플릿 + API key 인증 + 기본 endpoint = OpenRouter / vLLM. 둘 다 OpenAI-compatible이라 어댑터 1개로 충분.

**결정**:
- 모듈: `src/hi_em/llm.py` — `OpenAIChatLLM(api_key, base_url)` + `chat(messages, model, **kwargs) -> str`
- 의존성: `openai>=1.30` (requirements.txt 활성화, 설치된 버전 2.32.0)
- 인증 env var: `OPENAI_API_KEY` + `OPENAI_BASE_URL` (생성자 인자 우선, env fallback)
- model 인자는 호출 시 명시 (default 없음 — caller 책임)
- sampling 인자는 `**kwargs` 통과 (temperature, max_tokens 등 OpenAI SDK 그대로)
- 비-streaming, 에러 처리 최소 (raise as-is)
- Anthropic SDK·Google SDK 직접 사용 금지 (OpenRouter 경유)

**영향 범위**:
- `requirements.txt` openai>=1.30 활성화
- `src/hi_em/llm.py` (신규)
- `tests/test_llm.py` (5 tests, mock client)
- `src/hi_em/__init__.py` export
- `context/03-architecture.md`
- `plan.md` Phase 3 → Step 3-1/3-2/3-3 분할
- `handoff.md`
- `README.md`
- `memory/project_llm_backend.md` (project memory)

**대안 및 기각 사유**:
- 함수 1개 (`call_chat(...)`)만 (기각: client config 매번 재생성, env var 읽기 중복).
- 클래스 생성 시 model 고정 (기각: 한 orchestrator가 여러 model 비교 시 불편).
- streaming 즉시 지원 (기각: 현 단계 단순성 우선, downstream 평가에 streaming 불필요).
- `httpx` 직접 호출 (기각: 의존성 줄이는 이득 < 유지비. openai SDK는 안정적).
- Anthropic SDK 별도 어댑터 (기각: OpenRouter로 충분, scope creep. 필요해지면 future 추가).
- env var 분리 (`OPENROUTER_API_KEY` / `VLLM_BASE_URL`) (기각: 두 backend 동시 사용 시나리오 없음 — 한 번에 하나 선택. `OPENAI_*` 표준이 OpenAI SDK 자체 default와 정합).

**Phase 4/5 재검토 트리거**:
- streaming 필요 시 → `chat_stream(...)` 추가.
- retry/timeout/rate-limit 처리 필요 시 → wrapper 추가.
- 다른 backend (직접 Anthropic, Google) 필요 시 → 별도 adapter, 단 OpenRouter로 가능한지 먼저 확인.

---

### 2026-04-25: Step 3-3 smoke test PASS + response_filter (think strip) 옵션 추가

**근거**:
- vLLM 로컬 + Qwen/Qwen3-8B로 A→B→A 시나리오 실행. 결과: Turn 1·3 same `topic_id=0`, Turn 2 boundary 정확, Turn 3 LLM 응답이 Turn 1 정보(가을 시기) 명시 인지 → **memory window가 정확히 Turn 1을 prefill로 승격, LLM이 사용**.
- Qwen3-8B는 reasoning model — 응답에 `<think>...</think>` 블록 포함. 그대로 LTM에 저장하면 **다음 turn prefill 토큰 낭비**.
- 환경변수 관리: `.env` (gitignored) + `.env.example` (tracked, 협업자 안내) + `python-dotenv`. 라이브러리 코드 (`hi_em/*.py`)는 환경 변경 안 함, entry point 스크립트에서만 `load_dotenv()`.

**결정**:
- `HiEM.__init__`에 `response_filter: Callable[[str], str] | None` 옵션 추가:
  - **caller에 raw 응답 반환** (사용자가 thinking 보고 싶다면 그대로)
  - **LTM에는 filtered 응답 저장** (다음 prefill 토큰 절약)
- smoke_test 스크립트에 `strip_think_tags` helper + `--no-strip-think` 플래그.
- `TOKENIZERS_PARALLELISM=false` setdefault (HF tokenizers fork 경고 silence).
- 환경변수: `.env` 파일 + python-dotenv, entry point에서만 load.

**영향 범위**:
- `src/hi_em/orchestrator.py` (response_filter 인자)
- `tests/test_orchestrator.py` (test 1개 추가, 49/49)
- `scripts/smoke_test_orchestrator.py` (신규)
- `outputs/phase-3-smoke.md` (실 LLM trace)
- `.env.example` (신규), `.gitignore` (.env 추가), `requirements.txt` (python-dotenv 추가)
- `plan.md` Step 3-3 [x], `handoff.md`, `README.md`, `context/03-architecture.md`

**대안 및 기각 사유**:
- `strip_think: bool` 단순 flag (기각: thinking 형식이 model마다 다름 — `<think>` / `<thinking>` / `<reasoning>` 등. callable 주입이 더 일반적).
- 라이브러리에서 자동 think strip (기각: 라이브러리는 기본적으로 데이터 변형 안 하는 게 안전. 명시적 opt-in).
- `python-dotenv` 라이브러리 코드에서 자동 load (기각: 라이브러리는 환경 변경 부작용 없어야. entry point 책임).
- `.env`를 git tracked (기각: API key 유출 위험).
- direnv 사용 (기각: 추가 도구 학습 비용, python-dotenv가 더 보편적).

**Phase 4 진입 자격 충족** — 모든 단위 테스트 통과 + 실 LLM end-to-end trace 검증.

---

### 2026-04-25: 환경 관리 uv-native 마이그레이션 (`pyproject.toml` + `uv sync`)

**근거**:
- 현재 `requirements.txt` + `uv pip install` 방식은 (a) 버전 잠금 없음 → 재현성 약함, (b) deps 추가 시 `requirements.txt` 수동 갱신 필요, (c) `conftest.py` sys.path hack 의존.
- uv-native (`pyproject.toml` + `uv.lock` + `uv sync`)는 (a) lock 파일로 정확한 버전 재현, (b) `uv add X`로 자동 lock+pyproject 갱신, (c) editable install로 `hi_em` 모듈 자연 import.
- 협업자 onboarding: `uv sync` 한 줄로 모든 환경 복원.

**결정**:
- `pyproject.toml` 작성 (project metadata + dependencies + dev group + hatchling build + tool.pytest)
- `uv sync` 실행 → `.venv` 재생성 + `uv.lock` (406KB) 자동 생성
- `requirements.txt` **삭제** (uv 단일 source-of-truth)
- `conftest.py` **삭제** (editable install로 sys.path hack 불필요)
- `uv.lock` git tracked (재현성), `.venv`는 그대로 gitignored
- `[tool.uv]` `python-preference = "only-managed"` — uv-managed Python 강제 (lzma 등 stdlib 빌드 누락 회피)

**영향 범위**:
- `pyproject.toml` (신규)
- `uv.lock` (신규, git tracked)
- `requirements.txt` (삭제)
- `conftest.py` (삭제 — editable install로 `hi_em` 자동 import)
- `README.md` 빠른 시작 / 디렉토리 구조
- `handoff.md` 환경 셋업 섹션
- 향후 deps 추가는 `uv add X` (`pyproject.toml` + `uv.lock` 자동 갱신)

**검증**: `uv sync` 후 전체 테스트 회귀 **51/51 PASS** (2.80s).

**대안 및 기각 사유**:
- `requirements.txt` 유지 (기각: 두 source 동기화 부담, 단일 source가 정공법).
- `conftest.py` 유지 (기각: editable install로 자연 해결되는데 hack 유지할 이유 없음).
- `[project.optional-dependencies]` (기각: PEP 735 `[dependency-groups]`가 더 modern, uv 권장).
- `pyproject.toml` 라이브러리 mode 안 함 (기각: `[build-system]` + hatchling으로 editable install이 깨끗).

---

### 2026-04-26: Phase 4 W&B logging 메트릭 설계 — codex review 후 확정

**근거**:
- Phase 4가 Hi-EM 가치 증명 또는 정직한 폐기의 마지막 게이트. 메트릭 설계가 잘못되면 결론 자체가 흔들림.
- 1차안: accuracy + prefill_n_msgs/tokens + latency + (Hi-EM) n_topics/boundary_count/query_assigned_topic/selected_in_window. Visualizations: bar/heatmap/scatter/histogram/table.
- Codex review 받음 — 6개 review 항목 답변. 핵심: (1) `topic_revisit_hit_rate` 추가 (A→B→A 가치 직접 측정 — smoke test에서 단위 확인됐지만 정량 없음), (2) `error_or_empty_hypothesis_rate` 추가 (run_longmemeval.py가 예외 시 empty hypothesis 작성 → raw accuracy가 fragility 숨김), (3) over-designed metric 4개 drop, (4) tokenizer는 실제 Qwen chat template (15% 휴리스틱 오차는 효율 claim 약화), (5) run-per-method 구조, (6) latency histogram → scalar p50/p95.

**결정**:
- **per-question**: `accuracy`, `prefill_n_msgs`, `prefill_tokens`, `latency_sec`, `is_empty`, (Hi-EM only) `topic_revisit_hit`
- **summary**: `accuracy_overall`, `accuracy_by_qtype/{5축}`, `prefill_tokens_{avg,p50,p95}`, `latency_sec_{avg,p50,p95}`, `error_or_empty_rate`, (Hi-EM only) `topic_revisit_hit_rate`
- **config**: method, model, dataset, alpha/lmda/sigma, k_topics, k_turns_per_topic, sliding_k, rag_k, workers, limit, temperature, max_tokens
- **W&B 구조**: run-per-method, group=`<dataset_stem>-<UTC ts>` → 4 method 자동 비교 view
- **Run↔Judge 연결**: `<output>.wandb-run-id` sidecar 파일 → judge가 같은 run에 accuracy 추가
- **Tokenizer**: `transformers.AutoTokenizer.from_pretrained(model)` + `apply_chat_template` (정확). lazy cache.
- **No-op fallback**: WANDB_API_KEY 미설정 시 `WandbRun`이 모든 op no-op (스크립트 정상 동작)

**구현 영향**:
- `pyproject.toml`: `wandb>=0.26.1` 추가
- `src/hi_em/eval_logging.py` (신규): WandbRun, count_prefill_tokens, aggregate_summary
- `src/hi_em/orchestrator.py`: `handle_turn(return_debug=True)` 옵션 추가 (test 1개)
- `scripts/run_longmemeval.py`: baseline 함수 4개 모두 `(response, messages, extras)` tuple 반환. Hi-EM은 `topic_revisit_hit` 계산. wandb hooks.
- `scripts/judge_longmemeval.py`: sidecar resume + accuracy backfill
- `tests/test_eval_logging.py` (4 tests) → 전체 56/56 PASS
- `.env.example`: WANDB_API_KEY/PROJECT 추가
- `handoff.md`/`plan.md`: Step 4-4a 완료 + W&B 결과 활용법

**대안 및 기각 사유**:
- 1차안 그대로 유지 (기각: codex가 반박한 "Hi-EM 차별화 측정에 직접 metric 빠짐" 정당함).
- judge 결과를 별도 wandb run으로 (기각: 같은 method의 accuracy/efficiency가 다른 run이면 비교 view 깨짐).
- 토크나이저 휴리스틱 (`split * 1.3`) (기각: codex 지적 — 15% 오차는 "동일 accuracy + 적은 tokens" 효율 claim 정합성 깨짐).
- 4 method 한 run에 묶기 (기각: codex 지적 — W&B sweep/filter는 method가 config 필드일 때 자연. 현재 호출 구조와 일치).
- W&B 의존성 hard requirement (기각: 자격증명 없는 사용자/CI에서 스크립트 막힘. no-op fallback이 안전).

---

### 2026-04-26: Phase 4 1차 sanity 실패 → 4 fix (max_tokens, stratify, parse_yes_no)

**근거**:
- `uv run python scripts/run_phase4_all.py --limit 30` 1차 실행 → **모든 method overall accuracy 0~7%**. baseline 비교 무의미.
- 분석 (`outputs/phase-4-sanity-{full,sliding,rag,hi-em}.judged.jsonl` 직접 검사):
  1. **30 questions이 모두 `temporal-reasoning`** — LongMemEval oracle이 question_type 정렬, plain `--limit 30`은 첫 type 30개. 5축 비교 자체 불가.
  2. **응답 생성 max_tokens=300** — Qwen3-8B reasoning model의 `<think>` 블록이 잘림 → strip 후 의미 없는 string.
  3. **judge max_tokens=20** — judge도 동일 model이라 `<think>Okay, let's see. The user...`에서 끝남 → yes/no 추출 실패 → 모두 False.
  4. `parse_yes_no` 단순 (첫 token이 yes로 시작?). think 안 닫혀도 robust 처리 필요.

**결정** (4 fix):
- **`run_longmemeval.py --max-tokens` default 300 → 800** — Qwen `<think>` + answer cover.
- **`judge_longmemeval.py --max-tokens` default 20 → 256** — judge thinking + yes/no fit.
- **`run_longmemeval.py --stratify` 옵션 추가** (`run_phase4_all.py`도 통과) — question_type별 균등 sample. LongMemEval oracle 정렬 문제 회피.
- **`parse_judge_yes_no` 재작성** — closed think strip → unclosed think tail 200자만 → 마지막 yes/no token 검색. 못 찾으면 False (보수). `src/hi_em/eval_logging.py`로 이동 (testable). 11 case unit test.

**영향 범위**:
- `scripts/run_longmemeval.py`: `--max-tokens 800`, `--stratify` 추가, by_type stratify 로직
- `scripts/judge_longmemeval.py`: `--max-tokens 256`, `parse_judge_yes_no` import (이동)
- `scripts/run_phase4_all.py`: `--stratify` 통과
- `src/hi_em/eval_logging.py`: `parse_judge_yes_no` 추가
- `tests/test_eval_logging.py`: parse_judge_yes_no test (10 cases)
- `handoff.md`: sanity 명령 `--stratify` 필수 명시

**검증**:
- 57/57 PASS (parse test 1개 추가)
- 사용자 재실행 후 5 type 각 6 questions × 4 method accuracy 확인 필요 (이전 0% 결과는 폐기)

**대안 및 기각 사유**:
- max_tokens 그대로 + Qwen `enable_thinking=False` (chat_template_kwargs) (보류: vLLM endpoint side support 모름. max_tokens 늘리는 게 model-agnostic. Phase 4 결과 후 token 효율 비교 가능).
- stratify를 default on (기각: 사용자가 "전체 500" 돌릴 때 stratify 의미 없음 — 이미 모두 처리. opt-in이 명시적).
- parse_yes_no를 LLM judge에 넘기는 대신 정규식만 (기각: judge prompt가 정해져 있어 LLM 응답 robust parsing 필요. 정규식 fallback이 안전).
- think 안 닫혀도 yes/no 추출 (현재 동작, 9/10 test pass 중 1개는 unclosed think 안 yes를 True로) — 이건 ambiguous edge case. max_tokens 충분히 크면 거의 발생 안 함.

---

### 2026-04-26: Apple Silicon 가속 — PyTorch MPS auto-detect (MLX 폐기)

**근거**:
- 사용자 요구: M4 Pro 등 Apple Silicon에서 GPU 가속 동작하면서, **다른 환경(CPU/CUDA)과 모델이 다르면 안 됨** — 재현성 + 협업자 환경 일관성.
- MLX (mlx-community/...): 별도 conversion → model 다름. 위 요구 위반. **폐기**.
- PyTorch MPS: 같은 `BAAI/bge-base-en-v1.5` 가중치, device 인자(cuda/mps/cpu)만 변경. 결과 numerical 차이 ~1e-5 (kernel 차이), segmentation/RAG 영향 없음. **채택**.

**결정**:
- `QueryEncoder.__init__` auto-detect 우선순위: cuda → mps → cpu (Apple Silicon에서 자동 mps).
- `scripts/run_longmemeval.py --device` 인자 추가, env `HIEM_DEVICE` fallback.
- `.env.example`에 `HIEM_DEVICE` 안내 (auto / cuda / mps / cpu).
- 토크나이저 (HF tokenizers, Rust 기반): GPU 안 씀, 변경 없음.
- 외부 LLM (vLLM endpoint): 사용자 원격 GPU. 우리 client 무관.

**검증**:
- M4 Pro 환경: `mps available=True / built=True`, auto → mps, L2 norm=1.0 유지
- 57/57 tests PASS (FakeEncoder는 별개, 영향 없음)

**영향 범위**:
- `src/hi_em/embedding.py`: auto-detect mps 추가
- `scripts/run_longmemeval.py`: `--device` + `HIEM_DEVICE` env
- `.env.example`: HIEM_DEVICE 안내

**대안 및 기각 사유**:
- MLX (mlx-embeddings) 도입 (기각: model conversion으로 다른 환경과 model 불일치 — 사용자 핵심 요구 위반).
- MPS+MLX 둘 다 지원 (기각: model 일관성 위반 + 코드 복잡도. ROI 낮음 — MPS 대비 MLX 추가 가속은 ~10-30%인데 LLM call이 압도적 시간).
- 인텔/CUDA 자동 fallback 안 함 (기각: torch가 cuda detect 자동 — 그냥 우선순위만 추가).
- bge보다 작은 model로 교체 (기각: 평가 일관성 + 옵션 5에서 bge-base 검증됨).

---

### 2026-04-27: Phase 4-Re — research-experiment-infrastructure 적용 + archive 분리

**근거**:
- Phase 4 baseline (Hi-EM 0.562 < 모든 baseline) 측정 끝. 다음 실험들 (HP sweep / 새 method / Phase 2-Full STM / 다른 dataset)이 누적될 예정. 옛 단일 prefix (`outputs/phase-4-*`) 패턴은 **lost 측정 사례 발생** (set #3 hi-em이 set #4에 덮어써짐).
- 사용자 제공 SKILL `research-experiment-infrastructure` (`~/.claude/skills/`) 적용 — atomic save / round / resume / session 표준.
- 동시에 SKILL 자체에 5 개선점 patch + §7 replay 분기 추가 (정적 dataset + stateless eval은 replay 불필요).

**결정**:
- **Archive 분리**: 기존 `outputs/` + `data/ltm/` → `archive/2026-04-26-baseline/`. README.md 4 HP × sanity/full 표 + sample noise + lost 측정 명시. 새 실험은 `results/experiments/{exp_id}/` 격리.
- **단일 entry**: `scripts/run_experiment.py`. round 단위 atomic + resume + session. legacy entry는 보존.
- **Round = 50 questions** (oracle 500 → 10 rounds).
- **2-level summary 디스크 저장**: round-level (`rounds/round_NNN/summary.json`) + experiment-level (`{exp_dir}/summary.json`, 모든 round 합쳐). 둘 다 wandb 동시 push.
- **Metric**: per-method × per-qtype accuracy (6 qtype + overall) + prefill_tokens/latency p50/p95 + error_rate + topic_revisit_hit_rate.
- **Replay 폐기**: SKILL §7 분기 — session.json에 같은 dataset 가리키는 별도 experiment로 충분.

**영향 범위**:
- `src/hi_em/atomic_io.py` + `experiment.py` (신규)
- `scripts/run_experiment.py` (신규 entry)
- `tests/test_experiment.py` (17), `tests/test_run_experiment.py` (5) — SKILL §10 #13 reference vs interrupt+resume invariant 자동화
- `archive/2026-04-26-baseline/` (영구 보존)
- `.gitignore`: results/experiments/, archive/*/{ltm,outputs/*.jsonl} 추가
- `~/.claude/skills/research-experiment-infrastructure/SKILL.md`: 5 개선점 patch + §7 분기

**대안 및 기각 사유**:
- 옛 prefix 유지 (기각: lost 측정 누적, 충돌 보장 불가).
- Replay 인프라 도입 (기각: 정적 dataset에선 가치 0).
- Round = 25 / 100 (기각: 50이 latency-resume 균형 최적).
- 단일 jsonl per experiment (기각: round 격리가 atomic 보장 쉬움).
- working_state 즉시 도입 (기각: stateless eval. Phase 2-Full STM 도입 시 추가).

**검증**:
- 100/100 unit tests PASS (78 → 100, +22)
- R-9 실 vLLM smoke (5 questions × 2 rounds): 동작 확인
- R-7 자동 invariant: deterministic FakeLLM 으로 reference vs interrupt+resume 정확 일치
- R-11 (대기): Hi-EM persistence + full 500 → archive 0.562 ±0.02 일치 검증

**Phase 5 직전 재검토 트리거**:
- 100k+ questions 동시 평가 → `results/sessions/` orchestration 강화
- 외부 storage 동기화 → `working_state` 압축 패턴 도입

---

### 2026-04-27: R-11 종결 + Phase 5 진입 결정 (정직 reframing)

**근거**:
- R-11 sanity 단계 (sanity 30 × 4 method, `run_session.py` 새 infra) 결과:
  - sliding 0.867, full 0.833, rag 0.767, **hi-em 0.700**
  - archive set #2 (옛 infra, 같은 config) 대비 모든 cell-level Δ ≤ 0.40 — sample noise (5/qtype, temperature=0.7) 영역 안
  - Overall Δ ≤ 0.10, **상대 ranking 보존** (full > sliding ≈ rag > hi-em), Hi-EM **multi-session 약점 패턴 그대로** (0.00~0.20)
  - → **인프라 systematic bias 없음 확정**. R-11 full 500 재현은 시간 절약 결정 (sanity 검증 충분).
- Phase 4 baseline (full 500 archive) 결과는 결정적: **Hi-EM 0.562 / sliding 0.658 / full 0.712 / rag 0.692**. Hi-EM 4 method 중 꼴찌. 단 **ssp 0.97 (4 method 중 1위, full 0.93보다 +0.04)** — 좁은 강점 영역 존재.

**결정**:
- **R-11 종결** (sanity 검증으로 충분, full 재실행 안 함).
- **Phase 5 진입** — 정직 reframing 우선 (5-A) → 결과 위에서 추가 실험/논문 결정 (5-B/C/D).
- Hi-EM contribution을 **광범위 winning에서 ssp 좁은 영역**으로 정의 변경 — Phase 1-6 reframing의 자연 연장.
- Phase 5-A 산출물: `report.md` 갱신 + `outputs/phase-4-final.md` (4 method × 6 qtype × 4 HP × sanity/full × 두 인프라 종합 표).

**영향 범위**:
- `plan.md` Phase 4-Re R-10/R-11 [x] 처리 + Phase 5 섹션 재작성 (5-A~5-E)
- `handoff.md` 현재 Phase = Phase 5 정직 reframing 대기
- `context/06-decision-log.md` 본 entry

**대안 및 기각 사유**:
- Full 500 한 번 더 (기각: sanity 검증 충분, archive 결과 신뢰 가능. 시간 1~2시간 절약).
- Phase 5-B (다른 dataset) 먼저 (기각: 5-A 정직 정리 없이 추가 실험은 sunk cost. reframing이 결정 전 필수).
- Phase 5 폐기 + 즉시 새 알고리즘 (기각: Phase 1~4 결과 정리 없이 새 시도는 lessons 손실).
- HP sweep 추가 (기각: 4 HP regime 이미 검증됨, 모두 multi-session 0.20 그대로).

**Phase 5-A 시작 시 트리거**:
- ssp 강점이 다른 dataset에서도 재현되는지 확인하고 싶다면 5-B 먼저.
- 단순 결과 정리만 원하면 5-A 즉시 시작.

---

## 2026-04-27 — Phase 2-Full 구현 완료 (P2F-2 ~ P2F-6)

**결정**: `phase-2-full-design.md` P2F-1~6 모두 구현 + 통합 sanity 통과. 신규 method `hi-em-full` 도입.

**구현 내용**:
- `MemoryWindow` 클래스 — **topic-atomic invariant** API로 강제 (turn-level slicing 함수 부재). `promote/maybe_append_turn/evict_lowest_importance/evict_to_capacity` + threading.RLock.
- `RoundProcessor` — per-conv 5단계 (mention log → neighbor weights → compute_importance → promote ≥ threshold → evict_to_capacity). `process_async()` daemon thread, per-instance RLock으로 라운드 직렬화.
- `HiEM(use_stm=True, round_size=10)` — STM-first 분기, cache miss 시 LTM 전체 promote, in-sync turn append (cached topic만, atomicity 유지). `next_turn_id % (2*round_size) == 0`에 round 트리거.
- 사용자 명세 두 invariant 코드로 강제: (1) topic atomicity (slicing API 부재), (2) round_size = 10 user+assistant pair = 20 jsonl rows.

**검증**:
- 단위 56 tests (P2F 모듈) + 회귀 89 tests = **145/145 PASS**.
- 통합 smoke (vLLM, 25-turn A↔B 인터리브): 모든 invariant pass.
- LongMemEval oracle stratified 30Q × `hi-em-full`: error 0/30, 78s, revisit_hit 0.40.

**다음 (사용자 실행)**: 5-method (sliding/full/rag/hi-em/hi-em-full) sanity 30 비교 → multi-session 0.20 → 0.40+ 회복 가설 검증. 가설 성립 시 full 500. 미성립 시 정직 reframing (Phase 5-A).

---

## 2026-04-29 — LoCoMo 평가 인프라 구축 + conv-level 캐시 도입 (3.3× 가속)

**결정**: LoCoMo 벤치마크용 read-only `eval_query` API 신설 + `HiEMConvCache`로 conv 단위 1회 빌드 / Q마다 read-only 평가. hi-em-full LoCoMo sanity wallclock 60분 → 18분 (3.3× 가속).

**문제 진단 (사건 발생)**:
LoCoMo는 LongMemEval과 데이터 구조가 다르다:
- LongMemEval: 500개 질문 각각이 **독립 haystack** (자기 대화 history). 매 Q마다 빌드는 자연스러움.
- LoCoMo: 10개 conv × 평균 200 Q. **모든 Q가 같은 600턴 대화를 공유**.

기존 인프라(`run_experiment.py:phase_run`)는 LongMemEval 모양에 맞춰 매 Q마다 `shutil.rmtree(ltm_root) + HiEM() + preload_history()`. LoCoMo에서는 **같은 600턴을 200번 재구축** → 200× 낭비.

Smoke 측정:
- 비-cached `hi-em-full` 49Q: round 1 = 11.6분 (latency_p50 = 339s/Q, prefill = 14.7k tok)
- 50Q sanity 단독으로 60분 추정. 7-method 합산 시 100~120분.
- Full 1986Q × 7 method 추정 65~70시간 (≈ 3일). 비현실적.

**근본 원인**: 평가 프로토콜 mismatch. Hi-EM의 `handle_turn`은 **상태를 변경**한다 (segmenter 카운트 ↑, STM miss-promote, LTM jsonl append). LoCoMo의 평가 의도는 "대화가 끝난 시점의 메모리 상태에 대해 N개 질문을 던짐" — 즉 read-only query. 비교 논문(Mem0/MemoryBank/A-Mem)도 모두 conv당 1회 build, query는 frozen state.

**대안 비교**:
| 옵션 | 시비 가능성 | 판단 |
|---|---|---|
| (V1) Per-Q 재구축 (현재) | "왜 매번 재구축, 컴퓨팅 낭비 + 비현실적" | **시비 가능** |
| (V2) Q마다 누적 (이전 Q가 메모리에 흔적) | "Q 순서 → 결과 영향, 통제 어떻게" | **시비 가능** |
| (V3) conv당 1회 build, Q마다 snapshot+read-only | **표준 프로토콜** | **선택** |

**구현 (V3)**:
1. `HiEMSegmenter.predict_topic(s)` — read-only MAP assignment. counts/topics/prev_k mutate 안 함. (`src/hi_em/sem_core.py:124-145`)
2. `HiEM.eval_query(user_text, return_debug=False)` — 메모리 mutate 없이 prefill+LLM만. STM 미스 시 LTM에서 가져오되 **STM에 promote 안 함** (in-flight prefill에만 합쳐서 사용). LTM jsonl append 없음. (`src/hi_em/orchestrator.py:218-289`)
3. `HiEMConvCache` — `(sample_id, method)` 키로 HiEM instance를 lazy build/cache. Per-key build lock으로 concurrent 안전. (`scripts/run_experiment.py:HiEMConvCache class`)
4. `phase_run`에 `hiem_cache` 파라미터, LoCoMo 모드에서 hi-em/hi-em-full/hi-em-full-v2가 자동으로 cache 경유.

**검증**:
- 단위 6 tests (`tests/test_eval_query.py`): segmenter / STM / LTM 변경 없음, 반복 호출 idempotent, stateless 모드 fallback. **6/6 PASS**.
- 전체 회귀 202/202 PASS.
- LoCoMo 49Q × hi-em-full (cached) 실측:
  - Round 1: 291.6s (cold builds 5)
  - Round 2-3: ~196s (cache hits, +2 build each)
  - Round 4: 205.4s (cache 9/10, 1 new)
  - Round 5: 127.8s (9Q, all cached)
  - **Total: 18분** (vs 비-cached 60분 추정, **3.3× 가속**)
  - F1 overall: 0.233 (adversarial 0.90, multi-hop 0.18, 나머지 < 0.05)

**영향 범위**:
- 신규: `src/hi_em/orchestrator.py:eval_query`, `src/hi_em/sem_core.py:predict_topic`, `scripts/run_experiment.py:HiEMConvCache`, `tests/test_eval_query.py`
- LongMemEval 경로 영향 없음 — `hiem_cache=None` 기본값으로 기존 per-Q 빌드 유지.

**대안 및 기각 사유**:
- snapshot+restore (deepcopy STM/segmenter, 호출 후 복원): 가능하지만 LTM jsonl append를 되돌리려면 파일 truncate 필요 → 복잡 + 디스크 I/O. **eval_query가 더 깔끔** (애초에 mutate 안 함).
- 모든 method를 conv-cache로 통일 (rag도 encoder embeddings 캐시): 미구현. sliding/full/rag는 sanity에서 빠르므로 우선순위 낮음. Full mode에서 필요하면 추후 추가.
- topic atomicity로 인한 max_turns 무력화 (단일 topic 200+ turns) 별도 이슈로 남김. 현재 경고 로그만 출력 (`MemoryWindow.promote: topic 8 has 393 turns, exceeds max_turns=200. Storing anyway`). 평가 정확도엔 영향 없으나 prefill 토큰 19k까지 부풀음.

**장기 실행 작업 점검 규칙 신설**: 위 사건 진단 중 round 2가 stuck-처럼 보였으나 실은 정상 진행이었음. 10분 초과 작업은 반드시 한 번 진행 점검 의무화. `CLAUDE.md` "장기 실행 작업 진행 점검" 절 추가.

## 2026-05-08 v3.3.2 도입 (surprise hard boundary) + v3.3.1 default HP 가 mega-topic 의 직접 원인임을 확인

세 가지.

**(1) v3.3.1 default HP 의 mega-topic 확인 + 깨는 HP 확보** — 2026-05-07 비교 sweep 에서 v3.3.1 default (`cos=0.7, α=1, β=0.5`) 가 T1μ=386 turn (~LoCoMo 600 의 절반 이상 흡수) 으로 single-topic collapse. 2026-05-08 HP sweep (5 configs) 결과: `cos=0.9, α=100, β=0.25` 에서 T1μ=28.6, T2μ=23.2, T3μ=19.7 로 **진짜 multi-topic 환경** 형성. acc 0.215 (default) → **0.263** (best HP). rag-observation baseline 0.258 와 함께 v3.3.1 이 처음으로 RAG 와 동급/우위. 결론: v3.3.1 의 알고리즘 자체는 fine, **default HP 가 잘못 잡혀 있었음**. 향후 default 를 `cos=0.9, α=100, β=0.25` 로 옮기는 것 검토 필요. 자세한 결과: `outputs/experiments/2026-05-08_v331_hpsweep/`, `context/methodology/v3.3.1.md`.

**(2) v3.3.2 도입 (surprise hard boundary)** — v3.3.1 식 + `max_k cos(s, ŝ_k) < (1 − pe_threshold)` 면 MAP 무시하고 새 topic 강제. SEM2 의 surprise → boundary mechanism 을 cosine 거리 위에 explicit threshold 로 복원. CLAUDE.md SEM 계승 원칙 정당화 3 단계 통과 (SEM 본진 *implicit* 메커니즘의 *explicit* 변환). 코드: `src/hi_em/sem_core_v332_rnn_pe.py` + 5 단위 테스트 (`tests/test_v332_pe_boundary.py`). orchestrator + run_experiment.py 분기 + `--pe-threshold` CLI. `pe_threshold=1.0` (default) 이면 v3.3.1 과 numerically 동일.

**(3) v3.3.2 PE sweep 결과** — `outputs/experiments/2026-05-08_v332_pesweep/` 12 configs. v3.3.1 best HP (cos=0.9, α=100, β=0.25) 위에 (pe_threshold, lmda, promotion_threshold, tau, train_steps) sweep. best = `pe=0.5 + rnn_train_steps=3` → **acc 0.273**. 모든 RAG/sliding baseline 보다 우위. 관찰: pe_threshold 단독은 v3.3.1 best 와 거의 동등 (~0.256~0.258). 결정적 이득은 RNN train_steps 증가. λ↓ + promotion↓ 조합은 오히려 acc 하락. 즉 mega 가 깨진 환경에선 충분한 stickiness/promotion threshold 가 multi-topic STM 에 필요. 자세한 표는 `outputs/experiments/2026-05-08_v332_pesweep/REPORT.md`, `context/methodology/v3.3.2.md`.

## 2026-05-07 v3.3.1 도입 + methodology 디렉토리 + SEM 계승 원칙 명문화

세 가지 결정.

**(1) v3.3.1 도입** — SEM 본진의 *scene dynamics + prediction error* 를 v3 라인에 복원. likelihood term 만 `τ·cos(s, μ_k)` → `τ·cos(s, ŝ_k)` 로 교체, ŝ_k 는 per-topic GRU 예측. sticky-CRP prior 식은 v3.2.1 sub-linear count 그대로 상속. SEM 계승 원칙(아래) 측면에서 *복원* 이라 정당화 부담 없음. 코드: `src/hi_em/{event_rnn,topic_v331_rnn,sem_core_v331_rnn}.py` + 5 단위 테스트. 검증: pytest 223 pass, LoCoMo `--limit 3` e2e exit 0. 자세한 알고리즘은 `context/methodology/v3.3.1.md`.

**(2) `context/methodology/` 도입** — v1/v2/v3.1.1/v3.2.1/v3.3.1 마다 한 파일씩 + `infrastructure.md` (cross-cutting). 사유: 알고리즘·hyperparameter·SEM 계승·infrastructure 결정이 decision-log + 코드 docstring 으로 흩어져 있어 "지금 v3.X 가 정확히 어떻게 동작하는가" 의 단일 진실 원천이 없었다. CLAUDE.md 에 *최우선 관리 대상* 으로 명시.

**(3) SEM 계승 원칙 명문화** — Hi-EM 은 SEM/SEM2 계승 모델이다. SEM 에 없는 메커니즘(vMF, multi-prototype, per-topic κ 학습 등)은 default = 도입 안 함. 도입 검토 시 (a) SEM 에 왜 없는지 명시, (b) SEM 철학과 충돌 안 함을 보임, (c) 본 decision-log 에 근거+날짜 append — 셋 모두 충족 필요. CLAUDE.md 의 "SEM 계승 원칙 (최상위 설계 제약)" 섹션. 사유: 새 component 를 무절제하게 추가하면 정체성을 잃고 "임의로 조합된 retrieval heuristic" 으로 전락.

**LLM endpoint 사고 부수 기록**: 같은 날 v3.3.1 비교 sweep 의 첫 시도가 `--no-thinking` 누락으로 모든 run `error_rate=1.00`. qwen/qwen3.5-9b 가 답을 `message.reasoning` 에 넣고 `content=null` 로 보내서 hypothesis 가 모두 빈 문자열. 향후 sweep script 작성 시 `--no-thinking` + `--workers 100` 둘 다 default 포함. `context/methodology/infrastructure.md` 4번 항목.

## 2026-05-03 mega-topic 해결 상태 중간 점검

v3.1 Bounded Cosine MAP와 v3.2 Cosine Prediction Error는 TopiOCQA/TIAGE에서 Gaussian v2 대비 F1을 개선하거나 유지해 likelihood 신호 회복 가능성을 보였다. 그러나 두 벤치마크는 평균 12~15턴의 짧은 대화라 raw sCRP `C_k + λ` 누적 폭주가 본격화되는 long-conv mega-topic 검증으로는 부족하다. 특히 TIAGE v3.2 best는 F1=0.411로 높지만 avg max-share=0.491, avg topics=5.1이라 짧은 대화에서도 단일 topic 흡수 경향이 나타난 경고 신호다. 결론적으로 현재 결과는 "likelihood 개선" 증거이지 "C_k 누적 mega-topic 해결" 증거가 아니며, LoCoMo 600턴 재검증과 TopiOCQA/TIAGE concat long-conv stress test로 max-share/topic-count/F1을 함께 확인한 뒤 sqrt+decay prior 도입 여부를 최종 결정한다.

## 2026-05-09 v3.3.3 설계 결정 — SEM2 `log_likelihood_f0` restart 분기 복원

**결정**: v3.3.3 은 v3.3.2 의 cosine+shared-GRU PE 구조 위에 SEM2 `log_likelihood_f0` 의 same-label restart 분기를 복원한다. $k_{\text{prev}}=k$ 후보를 repeat 와 restart 두 score 로 나누고, restart 가 repeat 를 이기면 topic label 은 유지하되 boundary 로 판정한다.

**근거**:
- SEM2 `run()` 원 구현은 직전 event 에 대해 `log_likelihood_next(x_prev, x_curr)` 와 `log_likelihood_f0(x_curr)` 를 모두 계산한다.
- 같은 event label 이라도 restart 경로가 이기면 `event_boundary=True` 가 된다. 즉 SEM2 에서 "같은 label" 과 "episode continuation" 은 분리되어 있다.
- v1 / v3.1.1 / v3.3.1 은 이 분기를 cold-start centroid fallback 으로 우회했고, v3.3.2 는 hard PE boundary 로 surprise 일부만 복원했다. restart-vs-repeat 자체는 여전히 미포팅 상태다.

**SEM 계승 정당화**:
- SEM 에 없는 메커니즘이 아니라 SEM2 에 있던 인터페이스의 복원이다.
- Hi-EM 에 없었던 이유는 원리상 기각이 아니라 구현 단순화와 cosine surrogate 전환 과정의 미포팅이다.
- sticky-CRP prior, local MAP, online update, scene dynamics 와 충돌하지 않는다. $k_{\text{prev}}$ 의 likelihood 경로만 SEM2 처럼 분해한다.

**영향 범위**:
- 설계 문서: `context/methodology/v3.3.3.md`
- 구현 (2026-05-09 완료): `src/hi_em/sem_core_v333_rnn_f0.py`, `src/hi_em/orchestrator.py` (v3.3.3 분기), `scripts/run_experiment.py` (method mapping + HP CLI flags). v3.3.3 구현 outline 은 구현 완료 후 삭제 (`v3.3.3_impl_outline.md`).

**대안**:
- v3.3.2 hard PE boundary 만 유지 (기각: 새 topic 강제와 same-label restart 는 다른 현상. SEM2 원 분기를 계속 빠뜨림)
- restart 를 항상 새 topic 으로 처리 (기각: label 재사용을 허용하는 SEM2 구조와 불일치)
- f0 를 전역 상수로만 처리 (보류: cold-start fallback 으로는 가능하지만 SEM2 event별 f0 의미가 약해짐)

## 2026-05-09 v3.3.4 설계 결정 — per-topic $\sigma_k^2$ EMA variance calibration 도입 검토

**결정**: v3.3.4 는 v3.3.2 의 global hard PE threshold 한계를 검증하기 위해 topic별 cosine PE 분산 $\sigma_k^2$ 를 EMA 로 유지하고, likelihood 를 $-\mathrm{PE}^2/(2\sigma_k^2)$ 로 calibration 하는 실험 branch 로 설계한다. default 승격이 아니라 정당화 부담이 큰 후보로 취급한다.

**근거**:
- v3.3.2 는 `cos_threshold` 와 `pe_threshold` 가 전역 고정값이다. 좁은 topic 과 넓은 topic 의 cosine PE 분포가 다르면 같은 threshold 가 과분할 / 과흡수를 동시에 만들 수 있다.
- LoCoMo 는 factual thread, preference/social thread, long-range revisit 가 섞여 있어 topic별 PE 폭이 다를 가능성이 있다.
- SEM 의 본래 likelihood 는 prediction error 를 noise scale 과 함께 해석한다. cosine substitution 에서 사라진 calibration 을 최소 scalar variance 로 되돌리는 실험이다.

**SEM 계승 정당화**:
- SEM2 에 topic별 $\sigma_k^2$ EMA 는 없다. 가능한 이유는 controlled video/VAE 환경에서 공통 noise 로 충분했거나, 작은 event 의 variance 학습이 불안정해 의도적으로 단순화했기 때문이다.
- 따라서 v3.3.4 는 per-topic concentration $\kappa$ 학습과 같은 계열의 위험한 자유도다. min-sample guard, global fallback, floor/cap 을 필수로 두고, 성능 이득과 PE histogram 근거가 없으면 default 로 올리지 않는다.
- sCRP/Bayes 와 구조적으로는 충돌하지 않는다. sCRP 는 prior, $\sigma_k^2$ 는 likelihood noise scale 이다. 단 SEM2 원형보다 자유도가 늘어나는 절충을 명시한다.

**영향 범위**:
- 설계 문서: `context/methodology/v3.3.4.md`
- 구현 (2026-05-09 완료): `src/hi_em/sem_core_v334_rnn_var.py`, `src/hi_em/orchestrator.py` (v3.3.4 분기), `scripts/run_experiment.py` (method mapping + variance HP CLI flags). 구현 outline / sweep plan 은 구현 완료 후 삭제 (`v3.3.4_impl_outline.md`, `v3.3.3_v3.3.4_sweep_plan.md`).

**대안**:
- global `pe_threshold` sweep 지속 (기각: topic별 PE 폭 문제를 해결하지 못함)
- per-topic $\kappa$ 또는 vMF concentration 학습 (기각: SEM 계승 부담이 더 크고 cosine likelihood 구조를 더 많이 바꿈)
- per-conversation variance 만 사용 (보류: SEM 계승 부담은 낮지만 좁은/넓은 topic 차이를 직접 다루지 못함)

## 2026-05-09 v3.3.3 / v3.3.4 first full LoCoMo run + retrieval-metric 발견

**결정**: codex 가 추천한 default HP 로 v3.3.3 / v3.3.4 를 full LoCoMo 1986Q 1 회 평가. 결과 보고 + 다음 sweep 방향 확정.

**관찰 (2026-05-09_v333_v334_full_6method_par REPORT)**:

| method | acc | H@k | R@k | R-mh@k | P@k |
|---|---:|---:|---:|---:|---:|
| rag | 0.285 | 0.640 | 0.569 | 0.365 | 0.0732 |
| rag-observation | **0.296** | (재실행 중) | (재실행 중) | (재실행 중) | (재실행 중) |
| rag-summary | 0.273 | (재실행 중) | (재실행 중) | (재실행 중) | (재실행 중) |
| v3.3.2 | 0.258 | 0.421 | 0.355 | 0.251 | 0.0033 |
| v3.3.3 | 0.256 | 0.412 | 0.347 | 0.232 | 0.0033 |
| v3.3.4 | 0.259 | **0.445** | **0.386** | **0.303** | 0.0029 |

**핵심 발견**:
1. **v3.3.3 의 acc 효과 0**: same-label restart 분기는 default HP (`restart_pe_threshold=0.5`) 에서 v3.3.2 와 동일 segmentation (n_topics 10.00, T1μ 25.5 모두 같음). restart 분기가 거의 발화 안 함 — threshold 너무 보수적이거나 cosine surrogate 에서 f0 score 가 repeat 를 거의 못 이김.
2. **v3.3.4 가 retrieval 단에서 v3.3.2 대비 명확히 개선**: H@k +5.7%, R@k +8.7%, **R-multi-hop@k +20.7%**. 즉 calibrated likelihood 가 segmentation 을 더 의미있게 만든다. 그런데 acc 는 동일 — retrieval 개선이 LLM 답변으로 전달 안 됨.
3. **hi-em 의 P@k 가 RAG 보다 20× 낮음** (0.003 vs 0.073): topic 단위 prefill 의 본질적 noise. 정답 turn 이 retrieved set 안에 있어도 noise 가 너무 많아 LLM 이 답을 못 잡는 가설.
4. **RAG retrieval 우위 (H@k 0.64 vs hi-em 0.42)**: hi-em 의 STM/LTM 가 정답 turn 을 못 가져오는 편. acc gap 0.04 의 절반 이상은 retrieval 단 차이.

**버그 수정 (이번 세션)**: `scripts/run_experiment.py:1334` 의 `hiem_cache` 활성화 set 에 v3.3.3 / v3.3.4 누락 → 매 질문마다 600-turn 대화 재 preload (200x 낭비) → 제거 후 30x 가속. 자동화: `is_hi_em_method()` regex helper 로 변경, prefix 매칭하므로 새 버전 추가 시 set 갱신 불필요.

**REPORT 컬럼 보강**: H@k / R@k / R-multi-hop@k / P@k 컬럼이 summary.json 에는 있는데 `write_report` 가 출력 안 했던 것 수정. rag-summary / rag-observation 는 retrieved_dia_ids emit 안 하던 것도 수정 + 재실행.

**다음 단계**:
- v3.3.4 multi-hop 만 따로 acc breakdown 보기 (R-mh@k 가 +20% 인데 acc 영향 0 이 진짜인지 검증)
- v3.3.3 의 restart 분기가 발화하는 HP 영역 sweep (`restart_pe_threshold ∈ {0.2, 0.3, 0.4}`, `restart_margin ∈ {0, -2, -5}` 음수도 시도)
- v3.3.4 HP sweep (`pe_var_sigma0_sq ∈ {0.01, 0.04, 0.1}`, `pe_var_min_samples ∈ {3, 5, 10}`)
- hi-em P@k bottleneck 검증: `k_turns_per_topic` 줄여 prefill 압축 → acc 효과 보기

**영향 범위**: REPORT.md 컬럼 schema, decision-log, handoff, methodology 갱신 필요.

---

## 2026-05-10 — v3.3.3-2 / v3.3.4-2 강화안 1차 sanity sweep + iter2 재설계 결정

**컨텍스트**: 사용자 요구 — v3.3.3 / v3.3.4 강화안 (md 의존 X, 깊이 재설계) 도입. 사용자 자율 모드 ("엔터 안 누름"). codex 권장 (`019e10f0-...`):
- v3.3.3-2: boundary-start prototype top-M f0 + posterior-odds restart (`p_rst > 0.35`)
- v3.3.4-2: per-topic σ² Bayesian shrinkage (`σ_eff² = (n/(n+c))σ_k² + (c/(n+c))σ_0²`, default c=8) + optional robust MAD-EMA scale

**구현**: `src/hi_em/sem_core_v333_2.py`, `sem_core_v334_2.py` 추가. orchestrator dispatch + `scripts/run_experiment.py` method 등록 + CLI flag + cache `_build` 매핑.

**버그 발견 + 수정**: `_HI_EM_METHOD_RE = r"^hi-em(-full-v[\d.]+)?$"` regex 가 dash(`-`) 미허용 → `hi-em-full-v3.3.3-2` 매치 실패 → `needs_encoder=False` → encoder None → 첫 sanity 6 runs 가 18 초로 즉시 fail (`'NoneType' object has no attribute 'dim'`). Fix: `[\w.\-]+`.

**iter1 sanity 결과** (`outputs/experiments/2026-05-10_v33x_2_iter1/`, LoCoMo --limit 200 stratified, 196 Q × 3 run/method):

| method | acc | acc_mh | H@k | R@k | R-mh@k | T1μ | T1var | n_topics |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rag-observation | 0.267 | 0.135 | 0.647 | 0.524 | 0.422 | — | — | 0 |
| v3.3.3 | 0.254 ± 0.006 | 0.137 | 0.310 | 0.226 | **0.201** | 25.7 | 47.5 | 9.88 |
| **v3.3.3-2** | 0.255 ± 0.004 | 0.134 | **0.323** | **0.234** | **0.185** | 26.3 | **72.0** | 9.81 |
| v3.3.4 | 0.245 ± 0.001 | 0.136 | 0.255 | 0.165 | 0.189 | 47.5 | 18.8 | 4.81 |
| **v3.3.4-2** | **0.253 ± 0.001** | **0.153** | 0.254 | 0.157 | 0.194 | 61.3 | 27.5 | **3.24** |

**판정**:
- **v3.3.3-2 → 재구현 (iter2 v3.3.3-3)**. acc 향상 없음 (+0.001), R-mh@k 회귀 (-0.016), T1var 폭증 (47→72). 가설: posterior threshold 0.35 너무 낮아 restart 과다 발화 → episode 분산 → multi-hop strict 회수 손실.
- **v3.3.4-2 → 유지**. acc +0.008 (std 0.001 대비 8σ), acc_mh +0.017. shrinkage 가 fewer/larger topic 으로 안정화. retrieval 거의 유지. → full LoCoMo 1986Q × 3 runs (`outputs/experiments/2026-05-10_v3342_full/`) 진행 중.

**iter2 v3.3.3-3 재설계** (codex thread 이어서, `agentId: a4f3e7b1468a14b3f`):
1. restart 보수화: `restart_p_threshold=0.5` + `restart_prob_margin=0.15` + `episode_min_span=4` (hysteresis).
2. prototype softening: top-M `max` → `(1-γ)cos(s,mean) + γ·logmeanexp(cos(s,p))`, `γ=0.25`.
3. **episode atomicity 를 retrieval layer 에서 직접 구현** (iter1 미구현 핵심): segmenter 가 `(topic, episode)` 발급 + retrieval `R(i) = a·cos(q,s_i) + b·S_topic + e·S_episode - g·zPE - r·recency` 3단 rerank.

SEM 계승 정당화: prototype f0 / shrinkage 둘 다 SEM2 에 직접 없음. 정당화 (1) SEM2 controlled VAE 자극에서는 추가 자유도 불필요, (2) sCRP / Bayes / local MAP / scene dynamics 와 직교, (3) 본 entry append. (자세히는 `context/methodology/v3.3.3-2.md`, `v3.3.4-2.md`.)

**TIAGE (병행 결과)** (`outputs/experiments/2026-05-10_tiage_iter1/REPORT.md`): default LoCoMo HP 가 TIAGE 짧은 dialog 에 부적합 → v3.3.x 6 method 모두 동일 corner case (R=1.0, n_topics=15.6 = every-transition prediction). method 차이 분간 불가. 다음 iteration 에서 TIAGE-specific HP grid sweep 필요.

**영향 범위**: methodology v3.3.3-2.md / v3.3.4-2.md 신규, infrastructure.md (regex bug 항목 추가 예정), README / handoff / architecture (cascade 검사 대상).

---

## 2026-05-10 — v3.3.3-2 / v3.3.3-3 폐기 결정 + iter3 v3.3.3-4 (retrieval atomicity)

**결정**: v3.3.3-2 (iter1) 와 v3.3.3-3 (iter2) **둘 다 폐기**. iter3 v3.3.3-4 가 대체.

**근거**:

| method | iter | acc | R-mh@k | T1var | 판정 |
|---|---|---:|---:|---:|---|
| v3.3.3 baseline | — | 0.254 ± 0.005 | 0.213 | 44.2 | reference |
| v3.3.3-2 | 1 | 0.255 ± 0.001 | 0.185 (-0.016) | 60.0 (+12.5) | 회귀 (R-mh + T1var) |
| v3.3.3-3 | 2 | 0.247 ± 0.007 (-0.007) | 0.193 (-0.020) | 51.4 (+7.2) | acc 추가 회귀, R-mh 부분 회복만 |

**핵심 검증 결과**: **segmentation 만 변경하는 방향의 한계 확인**. iter1 codex 권장 (`019e10f0-...`) 의 핵심이 "(topic, episode) 단위 retrieval atomicity" 였는데 segmentation 변경만 적용 → prefill 에 효과 안 옴. v3.3.3-2 prototype f0, v3.3.3-3 hysteresis + softening 모두 cosmetic (acc 안 옴).

**보존 정책 (CLAUDE.md "archive 는 의도적 폐기")**:
- 코드 (`src/hi_em/sem_core_v333_2.py`, `sem_core_v333_3.py`, orchestrator dispatch, run_experiment.py method choices) **그대로 유지** — 재현성 + 향후 ablation 비교 + iter3 codex 가 v3.3.3-3 hysteresis 부분 reference 한다고 명시.
- methodology 파일 (`v3.3.3-2.md`, `v3.3.3-3.md`) 에 **DEPRECATED 헤더 + 후속 링크** 추가.
- archive 디렉토리 이동 안 함 — 단일 파일 단위라 archive/ 의 디렉토리 폐기 정책에 맞지 않음.

**iter3 v3.3.3-4 설계** (codex thread `019e10f0-...` 이어서, agent `a2b9ffe8f8011b9af`):

1. **Segmenter minimal**: v3.3.3 baseline + 보수화 (`restart_p_threshold=0.5`, `restart_prob_margin=0.15`, `episode_min_span=8`, `f0_proto_weight=0`, `f0_proto_max=1`). v3.3.3-3 의 hysteresis 만 살리고 prototype 사실상 비활성.
2. **(topic_id, episode_id) atomic retrieval**:
   - `assign()` 이 `(topic_id, episode_id)` emit
   - STM/LTM 에 `episode_id` 저장
   - `eval_query` 에서 episode-level rerank: `R(i) = a·cos(q,s_i) + b·S_topic + e·S_episode - g·zPE - r·recency` (a=1.0, b=0.10, e=0.35, g=0.05, r=0.03)
3. **`promotion_threshold = 0.3`** (default 0.5 → 0.3): STM 진입 임계 낮춰 evidence-bearing topic 살림.
4. **Dormant LTM safety net** (`dormant_ltm_top_n=8`): STM 못 들어간 topic 중 query cosine top-N 추가 → promotion miss 회피.

**SEM 계승**: episode = SEM2 `new_token()` 직접 대응. retrieval rerank / dormant LTM safety net 는 SEM 외 (QA readout 정책) 이지만 비학습 + 비파괴적 추가.

**병행 도구**: dormant evidence audit (`compute_dormant_evidence_audit`) — 모든 hi-em-full method 에 적용. STM 못 들어간 topic 중 evidence 몰린 비율 측정. 사용자 가설 ("importance score 정책 자체에 문제") 직접 검증.

**다음 단계**: (1) audit 도구 구현, (2) v3.3.3-4 segmenter + retrieval rerank + dormant safety net 구현, (3) iter3 sanity (v3.3.3 baseline / v3.3.3-4 / v3.3.4-2 비교).

---

## 2026-05-11 v3.3.3-4 구현 + iter1 sanity 시작

**구현 완료**:
- `src/hi_em/sem_core_v333_4.py` — `assign()` 이 `(topic_id, is_boundary, episode_id)` 3-tuple 반환. v3.3.3 baseline + 보수화 default (`restart_p_threshold=0.5, restart_prob_margin=0.15, episode_min_span=8, f0_proto_max=1, f0_proto_weight=0.0`). 6 종 diagnostic counter (assigns / label_change / same_label_restart / repeat_wins / restart_wins / hysteresis_blocked) 인스턴스에 보존.
- `src/hi_em/orchestrator.py`:
  - `_assign_with_episode()` wrapper — 모든 segmenter 의 2-tuple/3-tuple return 을 통일.
  - `_episode_rerank_prefill(q)` — `(topic_id, episode_id)` 그룹 score → top-K + dormant LTM safety net.
  - v3.3.3-4 dispatch 시 자동 set: `retrieval_mode="episode_rerank"`, `dormant_ltm_top_n=8`, `promotion_threshold=0.3`.
  - 모든 turn dict 에 `episode_id` 필드 추가 (`_make_turn`, preload, eval_query path).
- `scripts/run_experiment.py`:
  - `_HI_EM_METHOD_RE` regex 수정 (`[\w.\-]+`) — `hi-em-full-v3.3.3-4` 같이 dash 포함된 버전이 `is_hi_em_method()` true 가 되도록. (이전엔 `[\d.]+` 만 허용해 dash 가 빠지면 encoder None → `'NoneType' object has no attribute dim'` 에러로 18 초 만에 망가짐.)
  - method choices / `HiEMConvCache._build` HP map / dispatch 세 곳에 v3.3.3-4 추가.
  - `compute_dormant_evidence_audit(entry, hi)` 추가 — 모든 hi-em-full method 에 자동 적용 (LoCoMo evidence 있을 때만 emit).
  - 새 CLI: `--episode-top-k`, `--dormant-ltm-top-n`, `--rerank-{query,topic,episode}-weight`, `--rerank-pe-penalty`, `--rerank-recency-weight`, `--restart-p-threshold`, `--restart-prob-margin`, `--episode-min-span` 등.
- `src/hi_em/eval_logging.py` `aggregate_summary` — `dormant_ev_rate`, `n_topics_with_ev`, `top_ev_topic_promoted` 평균 추가.
- `scripts/experiment.py` REPORT.md 컬럼 — `dormant_ev_rate`, `top_ev_promoted` 추가.
- `context/methodology/v3.3.3-4.md` 신규 + README/infrastructure cascade.

**Smoke (5Q LoCoMo)**: `acc=0.059`, dispatch 정상 — segmenter / retrieval rerank / dormant safety net / audit 모두 호출됨 검증.

**iter1 sanity 시작 (`outputs/experiments/2026-05-11_v3334_iter1/`)**: LoCoMo limit=200 stratified, workers=25, no-thinking. 6 runs:
- v3.3.3 baseline ×3 (seed 42/43/44)
- v3.3.3-4 ×3 (seed 42/43/44)

검증 목표 (codex 가이드):
1. R-mh@k 회복 + 향상 (v3.3.3 0.213 ≤ v3.3.3-4)
2. acc +0.005 이상 (1σ 이상)
3. dormant_ev_rate baseline 대비 의미있는 감소
4. T1var 안정 (v3.3.3 47.5 ± 10.7 수준 유지)


---

## 2026-05-11 v3.3.3-4 iter1 sanity 결과 + 판정 (codex 위임)

**3-seed 평균 (LoCoMo 200Q stratified)**:

| 지표 | v3.3.3 (×3) | v3.3.3-4 (×3) | Δ |
|---|---:|---:|---:|
| acc | 0.2547 ± 0.0010 | 0.2551 ± 0.0020 | +0.0004 (noise) |
| H@k | 0.3226 ± 0.0109 | 0.6389 ± 0.0080 | +0.316 (29σ) |
| R-mh@k | 0.2000 ± 0.0186 | 0.4112 ± 0.0100 | +0.211 (11σ) |
| P@k | 0.0030 ± 0.0000 | 0.0260 ± 0.0007 | +0.023 (~30×) |
| dormant_ev_rate | 0.9397 ± 0.0096 | 0.9555 ± 0.0074 | +0.016 |
| top_ev_topic_promoted | 0.0534 ± 0.0160 | 0.0513 ± 0.0138 | -0.002 (noise) |

**판정 (codex thread `ab18160235a3fc644`)**: **method 채택 X — importance policy 재설계 우선**.

근거:
- retrieval 단 폭증 (H@k 2 배, P@k 30 배) 인데 acc 는 noise 안. retrieval 향상이 LLM 답으로 안 옮음 (prefill dilution / LLM 한계 / multi-hop reasoning gap 후보).
- promotion_threshold 0.5 → 0.3 완화에도 `top_ev_topic_promoted` 가 회복 안 됨 → threshold miscalibration 아님, 정책 자체가 query-time evidence relevance 신호를 안 갖고 있음.
- `topic_importance.py::compute_importance()` 사용 신호: count / mention EMA / recency / neighbor prior. **query relevance, episode evidence density, answer-time need 부재.**

**다음 단계**:
1. **`_episode_rerank_prefill()` 코드 vs 문서 weighted formula 불일치 수정** (먼저 작은 fix).
2. **LLM-side ablation**: `episode_top_k ∈ {1,2,3}` × `dormant_ltm_top_n ∈ {4,8}` — prefill dilution 가설 검증.
3. **보조 sweep**: `stm_max_topics` 확장 + `promotion_threshold` 0.3/0.2/0.1/0.0 sanity.
4. **importance policy 재설계 (codex 위임 예정)**: SEM 정신 안 깨고 query-time relevance / episode salience 통합. SEM 외 메커니즘이지만 segmentation posterior 변경 없는 non-destructive readout 이면 정당화 가능.

**SEM 정당화 후보 (재설계 시)**:
- [SEM 있음] episode-level scoring (이미 구현, 문서 vs 코드 불일치 수정).
- [SEM 없음, 도입 가능] topic-level evidence aggregation (비지도 readout, segmentation 영향 X).
- [SEM 없음, 도입 가능] query-time importance recompute (LTM topic backwrite X).


---

## 2026-05-11 audit metric 보강 (집중도 직접 측정)

**변경**: `compute_dormant_evidence_audit` 에 3 개 metric 추가:
- `n_dormant_topics_with_ev` — dormant 안 evidence-bearing topic 수
- `dormant_top_topic_n_ev` — top dormant topic 안 evidence 절대 수
- `dormant_top_topic_share` — top dormant topic 비중 (= top_n_ev / dormant_total). **1.0 → 한 dormant topic 에 다 몰림**

**왜**:
사용자 지적: 기존 audit (`dormant_ev_rate`, `top_ev_topic_promoted`, `n_topics_with_ev`) 는 "STM 못 들어간 비율" 만 측정. 원래 가설 ("dormant topic 들 중 evidence 가 *몰린* topic 이 있는가") 의 *집중도* 는 직접 안 봄. iter1 결과 `dormant_ev_rate=0.94, n_topics_with_ev=1.87` 에서 집중도가 *간접적으로* 높을 거라 추정만 가능. 이번 보강으로 직접 측정.

**해석 가이드**:
- `dormant_top_topic_share ≥ 0.7` → 한 dormant topic 에 evidence 집중. promotion policy 만 고치면 회수 가능 (importance redesign 효율 높음).
- `dormant_top_topic_share ≤ 0.4` → dormant 안에서도 흩어짐. segmentation 단계 cause 가능성 (evidence-bearing turn 들이 잘못 분리됨).
- 0.4~0.7 → 중간. 둘 다 필요.

**v3.3.4 audit 1986Q full run 재시작** (`outputs/experiments/2026-05-11_v334_audit_full/`): 보강된 metric 으로 측정. ~수시간 소요.

**cascade**: `eval_logging.py aggregate_summary` + `experiment.py` REPORT 컬럼 + `infrastructure.md § 7b` 모두 업데이트.


---

## 2026-05-11 v3.3.4 audit full LoCoMo (1986Q) 완료 — dormant 집중도 직접 검증

**`outputs/experiments/2026-05-11_v334_audit_full/`** (1986Q × 1 run, audit 보강 metric 적용).

| 지표 | 값 |
|---|---:|
| acc | 0.258 |
| H@k | 0.430 |
| R-mh@k | 0.300 |
| P@k | 0.003 |
| **dormant_ev_rate** | **0.701** |
| **top_ev_topic_promoted** | **0.307** |
| n_topics_with_ev | 1.44 |
| **n_dormant_topics_with_ev** | **1.06** |
| **dormant_top_topic_n_ev** | **0.79** |
| **dormant_top_topic_share** | **0.641** |

**핵심 결과**:

1. **`dormant_top_topic_share = 0.64`** — dormant 안에서 evidence 가 한 topic 에 64% 집중. 사용자 가설 ("한 dormant topic 에 evidence 가 몰려 있다") **부분 컨펌** (완전 100% 아님 — 36% 는 흩어짐).
2. **n_dormant_topics_with_ev = 1.06** — dormant 안 evidence 보유 topic 수 평균 1 개 → 대부분의 conv 에서 dormant evidence 는 단일 topic 에 모임.
3. **top_ev_topic_promoted = 0.31** — full 에서는 정답 top topic 의 STM 진입 비율 31%. 200Q sanity (0.05) 보다 6 배 높음. full data 가 evidence-rich conv 비중이 커 promotion 확률도 높음.

**해석**:
- ROI 명확: promotion policy 만 고쳐도 evidence 의 ~64% 회수 가능 (한 dormant topic 만 STM 올림).
- 나머지 36% 는 여러 dormant topics 에 흩어져 있음 → segmentation level 또는 multi-evidence-topic 검색 정책 필요.
- 이 결과는 codex iter1 판정 ("**importance policy 재설계 우선**") 의 직접 증거.

**다음 단계 (확정)**:
1. `_episode_rerank_prefill()` 코드 vs 문서 weighted formula 불일치 수정.
2. `episode_top_k × dormant_ltm_top_n` ablation (acc 정체 dilution 가설 검증).
3. **importance policy 재설계** (codex 새 thread 위임): query-time evidence relevance / episode salience aggregation / topic-level salience — SEM 외 메커니즘이지만 segmentation posterior backwrite 없는 readout 으로 정당화.
4. 보조: `stm_max_topics` cap 확장 sanity sweep.


---

## 2026-05-11 Retrieval = importance only (CLAUDE.md 최상위 설계 제약 추가)

**결정**: Hi-EM query-time retrieval 에서 query embedding cosine matching (RAG-style) 영구 폐기. **importance score + topic_id atomicity 만으로 retrieval 결정.**

**근거**:
- SEM 정신상 "지금 working memory 에 무엇이 있어야 하는가" 는 importance score 가 결정. 이미 결정된 STM 위에서 query-time cosine 으로 *다시* ranking → 두 메커니즘 경쟁 → 정합성 깨짐.
- v3.3.3-4 의 hybrid (importance + cosine) 가 retrieval 지표는 폭증시켰지만 acc 정체. cosine 우회로 importance policy 의 부실 (dormant_ev_rate 0.70~0.94) 이 가려짐.
- 근본 해결: importance policy 자체 강화. cosine 우회 X.

**구체적 폐기 대상**:
- `_episode_rerank_prefill()` 의 `max cos(q, embedding)` 기반 episode scoring → topic importance + episode 내부 salience 로 교체.
- `dormant_ltm_top_n` cosine top-N fallback → topic_id 기반 explicit fetch 또는 promotion threshold 인하로 교체.
- 모든 v3.x 계열 retrieval 정책 재검토.

**CLAUDE.md 추가 위치**: SEM 계승 원칙 § 바로 뒤. 신규 § "Retrieval 은 importance score 만으로 한다 (최상위 설계 제약, 2026-05-11)".

**다음 단계**:
1. retrieval 재설계 (codex 위임): importance-only retrieval 의 구체적 알고리즘 + episode-level salience 계산 + LTM fetch 정책 + acc 회복 가능성 평가.
2. 그 위에 importance policy 자체 보강 (어차피 codex iter1 판정의 재설계 우선 항목).

**SEM 계승 정당화 (이 규칙 자체)**:
- SEM/SEM2 의 retrieval 은 working memory (현재 segmented + active topic) 자체를 readout. query-aware re-ranking 없음. 본 규칙은 그 정신을 명시화한 것.
- 단 segmentation level 에서 의미 매칭 (cos(s, μ_k)) 은 SEM2 likelihood 로 정당화됨 (계속 사용). retrieval 단의 cosine 만 폐기.

이 결정은 매우 중요해서 CLAUDE.md 최상위 규칙으로 박음. 향후 어떤 retrieval 메커니즘 제안도 본 규칙 통과해야 함.


---

## 2026-05-11 v3.3.3-4 importance-only retrieval 구현 완료 (codex thread `aad9876319aea12cf`)

**구현**:
- `_episode_rerank_prefill(q)` → `_importance_prefill()` 로 rename + 단순화
- 함수 본체: `stm.all_turns()` sorted by turn_id. **그 외 어떤 query-time scoring 없음.**
- query embedding `q` 인자 제거 (사용 안 함).
- `dormant_ltm_top_n` cosine top-N fallback 완전 폐기.
- `episode_top_k`, `rerank_*_weight` HP 들 노이즈 — 무시됨 (코드에서 안 쓰임).
- v3.3.3-4 dispatch default: `retrieval_mode="importance_only"` (이전 "episode_rerank"), `promotion_threshold=0.3` 유지.
- `RoundProcessor.latest_importance` property 추가 — diagnostic 용 (직접 retrieval 에는 사용 안 함).

**SEM 정당화** (codex 정리):
- SEM/SEM2 의 retrieval 은 working memory (현재 segmented + active topic) 자체를 readout. query-aware re-ranking 없음. 본 변경은 그 정신을 명시화.
- Segmentation level 의 의미 매칭 (cos(s, μ_k)) 은 SEM2 likelihood 로 정당화됨 — 계속 사용. retrieval 단의 cosine 만 폐기.
- importance-only retrieval 자체는 SEM 정신 완전 일치 (cosine RAG hybrid 보다 오히려 정통).

**예상 acc 영향** (codex 분석):
- 제공 데이터 (200Q sanity / 1986Q full) 모두 cosine hybrid 로 H@k 두 배 올라도 acc 동일 → "evidence 더 넣으면 acc 오른다" 가설 미지지.
- importance policy 강화는 retrieval recall (H@k, R-mh@k) 개선에 효과 가능, acc 는 별도 병목 (multi-hop reasoning, P@k=0.003 즉 noise turn 다수, prompt dilution).
- 따라서 cosine 폐기 후 retrieval 지표는 떨어질 수 있지만, acc 손실은 제한적일 가능성. **acc 회복 레버는 cosine 이 아니라 importance policy 강화.**

**다음 단계**:
1. **검증 실험** — v3.3.3-4 importance-only 200Q sanity (3 seeds) vs v3.3.3 baseline. retrieval 지표 회귀 + acc 영향 측정.
2. **Importance policy 강화** (별도 codex thread): unsupervised internal density signals (revisit count, boundary frequency, PE salience, neighbor centrality, persistence) 를 `compute_importance()` 에 추가. SEM 외 메커니즘이지만 SEM 의 PE/boundary/scene dynamics 철학과 정합. 3-step 정당화 적용.
3. v3.3.3-4 methodology 파일 업데이트 (이미 완료).

**Codex 권장 우선순위**: minimal retrieval 변경 (현재 완료) → 200Q sanity 검증 → importance policy 강화 (메인 acc 회복 레버).

**Smoke test 결과** (5Q + 20Q smoke run): pipeline 정상 동작 (error_rate=0). 작은 sample 에서 retrieval 지표 baseline 수준 (H@k=0.05 — 20Q noise) — 이건 데이터 양 문제, 200Q+ 에서 본격 검증 필요.


---

## 2026-05-11 Importance policy v2 도입 (codex thread `a7b7046c387f8a434`)

**진단** (200Q iter2 importance-only sanity 결과 기반):
- promotion_threshold 0.5 → 0.3 완화에도 `top_ev_topic_promoted` 가 0.062 → 0.053 (오히려 미세 감소).
- 즉 threshold 인하만으로는 dormant evidence topic 의 *rank* 가 안 올라감.
- `compute_importance()` v1 (count + EMA + recency + neighbor) 이 evidence-rich-but-quiet topic 을 구조적으로 낮게 평가:
  - recency/EMA 편향 (조용한 topic 매번 0 근처)
  - count 항이 큰 topic peak anchor 됨 (normalize=1.0 정규화 효과)
  - query/evidence relevance 무지 (cosine 폐기 후 SEM 내부 signal 필요)

**처방 — compute_importance v2 (additive 통합)**:

$$raw_I(t) = w_1 s_{count} + w_2 s_{freq} + w_3 s_{recency} + w_4 s_{nbr} + w_5 s_{PE} + w_6 s_{boundary} + w_7 s_{span}$$

새 항 (모두 SEM-internal, unsupervised, bounded [0,1]):
- $s_{PE}$ = `clip((0.7·μ_PE + 0.3·max_PE) / PE_ref, 0, 1)` — SEM2 prediction error 직접 연결.
- $s_{boundary}$ = `1 − exp(−B_t / B_ref)` — SEM2 event boundary count.
- $s_{span}$ = `1 − exp(−(last − first + 1) / S_ref)` — persistence (scene dynamics 장기 흔적).

기존 항도 보강:
- $s_{count}$ = `log1p(min(n, n_cap)) / log1p(n_cap)` — count cap (default 30) 으로 대형 topic anchor 약화.
- 다른 3 항은 기존 v1 유지.

**default weights**: `(w1, w2, w3, w4, w5, w6, w7) = (0.70, 0.60, 0.45, 0.35, 0.90, 0.70, 0.25)`
**default refs**: `n_cap=30, pe_ref=1.0, boundary_ref=3.0, span_ref=12.0, min_floor=0.10`

**SEM 정당화** (CLAUDE.md 3-step):
- PE salience: SEM/SEM2 의 prediction error 와 직접 연결. promotion 용 적분이 원 SEM 구성요소는 아니지만 (SEM 에 없음, decision-log 기록), supervised label 안 쓰고 unsupervised event surprise 신호. SEM event boundary 철학과 충돌 없음.
- Boundary frequency: v3.3.3-4 가 이미 `episode_id` emit. boundary 풍부 = scene structure 풍부. SEM event dynamics 정신과 정합.
- Persistence span: scene dynamics 의 장기 흔적 요약. recency 와 별개로 "오래 유지된 latent scene" 의 안정성. 낮은 weight 보조항.

**구현**:
- `src/hi_em/topic_importance.py::compute_importance_v2()` 추가 (기존 함수와 backward-compat).
- `src/hi_em/sem_core_v333_4.py` — 각 assign() 마다 PE running mean/max 추적 + `pe_stats()` / `boundary_counts()` accessor 노출.
- `src/hi_em/round_processor.py` — `importance_version`, `salience_provider` (segmenter ref) 인자 추가.
- `src/hi_em/orchestrator.py` — v3.3.3-4 dispatch 가 `importance_version="v2"` 자동 default + 7-tuple weights.
- `scripts/run_experiment.py` — `--importance-version`, `--importance-alpha nargs='+'` 추가.

**검증 실험**: `2026-05-11_imp_v2_sanity` — 200Q × 6 runs (v3.3.3 baseline ×3 vs v3.3.3-4 importance v2 ×3).

**검증 목표**:
- `top_ev_topic_promoted`: 0.053 → **0.12+** (메인 성공 조건)
- `dormant_ev_rate`: 0.95 → 0.85 이하
- `H@k`/`R-mh@k`: cosine 없이 v333 baseline 수준 회복
- `acc`: codex 추정 ceiling 0.28~0.32 (LLM 한계 고려)

**실패 모드 / 한계**:
- 새 항이 noise 일 가능성 (특히 PE outlier 영향, segmentation 과보상)
- multi-evidence-topic ~36% 는 importance 만으로 못 잡음 (dormant_top_share=0.64 의 ceiling)
- LLM ceiling — retrieval 개선해도 acc 비례 X (cosine hybrid 도 H@k 두 배 올랐는데 acc 0)

**Smoke (20Q)**: pipeline 정상 (error_rate=0). 본격 평가는 200Q sanity 결과.

---

## 2026-05-14 — per_question_evidence_topics.csv 에서 n_ev=1 제외

**결정**: `scripts/analyze_evidence_topics.py` 가 생성하는 `per_question_evidence_topics.csv` 에서 `n_ev=1` (single-evidence) question 을 처음부터 제외. 기존 24개 CSV 도 일괄 정리 (47,664 → 10,152 행, 78.7% 가 n_ev=1).

**이유**:
- n_ev=1 은 trivially `n_topics_used=1` (single evidence 는 항상 1 topic). segmentation 평가 신호 없음.
- 같은 스크립트의 `evidence_topic_summary.json` 의 qtype 별 stats 는 이미 처음부터 `n_ev≥2 + n_ev_found>0` 필터 적용. CSV 만 unfiltered 였던 비일관성.
- CSV 직접 inspect 시 trivial 행이 79% 라 의미 있는 row 찾기 어려움.

**영향**:
- 코드: `scripts/analyze_evidence_topics.py:124` 에 `if len(flat) < 2: continue` 추가.
- 문서: `context/methodology/infrastructure.md` § 7c 행 수 / 변경 노트 갱신.
- 기존 데이터: `outputs/experiments/**/per_question_evidence_topics.csv`, `outputs_conv/experiments/**/per_question_evidence_topics.csv` 24개 in-place rewrite.
- `evidence_topic_summary.json` 은 변경 없음 (이미 동일 필터).

---

## 2026-05-16 — segmentation 진단: predict_topic 붕괴는 retrieval 버그 아님, 우선 병목은 evidence-topic 과분산 (codex 위임)

**결정**:
- `HiEMSegmenter.predict_topic()` 은 read-only MAP diagnostic 으로 유지하되, QA retrieval 판단 지표로 해석하지 않는다.
- LoCoMo conv0 진단에서 20개 질문 모두 topic 30 으로 붕괴한 현상은 `prev_k`·`lmda=10` sticky prior 가 query-time MAP 을 압도한 결과. 필요 시 sticky 제거 likelihood-only diagnostic 을 별도 API/명칭으로만 추가 (retrieval 연결 금지).
- 실제 우선 병목은 단일 answer evidence 가 여러 topic 으로 흩어지는 과분절. retrieval 은 2026-05-11 결정대로 importance-only.

**이유**:
- LoCoMo conv0: 419턴→69 topic, evidence 가 topic [2,5,26,56] 등으로 분산. `predict_topic(질문)` evidence topic 일치 0/20.
- SEM/SEM2 online assignment 는 sticky/sCRP prior + likelihood MAP 이나, frozen QA query 는 다음 turn 이 아니므로 직전 topic sticky 적용 시 진단 의미 왜곡.

**영향**:
- `context/methodology/infrastructure.md` §7c: evidence 분산 = importance-only retrieval atomicity 리스크로 해석. STM `n_topics` 와 raw segmenter topic count 분리 보고.
- `src/hi_em/sem_core.py`: `predict_topic` 은 retrieval 용 API 아님 (문서화 후보).
- 진단 도구: `scripts/inspect_locomo_segmentation.py` (turn/question 모드).

---

## 2026-05-16 — v3.3.4 EventRNN dead-code: GRU→LLM 교체 아님, α=100 과분절 우선 수정 (codex 위임)

**결정**:
- v3.3.4 의 EventRNN(GRU) scene-dynamics 를 LLM 기반 next-scene predictor 로 교체하지 않는다.
- 1차 병목은 GRU 자체가 아니라 `α=100` fresh-topic prior 가 만든 과분절. 우선순위 = `α/fresh-topic prior·boundary regime 재조정 → evidence concentration 재측정 → GRU vs centroid 재평가`.
- α 조정 후에도 GRU 가 반복적으로 centroid-only 보다 낮으면, 그때 static prototype 후퇴를 SEM 계승 한계로 명시하고 별도 결정.

**이유**:
- LoCoMo conv0, v3.3.4 실제 HP `α=100,λ=10,cos=0.9,β=0.25`: 60턴 topic 배정 `0,1,...,59` (매 턴 새 topic). `rnn_min_history=2` 미충족 → EventRNN 예측 단 한 번도 사용 안 됨 (死문).
- fresh-topic score `log100−0.125≈4.48` vs 기존 best `log11≈2.4` → fresh 구조적 우세. REPORT `n_topics(STM)=4.81` 은 STM eviction artifact.
- α=1 에서 GRU 발동 30턴: `cos(GRU,실제)=0.6688` < `cos(centroid,실제)=0.6873` (GRU 채택 근거 약함, 단 dead-code 의 직접 원인은 α=100).
- GRU 는 SEM2 EventModel 계승 요소. LLM 교체 = EventModel 교체로 3-step 정당화 필요 + main-LLM 결합/cost 문제.

**영향**:
- `context/methodology/v3.3.4.md`: α 큰 regime 에서 EventRNN 비활성 명시 + GRU<centroid 주석 + α sweep 시 `GRU_used_turns` 동반 보고.
- 진단 도구: `scripts/inspect_v334_gru_prediction.py`.

---

## 2026-05-16 — segmentation 평가 프로토콜: LLM 라벨은 gold 아닌 pseudo-label, 1차 metric = evidence cohesion (codex 위임)

**결정**:
- Qwen3.5-9B 가 생성한 boundary/cluster label 을 human gold 로 간주하지 않는다. LoCoMo/LongMemEval 에 topic annotation 이 없을 때의 pseudo_gold/diagnostic reference 로만 사용 (TIAGE/TopiOCQA human label 대체 아님).
- LongMemEval **cross-idx user-turn concat 금지** (서로 다른 idx = 독립 haystack → idx seam 이 인위적 topic boundary). 단일 idx haystack 통째로 사용.
- α×λ sweep 의 1차 목적·metric = evidence-topic co-location (`evidence_cohesion`), 보조 = pseudo-label 대비 boundary F1/ARI/NMI.

**이유**:
- LLM event segmentation 이 인간 분절과 상관된다는 연구는 narrative 환경 중심. Qwen 의 대화 topic label 이 human gold 라는 직접 근거 없음.
- 이전 진단 1순위 = evidence 과분산. HP 를 LLM-label 에 맞추는 것은 downstream 병목을 가릴 수 있어 evidence-cohesion 을 1차로 둠.

**영향**:
- 신규 스크립트 `scripts/llm_pseudolabel_sweep.py` (출력 `outputs/experiments/2026-05-16_llm_pseudolabel_sweep_lme<idx>/`).
- `context/methodology/infrastructure.md`: LLM pseudo-label metric 과 evidence-cohesion metric 역할 분리.

---

## 2026-05-16 — LongMemEval 500Q: hi-em-full-v3.3.4 vs rag — overall 소폭 우위, 채택 보류

**결정**: v3.3.4 (config default α=1,λ=10) 가 RAG 대비 overall +0.014 (0.722 vs 0.708) 이나 **채택 보류 (재검증 필요)**.

**이유**:
- qtype 별 분산: v3.3.4 가 multi-session +6.8pt / single-session-assistant +5.4 / knowledge-update +2.6 우위, 그러나 single-session-preference −6.7 / single-session-user −4.3 / temporal-reasoning −1.5 회귀.
- temporal 회귀가 위 과분절 진단과 방향 일치 — atomicity 손보기 전 채택 근거 약함.
- 단일 run·seed 미고정·std 없음 (CLAUDE.md 3-run 미충족). LongMemEval 은 evidence co-location/H@k 미산출 (analyzer LoCoMo dia_id 전제).

**영향**:
- REPORT: `outputs/experiments/2026-05-16_v334_rag_longmemeval/REPORT.md` (정식판).
- 후속: ≥3-run std + temporal/single-session 회귀 원인(importance policy) 점검.

**관련 — full-context 천장 (cross-ref, 잊지 말 것)**: full-context baseline 을 LongMemEval oracle **500Q** 에 돌린 결과가 `archive/2026-04-28/20260428_freq_shift_full_*` 에 존재 (Qwen3-8B, `--no-thinking`, seg α=10/λ=1). **full 0.734 / rag 0.728 / hi-em-full-v1 0.726 / sliding 0.636 / hi-em 0.562**. 천장(full)이 RAG 대비 **+0.6pt** 뿐 — oracle 은 distractor 거의 없어 RAG cosine top-k 도 증거 대부분 회수, "관련 세션 통째 prefill" 전략의 구조적 헤드룸이 작다. full prefill p50 5850 tok vs rag 2175 (~2.7×) 로 토큰 효율은 RAG 압도. 분절 전략 가치 검증엔 oracle 천장이 너무 낮음 → distractor 多 인 `longmemeval_s`(질문당 ~48세션)에서 full 천장 재측정 필요. (archive 폐기더미라 현행 grep 에 안 잡힘 — 본 cross-ref 가 유일 포인터.)

---

## 2026-05-17 — evidence_cohesion=1 metric 폐기, evidence_recall@K 채택 (codex 위임)

**결정**: LongMemEval temporal-reasoning(idx=374 등)처럼 정답 evidence 가 여러 날짜·scene 에 본질적으로 분산된 경우, `evidence_cohesion=1`(전 evidence 단일 topic)을 segmentation 설계 목표로 삼는 것은 **metric 자체의 오류**로 확정. 폐기. 대체 = `evidence_recall@K` (★turn 을 담은 모든 topic 이 importance 상위 K 에 드는가). Primary = topic-level, 보조 = session-level (LongMemEval label 호환).

**이유**:
- idx=374 의 ★ 6 turn 은 6 개 이질 세션(역사소설추천/AI헬스케어/페미니즘 …)에 1 개씩. 한 topic 으로 묶으려면 그 6 세션을 mega-topic 으로 over-merge 해야 함 → SEM sCRP local MAP / scene atomicity 정면 위배.
- ev_topics 4~6 은 실패가 아니라 scene atomicity 정상 작동 신호. 40 HP run 전부 evidence_cohesion=0 은 Hi-EM 결함이 아니라 metric 기준 오류.
- retrieval 은 atomic topic prefill 이므로 "흩어진 evidence topic 들이 importance 상위 K 에 다 살아남는가" 가 SEM 정합 질문.

**영향**:
- `outputs/experiments/2026-05-16_llm_pseudolabel_sweep_lme374/REPORT.md` 의 evidence_cohesion 해석 정정 대상.
- `context/methodology/infrastructure.md`: retrieval 평가 metric 정의(topic_evidence_recall@K, 보조 session_evidence_recall@K, topic_precision@K, prefill token cost).

---

## 2026-05-17 — v3.3.5: SEM2 f_is_trained cold-start gating 복원 (codex 위임)

**결정**: v3.3.4 의 young-topic centroid-PE 처벌을 제거하고 SEM2 의 `f_is_trained` gating 을 복원한 v3.3.5 (`HiEMSegmenterV335`) 도입. transition 미관측 topic 과 fresh slot 은 동일 untrained likelihood 상수 `L0` 를 받아 likelihood 동률 → young prev 생존은 sCRP λ stickiness 가 결정. gate = `transition_count_k ≥ min_transitions_for_pe`(기본 1, `rnn_min_history` 와 분리).

**이유**:
- SEM2 `log_likelihood_next/f0` 는 `f_is_trained=False` 면 고정 prior 상수 반환(centroid PE 아님). Hi-EM v3.3.4 가 centroid fallback 으로 대체한 것이 SEM2 이탈이자 chicken-and-egg(생존엔 예측, 예측엔 누적, 누적엔 생존) 의 원인.
- 복원이라 SEM 계승 3-step 부담 낮음. v3.3.4 variance calibration 과 무충돌(σ_k² 는 trained 후에만 적용).

**영향**:
- 신규 `src/hi_em/sem_core_v335.py`, orchestrator `version=="v3.3.5"`, `tests/test_sem_core_v335.py` (6), inspect 스크립트 `--version v3.3.5`.
- idx374 α=1λ=10: topic 23→11, #7~#9(+★#7) 동일 topic. methodology `v3.3.5.md`.

---

## 2026-05-17 — v3.3.6: SEM2 persistence+replay event dynamics + 재현성(seed) (codex 위임)

**결정**: v3.3.5 의 3턴 천장(trained 즉시 미학습 공유 RNN 으로 전환 → repeat PE 파국) 해소 위해 v3.3.6 (`HiEMSegmenterV336` + `TopicV336`) 도입: (1) untrained 예측 = identity/persistence(직전 임베딩), centroid fallback 제거 (2) per-topic 독립 모델(공유 가중치 간섭 제거) (3) topic history 전체 replay 학습(n_epochs, SEM2 estimate) (4) `rnn_ready` 를 `f_is_trained` 와 분리(`rnn_ready_min_transitions`). 추가로 EventRNN unseeded 결함(v3.3.4/5/6 공통, 앞선 v3.3.4 REPORT 수치 불일치의 정체)을 **per-topic 결정적 seed**로 수정.

**이유**:
- 트레이스: 'cent' 예측 cos 0.4~0.84(양호) vs 'rnn' cos 0.03~0.5(랜덤) — trained 전환 즉시 예측 붕괴. SEM2 는 untrained=identity, trained=event history batch replay; "min_history 후 centroid→RNN" 은 SEM2 부재 Hi-EM hack.
- 재현성: CLAUDE.md "모든 randomness seed 고정" 위반이었음. 논문 재현성 필수.

**영향**:
- 신규 `src/hi_em/sem_core_v336.py`, `src/hi_em/topic_v336.py`, orchestrator `v3.3.6`(+seed param), `tests/test_sem_core_v336.py` (7), inspect `--version v3.3.6`.
- idx374: topic 11→9, 3턴 천장 해소, 2회 독립 실행 동일. methodology `v3.3.6.md`. infrastructure: seed 정책.

---

## 2026-05-17 — v3.3.7: SEM2 map_variance σ² (n≥2 즉시추정) + 경험적 반증 (codex 위임)

**결정**: `pe_var_min_samples=5` gate 제거, SEM2 scaled-inv-χ² posterior mode `σ²=(ν₀·var₀+n·v)/(ν₀+n+2)` 를 n≥2 부터 적용한 v3.3.7 (`HiEMSegmenterV337`) 도입. restart 도 σ sample buffer 진척(transition_count 와 분리). **그러나 idx374 검증 결과 목표 증상(#14|#15 분절) 미해결 — codex 의 "min-samples gate 가 지배 원인" 진단은 경험적으로 반증됨**(저분산 2샘플에서 σ²가 prior mode 0.04→0.028 로 *하락*해 처벌 강화).

**이유**:
- SEM2 는 prediction_errors 2개부터 `map_variance` 적용(`event_models.py:384,18`); Hi-EM 의 min-samples gate 는 SEM2 이탈.
- 반증: v3.3.7 idx374 grouping 이 v3.3.6 과 사실상 동일(9 topic, T3=#12~14, #15 분절 유지). 변경의 검증 루프가 설계를 반증한 사례 — 기록 보존.

**영향**:
- 신규 `src/hi_em/sem_core_v337.py`, orchestrator `v3.3.7`(+pe_var_df0/pe_var_window), inspect `--version v3.3.7`. methodology `v3.3.7.md` 에 "반증" 명시.
- 다음 진단 = fresh-baseline calibration (아래 v3.3.8 entry).

---

## 2026-05-17 — v3.3.8: fresh/untrained baseline 이 지배 병목, SEM2 prior 로 재calibration (codex 위임)

**결정**: idx374 경계오류의 불변 지배 원인은 σ² 추정이 아니라 **fresh slot 을 `cos_threshold=0.9 → L0=-0.125`(새 topic 이 cos 0.9 로 예측한다는 비현실 가정)로 둔 구조적 SEM2 이탈**. v3.3.8 (`HiEMSegmenterV338`): (a) fresh/untrained L0 를 `pe_prior`(chance 수준 PE, unit 임베딩 E[cos]≈0 ⇒ PE≈1) 기반으로 — `cos_threshold` 가 L0 에 더 안 들어감. d-차원 SEM2 Gaussian 포팅은 스케일 불일치라 scalar-PE 세계에서 SEM2 *원칙*(untrained=chance)만 번역. (b) non-prev 기존 topic 을 repeat 아닌 **f0-likelihood**(SEM2 `k0≠k_prev`) 로 평가.

**이유 / 경험적 한계 (검증 미해결 기록)**:
- pe_prior=1.0(순수 chance, 원칙값) → idx374 mega-collapse(36턴 1 topic). cos_threshold=0.9 → 과분절(23). 작동점 pe_prior≈0.4(cos≈0.6) 는 **embedding-공간 의존이며 SEM2 로부터 도출 불가** — codex 가 2회 "blunt HP" 로 미룬 (c) 가 사실상 불가피.
- non-prev f0 는 LongMemEval 반복 도입문구("I'm looking for book recommendations")에 취약 → 무관 turn 허위병합(#1+#8). 한 결함 고치고 다른 결함 trade.
- 따라서 코드 default pe_prior=1.0(원칙값) 유지, 작동값은 **벤치마크 calibration 대상**(CLAUDE.md: 벤치마크 분석 없이 설계 확정 금지). N=1(idx374) production default 금지.

**영향**:
- 신규 `src/hi_em/sem_core_v338.py`(v337 base), orchestrator `v3.3.8`(+pe_prior), `tests/test_sem_core_v338.py` (7), inspect `--version v3.3.8`. methodology `v3.3.8.md` 에 경험적 한계 명시.
- pe_prior 확정은 longmemeval_s/TIAGE 평가 후속(보류).

---

## 2026-05-17 — LongMemEval session_id 는 외생 hard boundary, retrieval primary 단위 = 세션 내부 SEM topic (codex 위임)

**결정**: longmemeval_s 의 `session_id`/timestamp 는 SEM 이 재발견할 target 이 아니다. 세션은 외생 hard boundary 로만(세션 재현을 segmentation 목표로 삼지 않음). Hi-EM retrieval/evidence_recall 의 SEM 정합 **primary 단위 = 세션 내부 SEM topic**, session-level recall 은 LongMemEval label 호환 보조 지표. 권고 = (iii) within-session SEM 분절.

**이유**:
- SEM/SEM2 sCRP+local MAP+scene dynamics 는 관측 흐름에서 event 를 추론; 주어진 session_id 복제 모델 아님. 한 세션 내부가 topic-비일관(idx374 S1: 역사소설→sci-fi→시).
- distractor 가 외부 코퍼스 시간순 삽입 → 세션 간 인접성은 실제 연속 경험 아님. 세션 무시 전체 재분절 시 sCRP 인위적 연결 위험.

**영향**: longmemeval_s 평가 설계. infrastructure: evidence_recall 단위 정의. v3.3.5~8 무효 아님(해석 전환).

---

## 2026-05-17 — topic-return 으로 evidence_cohesion 부활 안 함 / v3.3.9 = emergent SEM segmenter 재정의 (codex 위임)

**결정**:
1. "세션 분절 잘 되면 증거 세션 유사성으로 topic-return 해 흩어진 evidence 가 한 재귀 topic 으로 모인다"는 가설은 **채택 안 함**. idx374 의 cross-session 유사도는 evidence-thread 가 아니라 generic recommendation opener 에서 발생 → topic-return 강화는 추천요청 템플릿을 묶도록 오최적화 + distractor opener 오회귀. evidence_cohesion 부활 ✗, multi-topic evidence_recall@K + importance policy 유지. non-prev f0/restart 는 보수적 보조 경로로만(강하게 밀지 않음).
2. **v3.3.9 재정의**: session_id hard-boundary 주입이 아니라 **session_id 미사용 emergent SEM segmenter**. 우선 SEM2 원형에 가까운 임베딩 PE/likelihood/fresh-baseline calibration 으로 세션 전환 근처 boundary 자연 발생 검증. `haystack_dates` 기반 time-aware prior 는 SEM2 부재 신규 관측변수 → 3-step 정당화 필요한 v3.3.10 후보로 보류. "세션 1:1 재현" 은 여전히 목표 아님(세션 내부 drift boundary 정상).

**이유**:
- sticky-CRP λ 는 prev_k 에만; 먼 과거 topic 재진입은 count prior+likelihood 만 → distractor 다수면 구조적으로 약함. CLAUDE.md query-cosine 금지라 opener 오회귀 자체 필터 불가.
- session_id 직접 사용은 benchmark metadata leakage 성격, SEM2 "boundary 를 추론" 철학보다 약함. emergent 가 더 SEM 정합.

**영향**: v3.3.9 정의 변경(emergent calibration, time-aware 는 v3.3.10 후보). methodology/infrastructure 반영. 신규 진단도구 `scripts/inspect_longmemeval_s.py`(증거/​distractor 분리), 데이터 `benchmarks/LongMemEval/data/longmemeval_s_cleaned.json`(추가, gitignore 대상).

---

## 2026-05-17 — 주 task 를 Long-Conversation QA → DTS(Dialogue Topic Segmentation) 로 전환, 별도 브랜치 `dts` 분리 (사용자 결정)

**결정**: 프로젝트의 1차 평가 task 를 downstream QA(LoCoMo/LongMemEval accuracy)에서 **DTS(Dialogue Topic Segmentation) 자체 품질**로 전환한다. 작업은 신규 브랜치 **`dts`** 에서 진행하고 `main` 은 QA-baseline 시점으로 보존. 현재 uncommitted 변경(v3.3.5~8, tiage/longmemeval inspection 스크립트 등 ~20파일)은 `dts` 브랜치로 이월(checkout -b, working tree 그대로).

**이유 (근거 데이터)**:
- LongMemEval oracle 500Q 에서 천장(full-context)조차 RAG 대비 **+0.6pt** (full 0.734 vs rag 0.728, ~2.7× 토큰). oracle 은 distractor 거의 없어 RAG cosine top-k 도 증거 대부분 회수 → "관련 세션 통째 prefill" 전략의 구조적 헤드룸이 작다 (2026-05-16 entry cross-ref). QA 우위 입증 경로가 데이터상 막힘.
- 따라서 QA accuracy 우위는 목표에서 제외하고, SEM/SEM2 계승 정체성의 핵심인 **분절 품질**(과분절·과병합 동시 회피)을 1차 목표로 재설정. 평가 체계는 직전 codex 위임 권고(segmentation primary = TIAGE/TopiOCQA/pseudo 대비 ARI·Boundary-F1, retrieval primary = topic-level evidence_recall@K, collapse guard = max_share/new_rate/raw_topics) 를 그대로 채택.
- 브랜치 분리: QA-지향 코드/문서 상태를 `main` 에 보존해 회귀 비교 가능하게 유지하고, DTS 재설계의 파괴적 변경을 격리.

**영향**:
- 브랜치 `dts` 생성(현 작업 브랜치). docs 반영 범위 = decision-log(본 entry) + handoff 만 (사용자 지정). methodology DTS metric 정의는 2026-05-17 codex 위임 entry 들로 이미 확정 — 중복 작성 안 함.
- 완료: TIAGE test 12-method×3-seed 비교(`outputs/experiments/2026-05-17_tiage_all_hiem_full/REPORT.md`) — DTS 첫 확정 측정. 결과: default HP 에서 전 method 실패(v3.3.1~3.3.4-2 every-turn, v1/v3.1.1 과분절, v3.3.5~7 과병합, v3.3.8 1-topic mega-collapse). codex 진단(boundary-F1 게임됨, pe_prior=1.0 붕괴)이 TIAGE 100conv 에서 재현 — idx374 N=1 과적합 아님 확정. 다음: ARI+collapse guard metric 도입 → TIAGE HP sweep.
- 후속 DTS 작업의 primary metric/데이터 셋업은 위 codex 권고 따름.

---

## 2026-05-17 — RNN(learned scene dynamics) 제거/교체 보류 + ARI·collapse guard 도입 (codex 위임 2건 결정)

**질문 (사용자)**: ① v3.3.6(per-topic+persistence+history replay)이 v3.3.5보다 TIAGE에서 나쁘니 learned dynamics RNN을 아예 빼버려야 하나? ② 빼는 게 아니라 f 아키텍처를 RNN보다 현대적 모델(Transformer/SSM 등)로 교체하면?

**결정 (codex `codex:rescue` 2회 위임, 한국어 답변)**:
- ① **(C) 결론 보류 — ARI+HP sweep 선행.** v3.3.6 부진은 "RNN 무용 증거" 아님 — RNN을 제거한 게 아니라 per-topic 분리+persistence prior+history replay 보강 조합이 TIAGE 짧은 대화에서 역효과. + default HP(LoCoMo값, TIAGE 미보정)에서 v3.3.8 mega-collapse까지 나오는 "붕괴 vs 붕괴" 상태라 method 변별 신호 자체가 불안정 → 이 상태로 구조 결정은 위험. RNN 제거(A)는 SEM 핵심 component 제거라 3-step 정당화 부담, 데이터 지지 약함.
- ② **(3) 아키텍처 교체는 부차적 — 학습 신호/데이터 문제 먼저.** SEM2 기본 f = `GRUEvent`(코드에 `LinearEvent`/`StationaryEvent`/`SimpleRNN`/LSTM 계열 존재) → f 교체는 learned dynamics 유지하는 한 SEM 정합 범주(제거보다 정당화 부담 훨씬 낮음). 그러나 TIAGE 15턴/topic·저표본·online·CPU-only 제약상 Transformer/Mamba가 RNN보다 낫다는 근거 없음(과적합·과함). v3.3.5 학습부실 원인 = "RNN이라서"가 아니라 공유 EventRNN+단발 3-step 학습+cross-topic 간섭+unseeded+trained 전환 즉시 랜덤예측 사용 → **저표본 online 학습 + likelihood/fresh-baseline calibration 문제**. 아키텍처 교체로 안 풀림. 교체한다면 더 큰 모델이 아니라 더 단순한 것(`persistence`/`learned linear`/`small GRU`/`EMA-mixer`)이 후보 — 그것도 ARI+sweep 으로 "dynamics 실패 vs calibration 실패" 가린 뒤.

**확정 액션 (분기 없음)**:
1. **v3.3.5 를 현 best baseline 으로 고정.** v3.3.6/7/8 보강은 채택 안 함(폐기 아님, 비교군 유지).
2. **ARI + collapse guard 를 `scripts/run_tiage_full_compare.py` 에 도입** (본 entry 와 함께 구현 완료). ARI = GT segment partition vs 예측 topic-id partition, 대화별 `adjusted_rand_score` macro 평균 — 과분절·과병합·collapse 동시 페널티, collapse-immune → **primary metric**(REPORT 정렬·best 기준 = ARI). collapse_rate = 1-topic 병합 대화 비율; ≥50% 면 `†` 표시 + F1/Pk/WD best 선정에서 제외(v3.3.8 "best WD=0.510" 같은 무경계 artifact 차단). Pk 무변별(0.467~0.521)·WD degenerate 취약성 보강 = ARI 도입 직접 동기.
3. **v3.3.5/6/7 HP sweep** (alpha·lmda·pe_prior) — ARI/WD/n_topics 동시. TIAGE 는 항상 **full(100 conv × 3 seed), subset/sanity 금지**(사용자 결정). GPU 복구 불가(드라이버 11040, CUDA 비활성) 확정 → CPU-only. sweep 가능화 = **임베딩 1회 인코딩 후 캐시 재사용**(HP 무관 불변) + HP 조합 코어 병렬. Crts 프록시는 LLM/임베딩 API 라 세그멘터(EventRNN/sCRP) 연산 오프로드 불가 — 임베딩 단계만 대체 가능하나 캐시가 더 우월.
4. sweep 후 분기: HP 보정해도 v3.3.5 가 GT(4.15 topic) 수렴 실패 → dynamics 실패 → `persistence`/`learned linear`/`small GRU`/`EMA-mixer` ablation(Transformer/SSM 아님). 보정으로 수렴 → calibration 문제, 아키텍처 불변. RNN 제거(A)는 단순 dynamics baseline 에도 패배 확인 시에만 SEM 3-step 정당화와 함께 재상정.

**영향**: `run_tiage_full_compare.py` metric 확장(ARI/collapse_rate 컬럼, ARI 정렬, degenerate † guard). 기존 TIAGE REPORT 3종(`2026-05-17_tiage_v33x_compare`/`_a1`/`_all_hiem_full`)은 Pk/WD 까지만 반영됨 — ARI 는 다음 full run 에서 추가. methodology/infrastructure DTS metric § 갱신 대상. handoff 갱신 대상.

---

## 2026-05-18 — SEM-vs-prev-cos 근본진단 → v3.3.9 (prev-cos 를 SEM PE 로 정합 복원) (codex 위임 2건)

**진단 (codex `codex:rescue` 위임 #1)**: 같은 인코더·같은 gt_shifts+F1 에서 1줄 baseline "직전 발화 cosine < θ → shift" 가 F1 **0.433~0.440** (3 인코더 multi-qa/all-mpnet/MiniLM 모두, ≈ SOTA 0.427). 반면 우리 SEM 전 변형 F1 0.25~0.36, v1 0.363. **인코더 무관·eval 무관 → SEM 파이프라인 자체가 인접 의미거리 신호를 1줄 threshold 보다 못 살리고 죽임.** 근본원인 = fresh/untrained baseline `L0` 오보정 + scalar cosine-PE σ² 포화 (v3.3.5 `L0=−0.125` 과분절 / v3.3.8 `L0=−12.5` collapse; sCRP·RNN 단독 주범 아님). prev-cos 는 SEM2 untrained `predict_next` = identity dynamics `f(x)=x_prev` 의 cosine PE 그 자체 → SEM 밖 heuristic 아님.

**결정**: codex 권고 **(A) prev-cos 신호를 SEM likelihood/PE 에 SEM-정합 복원** 채택. **CLAUDE.md cosine 금지는 retrieval 한정** — segmentation 의 prediction-error 는 본질적으로 인접 의미거리이고 SEM PE 가 prev-cos 의 일반화이므로, segmentation 단계 `cos(s_{t-1},s_t)` 사용은 retrieval-cosine 금지 위반이 **아니라 SEM2 cold-start 복원**임을 명시(예외 아님, 규칙 범위 밖). SEM 계승 3-step: (1) SEM2 에 있음(identity dynamics PE), (2) sCRP/Bayes/local MAP 와 충돌 없음(prior 상쇄만, 제거 아님; η<1 시 learned f 혼합 유지), (3) 본 entry.

**구현 → v3.3.9 (`sem_core_v339.py`, v3.3.8 기반)**:
- repeat(k=prev_k, trained·untrained 공통) `Lδ(δ_eff)`, `δ_eff²=η·δ_prev²+(1−η)·δ_model²`, η 기본 1.0.
- **회귀 (codex 위임 #2)**: v3.3.9-pre 가 ARI 0.36→0.126, n_topics 2.4 과병합. 원인 = 고정 `r0=λ/α` prior mismatch(`P_prev/α` 가 turn 진행에 커져 threshold 밀림) + σδ² 를 전체 δ_prev population variance(경계 점프 혼합)로 online calibration → discrimination 씻김.
- **codex calibration-fix (#2)**: ① σδ² = **고정** softness temperature `c·δ*²` (c 기본 1/16; online calibration 폐기) ② fresh baseline 을 **time-dependent prior-corrected** `B0_t = Lδ(δ*) + log(P_prev/P_new)` → `_scores` 의 prior 항과 상쇄되어 prev-vs-fresh MAP 결정이 **매 turn `δ_eff < δ*` hard cut 과 정확 동치** ③ untrained prev 도 δ_prev gate(η=1 ⇒ δ_eff=δ_prev; 기존 `_l0()` 무조건 병합 bias 제거) ④ `δ*` 를 **TIAGE train split** prev-cos 에서 재추정(0.5557 @ cos_th 0.4443, F1 0.437, n_conv=300) — test leakage 제거(이전 0.536 은 test 유래라 폐기).

**결과 (TIAGE dev full, α=1 λ=10, δ* train, seed0)**: ARI 0.352 / WD 0.628 / **F1 0.440** / Pk 0.412 / n_topics 7.1 / collapse 0%. **F1 0.257→0.440 = prev-cos baseline·SOTA F1 동급 회복** (SEM 이 신호 더 이상 죽이지 않음 = 진단 근본목표 달성, bimodal 붕괴 해소). 단 WD 0.628 여전히 나쁨(약한 과분절 nt 7.1 ≫ GT 4.15) → 구조 붕괴는 해소, calibration 거리 문제로 전환.

**영향**:
- 신규 `src/hi_em/sem_core_v339.py` + `context/methodology/v3.3.9.md` + `run_tiage_full_compare.py` 배선(import/factory/METHODS/SWEEP_CLASSES) + handoff 갱신.
- **full TIAGE test 3-seed 확정 (2026-05-18, `outputs/experiments/2026-05-18_tiage_v339_full/`)**: v3.3.9 = ARI 0.408 / F1 0.437 / Pk 0.415 / WD 0.605 / nt 6.9 / collapse 0% — 13-method 중 **primary(ARI)·F1·Pk 동시 best** (이전 best ARI v3.3.6 0.359 → +0.049; F1 = prev-cos test 0.433 동급/근소상회 = SEM 이 1줄 baseline 따라잡음, 진단 근본목표 완수). dev 0.440↔test 0.437 일관. **v3.3.9 = 새 baseline** (v3.3.5 잠정 baseline 대체 확정). 잔여 약점 = WD 0.605 ≫ SOTA 0.420 (약과분절 nt 6.9 ≫ GT 4.15; best WD 여전히 v3.3.5 0.598).
- **codex 위임 #3 (2026-05-18, RNN→BERT 질문)**: 권고 **(C) BERT-f 보류**. 병목은 f 아키텍처 아니라 calibration — v3.3.9 가 η=1(RNN 미사용)로 F1 회복. BERT-f 는 RNN 보다 파라미터 커서 TIAGE 저표본·CPU·online 에서 v3.3.5 학습부실 악화 위험; supervised BERT classifier 는 SEM unsupervised 철학 정면충돌(+그 "0.55~0.65"가 supervised 면 비교 unfair). 순서 = (i) v3.3.9 test 확정[완료] → (ii) strong similarity 천장(TextTiling-SBERT depth/windowed/adaptive) 우리 eval 측정 → (iii) BERT-f 는 η<1 에서 δ_model 이 δ_prev 를 dev 에서 유의 개선할 때만 승격. + v3.3.9 HP sweep(δ*·σδ_c·η·α·λ, n_topics→4.15 로 WD/ARI 동반개선).
- **SOTA caveat (codex #1·#2)**: SOTA(F1 0.427/Pk 0.40/WD 0.42) 출처·split·boundary 정의·threshold 선택 방식 사용자 미확정 → 외부 SOTA 초과 주장 금지. "같은 인코더·eval 에서 SEM<prev-cos→v3.3.9 가 prev-cos 동급 회복" 결론은 caveat 무관 유효. HP(δ*/σδ_c/r0-strategy)는 train/dev-tune 에서만 선택, test 1회 평가 절차 준수.
- Stage A sweep 은 103/162 BrokenPipe 중단(partial 보존); codex 진단상 구조문제라 재개 critical-path 아님 — v3.3.9 sweep 으로 대체.

---

## 2026-05-18 — 판정 지표를 ARI-primary → **WD/F1/Pk target** 으로 변경 (사용자 결정, 2026-05-17 ARI-primary entry 대체)

**결정 (사용자)**: TIAGE 평가의 판정·정렬·best 기준을 **WD↓ / F1↑ / Pk↓** 로 한다. 이유: 외부 비교 — TIAGE/segmentation 문헌·SOTA 는 Pk/WD/F1 로 보고하고 ARI 를 안 쓴다. 내부 robustness 용으로 ARI 를 primary 로 뒀던 2026-05-17 entry(codex 백킹)를 본 entry 가 **대체**한다.

**유지하는 가드 (제거 시 과거 오판 재발 — 반드시 동반)**:
- **ARI 를 가드 컬럼으로 존치**: F1 은 과분절로 게임됨 (v3.3.1~3.3.4-2 가 매-turn 경계로 F1 0.354 받는데 ARI≈0 = GT 분할 무상관). F1 단독 정렬 시 이 degenerate 가 상위. ARI 컬럼이 over-seg 탐지기.
- **collapse-guard † 존치**: collapse≥50% method 는 *모든* best 에서 제외. 없으면 v3.3.8 (1-topic, F1=0) 이 "best WD 0.510" 으로 다시 뽑힘 (무경계 artifact).
- **n_topics 컬럼 존치** (GT≈4.15) — 과분절·과병합 직접 가시화.
- best F1/WD/Pk 는 전부 non-degenerate(collapse<50%) 한정.

**구현**: `run_tiage_full_compare.py` 메인 REPORT + sweep REPORT 둘 다 — 정렬 key `ari→f1` desc, 선두 컬럼 `F1/WD/Pk`, `ARI(guard)/n_topics/collapse` 후미 가드, best F1/WD/Pk(target) + best ARI(guard, non-target) 라벨. docstring/헤더 문구 갱신. f1_s(std) 집계 추가.

**영향**: 향후 모든 TIAGE REPORT 가 WD/F1/Pk 정렬·판정. 기존 산출(`2026-05-18_tiage_v339_full` 등)의 "best ARI" 해석은 가드로 강등 — v3.3.9 가 F1 0.437/Pk 0.415 로 target 기준에서도 best(non-degenerate)임은 불변(WD 0.605 는 약점 그대로). methodology v3.3.9.md / handoff 갱신 대상. SOTA 비교 caveat(출처·split·supervised 미확정, 외부 초과주장 금지) 그대로 유효.

---

## 2026-05-18 — 프로젝트 default segmenter 를 v3.3.9 (현 BEST) 로 (사용자 지시)

**결정 (사용자: "지금까지의 BEST를 디폴트로")**: `orchestrator.HiEM` 기본 `version` 을 `"v2"` → **`"v3.3.9"`** 로 변경. v3.3.9 분기 추가(없었음). 근거: full TIAGE test (target WD/F1/Pk + ARI guard) 13-method 중 v3.3.9 가 F1 0.437 / WD 0.605 / Pk 0.415 / ARI 0.408 로 4개 전부 best(non-degenerate). v3.3.9 `__init__` 기본값(eta_prev=1.0·delta_star=0.5557·sigma_delta_c=0.0625·α=1·λ=10)이 곧 그 BEST config 이므로 orchestrator 분기는 공유 HP 만 passthrough, v3.3.9-specific calibration default 는 미override (BEST 위임). run_tiage `_factory_v339` 도 이미 클래스 default(=BEST) 사용.

**caveat**: orchestrator default 변경은 `version` 명시 안 한 호출자(구 QA 파이프라인)에 영향. 실험 harness(`experiment.py`/`run_experiment.py`/`run_tiage_full_compare.py`)는 version 을 명시 전달하므로 무영향 — 본 변경은 *canonical default* 설정 성격. RNN 처리(A3 identity 고정)·v3.3.10(causal-window) 설계 결정은 미반영(별도, decision-log 2026-05-18 RNN/A3 entry 참조). δ*=0.5557 은 TIAGE train 의존(타 벤치 재calibration 필요).

---

## 2026-05-18 — 표준 segmentation 벤치 도입(SuperDialseg/Dialseg711) + v3.3.10 실벤치 검증 (사용자 지시)

**결정/근거**: TIAGE 단일(잡담형, 노이즈 큼) 한계 → 표준 dialogue-segmentation 벤치 **SuperDialseg + Dialseg711** 도입. 소스 = GitHub `coldog2333/SuperDialseg`(clone, `benchmarks/superdialseg/`+`benchmarks/superdialseg_data/`, gitignored). 신규 `scripts/run_superdialseg_eval.py` — 그들 **공식 Pk/WD/F1 + Score**(=0.5·F1+0.25·(1−Pk)+0.25·(1−WD), `metrics/segmentation.py` verbatim, window='auto') 사용해 문헌-비교 가능. GT = dataset `segmentation_label`(last→0 공식 규약). 임베딩 캐시·우리 segmenter→seg-id→boundary 변환.

**SuperDialseg test 결과 (1322 dial/17328 turn/bnd 0.232, `outputs/experiments/2026-05-18_superseg_test_v3310/`)**:
- prev-cos@oracleθ(천장): Score 0.447 / F1 0.458 / Pk 0.467 / WD 0.660
- v3.3.9 @TIAGE-δ* zero-shot: Score 0.460 / F1 **0.449** / Pk 0.468 / WD 0.589 / predr 0.418
- **v3.3.10 @TIAGE-cfg(m2,ρ0.7,a0.5,δ*0.5594) zero-shot: Score 0.463 / F1 0.432 / Pk 0.471 / WD 0.541 / predr 0.316**
- 전 oracle-δ* 행 수렴 → F1 0.458 / Score ~0.447

**판정**:
1. **v3.3.10 > v3.3.9 (Score·WD·과분절억제)** — Score 0.460→0.463, WD 0.589→0.541, predr 0.418→0.316(GT 0.232 근접). codex causal-window FP-감소 예측이 *구조화된 실벤치*에서 실증. **TIAGE-train "+0.006=노이즈" 판정은 TIAGE 잡담 특유의 가림**이었음 — v3.3.10 무가치 아님 (이전 ledger/판정 정정).
2. 단 F1 은 v3.3.9 우위(0.449>0.432): v3.3.10 경계 덜 찍어 recall↓ ↔ WD/Score 이득. "이긴다"는 지표 의존(Score=공식 종합 기준 v3.3.10).
3. **인접-cosine unsupervised 천장 ≈ F1 0.46 / Score 0.45** (prev-cos·v3.3.9·v3.3.10 oracle 전부 수렴). 문헌 0.55~0.65 는 unsupervised 도달 불가 → **supervised regime** 강하게 시사 (codex SOTA-caveat 실증). 인코더·split·supervised 미확정이라 외부 우열 주장 금지.

**영향**: 신규 벤치/스크립트/데이터 + ledger 를 전 seg벤치로 확장(`outputs/reports/tiage_hp_sweep_ledger.md`) + handoff 갱신. 다음: Dialseg711 eval(어댑터 준비됨), superseg-val δ* 재calibration(현재 TIAGE-δ* zero-shot 전이라 향상 여지), 인코더 정합 ablation, η=1 RNN-skip fast-path(eval ~4배 가속 — RNN dead-weight forward 제거). v3.3.10 은 SuperDialseg Score-우위라 default 승격 후보(단 F1 trade·δ* 미보정이라 보류, superseg-val calibration 후 재판정).

---

## 2026-05-18 — Dialseg711 결과 + v3.3.10 → v4.1.1 rename (새 메이저 라인) (사용자 지시)

**Dialseg711 test (711d/19350t/bnd 0.142, 공식 SuperDialseg metric, `outputs/experiments/2026-05-18_dialseg711_test_v3310/`)**:
- prev-cos@oracleθ(천장): Score 0.590 / F1 0.536 / Pk 0.322 / WD 0.389
- v3.3.9 @TIAGE-δ* zero-shot: Score 0.514 / F1 0.492 / Pk 0.394 / WD 0.534
- **v4.1.1(구3.3.10) @TIAGE-cfg zero-shot: Score 0.590 / F1 0.549 / Pk 0.325 / WD 0.415**
- v4.1.1 @oracle-δ*: Score 0.629 / F1 0.560 / Pk 0.285 / WD 0.319

**판정**: v4.1.1 zero-shot 이 **튜닝 0 으로 유사도 천장(Score 0.590) 매칭**, v3.3.9 대비 **+0.076 Score**(F1 0.492→0.549). superseg 에 이어 두 번째 표준벤치에서 causal-window 우위 일관 — TIAGE-train "+0.006=노이즈" 는 잡담형 특유 가림이었음 확정. dialseg711(합성·경계 선명)은 유사도 방법 적합 벤치, oracle Pk 0.285/WD 0.319 = unsupervised 문헌 경쟁권(단 oracle=test-tuned 상한, 외부주장 불가; zero-shot 0.590 이 정직값=천장).

**rename 결정 (사용자)**: v3.3.10 → **v4.1.1**. 이유: v3.3.x = "SEM2 RNN scene-dynamics 복원" 계열인데, η-ablation(RNN 무용~유해) + v3.3.9(prev-cos=identity PE 정합복원)로 사실상 learned-dynamics 를 버리고 identity+calibrated-MAP 로 전환됨. v4.1.1(identity+causal-window)은 v3.3.x 정체성과 질적으로 다른 라인 → **메이저 분리**. RNN 코드는 `use_rnn` 플래그로 보존(제거 아님, η=1 자동 skip).

**구현**: `git mv sem_core_v3310.py→sem_core_v411.py`, `calibrate_v3310_delta_star.py→calibrate_v411_delta_star.py` (이력 보존). 코드 일괄 치환 HiEMSegmenterV3310→V411 / v3310→v411 / V3310_→V411_ / v3.3.10→v4.1.1 (run_tiage_full_compare·run_superdialseg_eval·factory·METHODS·SWEEP). 신규 `context/methodology/v4.1.1.md`. **pre-rename 산출물**(`outputs/experiments/2026-05-18_{superseg,dialseg711}_test_v3310/`, ledger 표의 exp명)은 *역사적 명칭(v3310) 유지* — 커밋된 참조·git history 보존. 이후 신규 산출물부터 v411.

**영향**: orchestrator default 는 여전히 v3.3.9 (v4.1.1 승격은 superseg-val δ* 재calibration 후 재판정 — 보류). 다음: superseg val-calibrated 재평가(`--calib-dataset superseg --calib-split validation`, v411 코드). ledger/handoff 갱신. git: dts push (정책 2026-05-18: 명시요청 시 Claude 직접 실행).

---

## 2026-05-19 — superseg val-calibrated δ* 재평가: "miscalibration artifact" 가설 반증

**실행**: `2026-05-18_superseg_test_v411_calib` — superseg validation 에서 best-F1 δ* 산출(δ*_prev=0.520, δ*_eff=0.502; TIAGE 0.556/0.559 보다 낮음) → superseg test 재평가(v3.3.9·v4.1.1, v411 코드).

**결과 (공식 metric, GT bnd 0.232)**:
- v3.3.9: TIAGE-δ* zero-shot Score 0.460 → superseg-val-δ* **0.451** (하락). F1 0.449→0.455(↑) but WD 0.589→0.637·pred 0.418→0.500(과분절↑).
- v4.1.1: TIAGE-cfg zero-shot Score 0.463 → val-cal **0.453** (하락). F1 0.432→0.451(↑) but WD 0.541→0.615·pred 0.316→0.461.

**판정**:
1. **"superseg 저조는 TIAGE-δ* miscalibration artifact" 가설 = 반증.** val-calibration 이 Score 를 오히려 낮춤. 앞선 superseg 수치는 calibration 탓이 아니라 **진짜 task 난이도** — 전 method F1~0.45/Score~0.45 가 인접-cosine 유사도 천장(F1 0.46)에 붙어있음.
2. **원인 = calibration objective 불일치**: val-cal 이 *F1* 을 최대화 → F1-최적 δ* 는 과분절(pred_rate 0.46~0.50 ≫ GT 0.232) → WD/Pk 악화 → Score 순손실. (F1-게임-by-과분절; ARI 가드 존재 이유와 동일 패턴.) **교훈: target=Score/WD 면 δ* 를 F1 로 calibrate 하면 안 됨 — Score/WD-aware calibration 필요.**
3. **v4.1.1 ≥ v3.3.9 일관** (Score 두 regime, zero-shot WD 크게 우위). causal-window 가치 유지.
4. **v4.1.1 운용점 = TIAGE-cfg zero-shot (Score 0.463) 확정.** val-calibrated 승격 불가(Score 더 낮음). orchestrator default 는 v3.3.9 유지(v4.1.1 default 승격은 TIAGE/Dialseg711 재확인 + Score-aware calib 후 재판정 — 보류).
5. SOTA/문헌 0.55~0.65 가 unsupervised 로 superseg 에서 도달 불가(천장 F1 0.46) 재확인 → supervised regime. 외부 초과주장 금지(인코더·split·supervised 미확정).

**영향**: ledger/handoff 갱신. 다음 후보: (a) Score/WD-aware δ* 탐색, (b) 인코더 정합 ablation, (c) 목표 포지셔닝(unsupervised-competitive 인정 vs supervised 도입) codex 위임.

---

## 2026-05-19 — per-topic δ* (v4.1.2-exp): 구현·검증 → "안전하나 무이득", default 승격 안 함

**결정**: 전역 δ* 의 v4.1.1 을 default 로 **유지**. per-topic δ*_k 는 `src/hi_em/sem_core_v412_exp.py`(`HiEMSegmenterV412Exp`) 로 **experimental 구현 + 실측 검증 완료** — codex 2026-05-19 P2 그대로, default 승격 **안 함**.

**구현 (codex 안전 수식)**: D_k = accepted within-continuation δ_eff only(boundary/fresh/same-label-restart 제외 — v3.3.9-pre mixture 붕괴 차단). n_k<N_min(6) → γ=0 → δ*_k=δ*_0 (전역과 정확 동일, graceful fallback). n_k≥N_min → r_k=Q0.80(D_k)+κ·MAD, γ=γ_max·(m+1)/(m+1+ν), δ*_k=clip((1−γ)δ*_0+γ·r_k, [0.85,1.15]δ*_0). σδ² 전역 고정 유지. prev-topic δ*_{k_prev} 로 prior-corrected fresh baseline → repeat>fresh ⟺ δ_eff<δ*_{k_prev} 동치 유지. sanity: 짧은토픽 v412exp==v411 비트동일, 긴토픽 발동 확인.

**검증 (공식 metric, 임베딩 캐시, RNN-skip)**:
- SuperDialseg test: v4.1.2-exp Score 0.463/F1 0.432/Pk 0.471/WD 0.541/predr 0.316 = **v4.1.1 과 모든 자리 동일** → 가드 통과(짧은 segment 3~4턴<N_min → 무발동). 무해 실증.
- Dialseg711 test: v4.1.2-exp Score **0.589** vs v4.1.1 0.590 (F1/Pk 동일, WD 0.415→0.416) → **이득 없음, 노이즈 내 미세 손실**. "긴 대화서 per-topic 이득" 가설 **데이터 기각**.

**원인 (구조적, 재오픈 조건 명시)**: segment ~7턴(dialseg711)이라도 within-continuation N_min=6 모이면 segment 가 거의 종료 → δ*_k 가 segment 끝 1~2턴에만 늦게 발동 → 순효과≈0. **안전 게이트(N_min, 붕괴방지 필수) vs 발동시점 근본 충돌**: N_min 낮추면 v3.3.9-pre식 noise/붕괴 위험, 안전하면 표준 DTS 대화길이엔 너무 늦어 무의미. → per-topic δ* 는 *훨씬 긴 대화(segment ≫ N_min)* 데이터에서만 재검토 가치. 현 벤치(TIAGE/superseg/dialseg711)에선 종결.

**영향**: v4.1.1 = 현 라인 유지(orchestrator default 는 여전히 v3.3.9; v4.1.1 승격은 별건). v4.1.2-exp 코드·어댑터 배선 보존(재오픈용). methodology 는 v4.1.1.md 에 본 negative result 한 줄. ledger/handoff 갱신. 깨끗한 negative result — 가설 검증·기각·메커니즘 규명 완료.

## 2026-05-19 — baseline: Def-DTS / Def-DTS-based-online 분리

**맥락**: DTS baseline 으로 Def-DTS(ElPlaguister, ACL2025) 를 Crts/gpt-4o 로 도입. Def-DTS 는 offline·whole-dialogue (대화 1개 = LLM 1콜, 전체 발화 joint intent 추론). 사용자가 Hi-EM(online·턴단위)과 공정 latency 비교 위해 "턴당 1콜" 개조 요구.

**결정 (codex:rescue 위임 + 후속 논의)**:
1. Def-DTS 원형은 **개조하지 않음**. 대화당 latency + amortized 턴당(=대화latency/발화수, "offline 사후 분배 추정치"로 명시) 으로만 보고. 논문 비교 가능성 보존.
2. 진짜 online 턴당 latency 가 필요하면 **별도 baseline `Def-DTS-based-online`** 신설. 정의: 턴 t 에서 `dialogue[0:t]`(미래 미관측)만 같은 Def-DTS 프롬프트에 넣고 **마지막 발화 topic_shift 만** 채택 → 턴당 1콜. **Def-DTS 결과와 절대 혼합 금지**, 이름·REPORT 분리.

**근거**: (a) "전체 대화 매턴 반복"(변형 a)은 미래 발화를 알아야 하므로 정의상 online 불가(offline 반복). (b) online 은 필연적으로 prefix-only(변형 b) → 문맥이 joint→causal 로 바뀌어 예측·점수가 원 Def-DTS 와 달라짐 → 다른 방법. 같은 프롬프트 텍스트라도 *적용 입력*이 달라 계산되는 양이 다름. 정직한 벤치마킹 = 다른 방법은 다른 이름.

**영향**: `scripts/run_defdts_online.py` 신설(전용 experiment/REPORT). 목적은 정밀 점수 아닌 데이터셋별 턴 100개 표본 평균 latency → full 불필요, 누적 발화≈100 까지 대화 표본. resume+사이드카(crash-safe), workers=1(비경합 isolated 턴 latency). Crts USD quota 복구 전 실행 불가(429) — 코드/하네스는 선구축.

## 2026-05-19 — baseline: Plain LLM prompting (online) 신설

**맥락**: 비교표(Offline CSM/Def-DTS [bi-direction] vs Online CSM/Plain LLM prompting/Ours [past-only]) 의 "Plain LLM prompting | past-only" 행 충당. 원 SuperDialseg Plain Text Prompting(`benchmarks/superdialseg/.../models/llm` ChatGPTSegmenter) 은 offline·whole-dialogue·대화당 1콜.

**결정 (codex:rescue 위임 결과, Def-DTS-online 과 동일 논리)**: 원형 개조 금지. 진짜 online 이 필요하면 **별도 baseline `Plain LLM prompting (online)`** 신설 — turn t 에 `U1..Ut`(미래 미관측)만 동일 plain 프롬프트로 1콜, Ut 에서 새 part 시작 여부 = 경계. offline 결과와 **표·집계·해석 분리**, prefix-causal 임을 REPORT 명시. parsing 규칙·실패처리 명시 필수(codex caution #4).

**산출**: `scripts/run_plainprompt_online.py` (run_defdts_online.py 동형: Crts gpt-4o, workers=1 비경합 latency, resume+사이드카, 데이터셋별 누적 발화≈100 표본). 수집 컬럼 = Pk/WD/F1/**Score(0.5·F1+0.25·(1-Pk)+0.25·(1-WD), 공식 SuperDialseg)**/턴당 latency(ms)/턴당 LLM 호출수/턴당 토큰. 전용 experiment/REPORT.

**영향**: 비교표 행 매핑 — Offline Def-DTS=run_defdts_crts.py(완료), Plain LLM prompting(online)=본 스크립트, Online CSM=별도(추후), Ours=Hi-EM. 모든 행 동일 metric/표본 정의로 채워야 공정.

## 2026-05-20 — `methods/` 디렉토리 신설 (offline 원본 / online 수정본 정리)

**결정(사용자)**: baseline 의 원본(offline)·Hi-EM 수정본(online)을 레포 홈
`methods/` 에 정리. 범위 = TextTiling, BayesSeg 둘만(추후 확장 가능).
방식 = **A(wrapper, 코드 복사 없음)**: `benchmarks/superdialseg`(read-only)
원본 무복사·무수정, `methods/<m>/offline.py` 는 원본 알고리즘 호출만,
`online.py` 는 `scripts/run_*_prefix.py`(검증본) 실행 진입점.

**근거**: CLAUDE.md `benchmarks/* 읽기전용·코드복사 금지` 준수 + offline/
online 을 동일 harness(Def-DTS 번들 데이터, autoseg Pk/WD+F1, Score=
0.5F1+0.25(1-Pk)+0.25(1-WD))로 비교. online(prefix-causal)은 codex
2026-05-20 결정대로 AUXILIARY(핵심 5행표 제외, Pk/F1 indicative,
latency 가 비교값). offline Pk/F1 은 원 SuperDialseg 논문값과 데이터·
공식 metric 차이로 정확 일치 아님(방향·정상동작 검증용).

**영향**: 신규 `methods/{texttiling,bayesseg}/{offline,online}.py`+README.
산출 `outputs/experiments/<name>/REPORT.md`. architecture 문서 반영.

## 2026-05-20 — TextTiling-online-streaming 신설 (true O(w)/turn 변형)

**맥락**: 기존 `methods/texttiling/online.py` (= `scripts/run_texttiling_prefix.py`)
는 인터페이스만 causal (past-only) 일 뿐 매 turn `nltk.TextTilingTokenizer.tokenize(utts[:t])`
를 fresh 호출 → 전체 O(n²) recompute. *online baseline 핵심 비교값 = per-turn
latency* (decision-log 2026-05-20 위 entry) 라는 정책에 맞춰, 진짜 streaming
TextTiling 을 별도 method 로 추가.

**codex:rescue 위임 결과 (한국어 답변)**:
- 옵션 1 (진짜 streaming) 채택. 단 **이름 분리** — `TextTiling-online-streaming`
  (또는 `CausalTextTiling-RS`) 로 호명. 원본 NLTK/SuperDialseg TextTiling 점수
  *재현 안 함*. running mean/std threshold + one-sided depth 라 boundary set
  자체가 NLTK 와 다름. 정직성 핵심 = 같은 표·같은 이름으로 섞지 않기.
- **유지**: lowercase/punct/stopword, block-cosine, depth-score valley 개념.
- **변형**: depth threshold = Welford running mean+c·std; close-boundary
  suppression = 최근 boundary 와 거리 < min_gap 인 causal rule.
- **폐기**: NLTK 전체 문서 재토큰화, 전체 depth 재정렬, 미래 정보 기반 global
  revision.

**결정**: `src/hi_em/baselines/texttiling_streaming.py` 의 `StreamingTextTiling`
class (push()/flush() API) + `methods/texttiling/online_streaming.py` runner
(tiage anno_test.json 직접 로드, segeval 직접, **Def-DTS 의존 없음**) +
`tests/test_texttiling_streaming.py` (11 테스트).

**HP default 결정**: codex 권고 w=10/k=6 은 SuperDialseg 기준. tiage (평균
대화 16발화 × ~5단어) 에선 2k pseudo-sentence 못 채워 boundary 0. runner 의
default 를 w=5/k=3/min_gap=3 으로 축소 (REPORT setup §에 근거 명시). class
default 는 w=10/k=6 유지 (Hi-EM 본체 통합 등 일반 사용에 fair).

**산출 (3-benchmark full test set, 2026-05-20)**:
`outputs/experiments/2026-05-20_texttiling_streaming/REPORT.md`

| dataset | n(dial/turn) | Pk | WD | F1 | Score | lat/turn mean | p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| tiage | 100/1564 | 0.527 | 0.548 | 0.225 | 0.344 | 0.008ms | 0.015ms |
| dialseg711 | 711/18639 | 0.471 | 0.490 | 0.271 | 0.395 | 0.010ms | 0.020ms |
| superseg | 1322/16006 | 0.462 | 0.467 | 0.262 | 0.399 | 0.011ms | 0.026ms |

Pk/WD/F1 = INDICATIVE (codex 의 intended algorithmic difference). **per-turn
latency 모두 ms 단위 이하** — baseline 핵심 주장 검증. nltk prefix-recompute
(`run_texttiling_prefix.py`) 와의 turn-by-turn diff 는 별도 작업.

**영향**: `methods/README.md` 3종 구분 (offline / online prefix-recompute /
online streaming). `pyproject.toml` 에 `segeval` 의존 추가. `benchmarks/Def-DTS`
clone (데이터 로드 전용). SEM 계승 원칙 적용 대상 아님 (TextTiling 은 SEM
외부 baseline).

## 2026-05-21 — GreedySeg-online-delay2 신설 (BERT bounded-lookahead)

**맥락**: SuperDialseg 의 GreedySeg (Jiang et al. 2023, `models/greedyseg/
modeling_greedyseg.py`) 를 online 비교표에 추가. 원본 = offline whole-document
BERT cosine segmentation (left/center/right context, larger=max, argmin greedy
선택, 비가역).

**codex:rescue 위임 결과 (한국어)**:
- 옵션 B (**bounded-lookahead, delay=2**) 채택. 원본 score 공식·HP·argmin
  선택을 *그대로 보존*. 입력 인터페이스 streaming + boundary emit 만 right
  context (WINDOW_SIZE=2 미래 발화) 확보 후로 지연.
- **codex 2026-05-21 algorithm-integrity 검증 통과**: 강한 명명
  `GreedySeg-online-delay2` 정직 — score 공식·greedy 선택·HP·encoder 모두
  보존. **본 plan 의 3 baseline (TextTiling-streaming, GreedySeg-delay2,
  GraphSeg-window-d) 중 유일하게 "5행 핵심표 가능"** (codex 검증). 단 offline
  결과와 별도 열/블록 분리 보고 의무.
- *strict prefix-causal 은 아님* — right_sent 의존성. 이름이 강함을 정당화하는
  근거 = "same scoring/selection, delayed emission".

**device-agnostic 정책 (codex 2026-05-21 권고)**:
- `--device {auto,cuda,mps,cpu}` (default `auto`), 우선순위 cuda → mps → cpu.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` env var — torch/transformers import *이전*
  설정 필수 (runner 모듈 최상단 `os.environ.setdefault`).
- `model.to(device)` + tokenizer output 도 동일 device 이동.
- `model.eval()` + `torch.inference_mode()` 고정.
- 결정성·reproducibility: CPU > CUDA > MPS. seed·버전·device·fallback 여부
  REPORT 명기. 동일 device 반복 측정.

**결정**: `src/hi_em/baselines/greedyseg_delay2.py` 의 `GreedySegOnlineDelay2`
class (push()/flush()/state() API, lazy BERT load) + `src/hi_em/baselines/
_device.py` (`resolve_device`, `enable_mps_fallback`) + `src/hi_em/baselines/
_seg_utils.py` (TextTiling-streaming 과 공유: parse_defdts / bnds_to_masses /
boundary_set_f1 / latency_stats / pk_wd) + `methods/greedyseg/online_delay2.py`
runner + `tests/test_greedyseg_delay2.py` (6 ea, tiny-distilbert fixture).

**미해결 / 가정**:
- 원본 GreedySeg 의 BERT pooling 방식 (CLS vs mean) 확인 못함 — superdialseg
  install 없음. 본 구현 = **segment-concat → [CLS] embedding** default. REPORT
  한계 §에 명시.
- 원본 paper 점수와 직접 비교 불가 (데이터·metric·인터페이스 모두 차이) —
  방향성·정상동작 검증용.

**산출 (3-benchmark full test set, 2026-05-21 device=mps M4 Pro)**:
`outputs/experiments/2026-05-21_greedyseg_online_delay2/REPORT.md`

| dataset | n(dial/turn) | Pk | WD | F1 | Score | lat/turn mean | p95 | bert_fwd/utt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| tiage | 100/1564 | 0.537 | 0.554 | 0.142 | 0.298 | 13.8ms | 67.3ms | 2.13 |
| dialseg711 | 711/18639 | 0.416 | 0.443 | 0.412 | 0.491 | 16.3ms | 81.8ms | 2.72 |
| superseg | 1322/16006 | 0.507 | 0.511 | 0.278 | 0.384 | 9.1ms | 41.4ms | 1.91 |

Pk/WD/F1 = INDICATIVE (원본 paper 점수와 데이터·metric·인터페이스 모두 차이).
**5행 핵심표 대상** (codex 검증 통과) 이지만 *offline GreedySeg 결과와 별도
열·블록 분리 보고 의무*. lat/turn 9-16ms (BERT forward 2-3회/utt) — encoder
cost 때문에 TextTiling-streaming (0.01ms) 과 같은 latency 표 배치 금지.

**영향**: `methods/README.md` GreedySeg 행 추가 + 실행 가이드 §, `methods/
greedyseg/` 신규 디렉토리, `src/hi_em/baselines/{_device,_seg_utils,
greedyseg_delay2}.py` 신규. 기존 TextTiling-streaming runner 도 `_seg_utils`
공통 모듈 사용으로 refactor. SEM 계승 원칙 적용 대상 아님 (외부 baseline).

## 2026-05-21 — GraphSeg-inspired bounded-window 신설 (graph-based AUXILIARY)

**맥락**: GraphSeg (Glavaš et al. 2016, SemEval; 구현 = Dobatymo/graphseg-python)
의 sentence similarity (IC × GloVe + Hungarian) + Bron-Kerbosch maximal clique
+ sequential merge 3-phase 를 online 변형으로 추가. 원본은 전체 대화 graph
위에서 1회 수행하는 *전역 구조화*.

**codex:rescue 위임 결과 (한국어, 2026-05-20)**:
- 옵션 A (**bounded-lookahead window=d**) 채택. window 안에서만 graph 구축 +
  clique + merge 재계산.
- **codex 2026-05-21 algorithm-integrity 검증**: 전역 graph 본질이 깨짐 →
  강한 `GraphSeg-online` 명명 **금지**. 정직 명명 = `GraphSeg-inspired
  bounded-window` (paper 본문), short `GraphSeg-window-d` (file/CLI/표 헤더만).
  **AUXILIARY only** — 원본 GraphSeg paper 결과와 같은 표 등재 금지.

**유지/양보** (codex 권고):
- 보존: sentence similarity 공식 (IC × GloVe + Hungarian), Bron-Kerbosch
  maximal clique, sequential merge 3-phase, content-word POS filter.
- 양보: full-dialogue graph (→ window=d), global clique structure
  (→ window-local), single-pass global merge (→ window 마다 재계산),
  backtracking (= 불가, boundary 비가역 lag-emission).

**결정**: `src/hi_em/baselines/graphseg_window.py` (`GraphSegWindowD`,
push/flush/state, lazy GloVe + NLTK brown IC table) + `methods/graphseg/
online_window.py` runner + `tests/test_graphseg_window.py` (9 ea, tiny
embedding fixture). 신규 의존: `networkx`, `scipy` (uv add). 외부 download:
GloVe 6B.300d (~1GB 압축해제, `benchmarks/glove/`, gitignored).

**산출 (3-benchmark full test set, 2026-05-21 device=cpu, M4 Pro)**:
`outputs/experiments/2026-05-21_graphseg_window_d/REPORT.md`

| dataset | n(dial/turn) | Pk | WD | F1 | Score | lat/turn mean | p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| tiage | 100/1564 | 0.493 | 0.516 | 0.252 | 0.374 | 0.85ms | 1.68ms |
| dialseg711 | 711/18639 | 0.449 | 0.482 | 0.366 | 0.450 | 1.15ms | 1.84ms |
| superseg | 1322/16006 | 0.539 | 0.542 | 0.164 | 0.312 | 0.62ms | 1.88ms |

cold-start = 13.66s (GloVe 400k vocab + brown IC table). Pk/WD/F1 = INDICATIVE
(원본 paper 점수와 데이터·metric·인터페이스·전역 graph 모두 차이). encoder-
light (no BERT) 라 latency 가 BERT-based GreedySeg-online-delay2 보다 1~2자릿수
빠름.

**미해결 / 가정**:
- 원본 paper 의 word frequency 출처 (Wikipedia) 대신 NLTK brown corpus 사용
  (실용적 선택, REPORT 한계 §에 명시).
- POS filter = NLTK averaged_perceptron_tagger.
- boundary lag-emission: 한 boundary 가 여러 window 평가에서 재출현 가능,
  본 구현은 *최초 발견 즉시 비가역 채택*.

**영향**: `methods/README.md` GraphSeg 행 추가 + 실행 가이드 §, `methods/
graphseg/` 신규 디렉토리, `benchmarks/glove/` gitignored. SEM 계승 원칙 적용
대상 아님 (외부 baseline).

## 2026-05-21 — `sem_core_v412_exp.py` (per-topic δ*_k) 코드 삭제

2026-05-19 검증으로 default 미승격(negative result, 무이득) 확정 후 보존만 하던 실험 코드. 사용자 정리 요청으로 `src/hi_em/sem_core_v412_exp.py` 및 `scripts/run_superdialseg_eval.py` 의 v4.1.2-exp 참조(import/seg_pred_v412exp/eval row) 일괄 제거. 결과·결정(2026-05-19 entry) 은 git history + 위 문서 기록으로 보존. 재오픈 필요 시 git revert 또는 history 에서 복원.

## 2026-05-20 — v4.2.2 신설: scene encoder swap (bge-base → aws-ai/dse-bert-base)

사용자 요청 "v4.1.1 의 RNN 을 `aws-ai/dse-bert-base` 로 대체한 v4.2.2"
재해석: v4.1.1 default 가 이미 RNN 미사용 (`eta_prev=1` → `_uses_rnn=False`
auto-skip) → 실질 변경점은 **scene vector encoder 교체** (`baai/bge-base-en-v1.5`
→ `aws-ai/dse-bert-base`). codex 분석 (option 1/2/3 비교) 결과 option 1 채택.

**SEM 계승 3-step 통과**: encoder 선택은 SEM2 algorithmic core 에 포함 안 됨
(SEM2 는 정규화 임베딩이면 무엇이든 수용). sCRP / Bayes / local-MAP /
prior-corrected B0 와 충돌 0 — δ_eff 수식 동일, scale 만 corpus·encoder 별
이동. retrieval § 위반 아님 (segmentation upstream).

**산출**:
- `src/hi_em/sem_core_v422.py` — thin subclass `HiEMSegmenterV422(HiEMSegmenterV411)`,
  `encoder_name` kwarg 로 expected (`aws-ai/dse-bert-base`) 미일치 시 warn.
- `context/methodology/v4.2.2.md` — 운영 절차 + δ\* 재calibration 필수 명시.

**필수 후속 (보고 전)**:
1. `.env`: `HIEM_EMBEDDING_MODEL=aws-ai/dse-bert-base` (Crts api 서빙 가능 여부
   첫 호출 확인; 실패시 `HIEM_EMBEDDING_BACKEND=local` fallback).
2. δ\* 재calibration: SuperDialseg/TIAGE/Dialseg711 train split, encoder=DSE-BERT.
   v4.1.1 default 0.5557 (bge+TIAGE-train) 은 DSE-BERT 환경 무효.
3. smoke 실험: train/dev 에서 v4.1.1(bge) vs v4.2.2(DSE-BERT) Score 비교.
   노이즈 밖 우위 없으면 본 doc "결과" 섹션 negative result + 본 log append.

**근거**: codex 권고 ("encoder 교체 라면 SEM 계승 3-step 통과 OK, 단 δ\* 재보정
필수"). 사용자 진행 의사 명시 (옵션 B: 곧장 v4.2.2 만들기).

**영향**: methodology 신규 doc, 코드 한 파일 추가, 실험 라인 1개 추가. 알고리즘
본체 무변경. v4.2.1 (`f0_min_starts=1`) 과 직교 — 동시 적용 가능.

## 2026-05-20 — v4.1.1.md 결과 표에 TIAGE-test row 추가

사용자 지시 — v4.2.x encoder ablation smoke 의 baseline 으로 측정된
v4.1.1 의 TIAGE-test 결과 (Score 0.468 / F1 0.410 / Pk 0.442 / WD 0.508)
를 `context/methodology/v4.1.1.md` 결과 표에 row 로 추가. ⚠ disclosure
한 줄 (calibration source = TIAGE-train 과 같은 corpus → in-domain 측정)
같이 명시. 다른 두 벤치 (superseg / dialseg711) 의 transfer 측정은 변경
없음.

**근거**: encoder ablation 비교에 TIAGE-test 가 필요했고 측정해보니 sanity
match (w_topic=1.0 결과와 일치). 측정값 자체는 valid 한 v4.1.1 성능 —
공식 doc 에 누락된 row 였을 뿐. 사용자 요청으로 정식 기록.

**영향**: v4.1.1.md 결과 표 1줄 추가 + 변경 이력 1줄. 알고리즘 / 코드 무관.

## 2026-05-20 — v4.2.3-exp 신설 → exp 보존 (default 승격 안 됨)

사용자 + codex 흐름: v4.2.2 smoke 의 mixed result (mpnet vs DSE-BERT
도메인별 우위) → "두 encoder 가 다른 dialogue 정보" 가설 → v4.2.3-exp
설계 (codex option E = calibrated energy combination).

**알고리즘**: v4.1.1 algorithm 무변경, scene vector 만 dual-channel
threshold-normalized energy `r = √(w_t·z_t² + w_f·z_f²)` 로 확장.
boundary ⇔ `r ≥ 1` (internal `δ*=1.0` trick 으로 V411 머신러리 그대로
재사용). topic 채널 = mpnet, flow 채널 = DSE-BERT. 각 채널 TIAGE-train
calibration HP 그대로.

**SEM 계승 3-step**: CONDITIONAL PASS (codex 2026-05-20) — framing 을
"SEM2 residual likelihood 의 calibrated two-feature reduction" 으로 유지.
Zacks 2007 EST 는 개념 인용만, 성능 근거로 사용 금지.

**산출**:
- `src/hi_em/sem_core_v423_exp.py` (HiEMSegmenterV423Exp(V411))
- `context/methodology/v4.2.3.md`
- `scripts/run_v423_smoke.py` (single trial, w=0.75)
- `scripts/run_v423_weight_sweep.py` (w_topic ∈ {1.0~0.5} 7-point)
- `outputs/experiments/2026-05-20_v423_smoke/REPORT.md`
- `outputs/experiments/2026-05-20_v423_weight_sweep/REPORT.md`

**결과 요약**:
- Single trial (w=0.75): TIAGE +2.29pp, Dialseg711 -2.31pp, SuperDialseg -3.01pp
- Sweep mean: w=1.0 (=v4.1.1) 가 best (0.5068). dual 어떤 weight 도 평균 못 이김.
- Per-dataset best: TIAGE w=0.5 (+2.98pp), Dialseg711 w=0.95 (+0.26pp), SuperDialseg w=1.0 (DSE 항상 해로움)
- 진단: corr(z_t, z_f) 가 dialseg711 에서 0.765 (가장 높음, only_topic 거의 0)
  → 두 encoder noise correlation 높을 때 dual 이 mpnet decision 을 희석

**결정**: v4.2.3-**exp** suffix 보존, default 미승격. v4.1.1 유지.
도메인 특화 사용 옵션 보존 (`HiEMSegmenterV423Exp(w_topic=0.5)` 자연 대화용).
adaptive weighting / joint calibration / 다른 flow encoder 등 v4.2.4 후보.

**근거**: dual channel 가설은 *부분 입증* (TIAGE +3pp), *부분 반증*
(정형 topic-shift 손해). single global weight 로는 v4.1.1 못 이김. codex
권고 "exp 로 시작, 성능 주장 smoke 전 금지" 그대로 지킴.

**영향**: 코드 4개 (segmenter / 2 script / methodology doc), 결과
REPORT 3개, decision-log 1줄. v4.1.1 default 무변경.

## 2026-05-21 — v4.2.4-exp 신설: DSE 가 δ_model (RNN slot) 자리 대체

사용자 + Claude 흐름: v4.2.2 (DSE 단독, mpnet 대체) / v4.2.3-exp (calibrated
dual-channel energy) 모두 single global setting 으로 v4.1.1 못 이김.
v4.2.4 는 **v4.1.1 식 구조 보존 + RNN slot 만 DSE 로 교체** 형태로 다른
각도 시도. codex 2026-05-18 P2 ("frozen-pretrained-f, C 천장측정 후 재검토")
의 직접 실현.

**알고리즘**:
- mpnet channel (δ_adj = a·δ_prev + (1-a)·δ_ctx): v4.1.1 default 그대로
  (m=2, ρ=0.7, a=0.5, δ*=0.5594)
- DSE channel: δ_model = a_dse·δ_prev_dse + (1-a_dse)·δ_ctx_dse,
  HP = v4.2.2 best (m_dse=2, ρ_dse=0.5, a_dse=0.0)
- Blend: δ_eff² = η·δ_adj² + (1-η)·δ_model² (v4.1.1 식 그대로)
- δ* = 0.5594 (mpnet, 그대로). σδ² 도 그대로.
- use_rnn=False 강제 (RNN compute 회피).

**SEM 계승 3-step**: PASS (conditional). SEM2 의 dynamics slot 자체는 존재.
v4.2.4 는 같은 slot 의 구현을 learned-RNN → frozen-pretrained-DSE 로
교체. **새 mechanism 도입 아님**. sCRP / Bayes / local-MAP / B0 무변경.
likelihood 식 무변경. framing 은 "RNN 학습 환경 부족 → frozen 대체"
유지. Zacks EST 는 개념 인용만, 성능 근거 금지.

**산출**:
- `src/hi_em/sem_core_v424_exp.py` (HiEMSegmenterV424Exp(V411),
  assign_pair(s_topic, s_dse))
- `context/methodology/v4.2.4.md`
- `scripts/run_v424_eta_sweep.py`
- `outputs/experiments/2026-05-21_v424_eta_sweep/REPORT.md`

**결과 요약 (η sweep)**:
| η    | TIAGE  | dialseg711 | superseg | mean   |
|------|--------|------------|----------|--------|
| 1.00 | 0.4675 | 0.5897     | 0.4631   | 0.5068 (=v4.1.1 sanity ✓)|
| 0.75 | 0.4724 | 0.6254     | 0.4275   | **0.5084** ⭐|
| 0.50 | 0.4892 | **0.6480** | 0.3806   | 0.5059 |
| 0.25 | 0.4860 | 0.6455     | 0.3368   | 0.4894 |
| 0.00 | 0.4488 | 0.6352     | 0.3036   | 0.4625 |

**핵심 발견**:
1. **Dialseg711 의 천장 돌파** — v4.1.1.md 명시 oracle-δ* (0.629, test 누설)
   상한을 v4.2.4 @η=0.5 (TIAGE-cfg, test 누설 없음) 가 **0.6480** 으로 돌파.
   "유사도 천장 0.590" 가설 약화.
2. **mean Score 첫 우위** — η=0.75 mean 0.5084 가 v4.1.1 의 0.5068 보다
   +0.0016 (미세하지만 v4.2.x 라인 첫 우위 사례).
3. **SuperDialseg 일관 회귀** — DSE 가 어떤 η 든 superseg 에 해로움.
   under-segmentation 의심 (DSE δ raw scale 작아 mixed δ_eff 작아짐).

**메커니즘 caveat**: η<1 의 효과는 (a) DSE dialogue-flow 정보 추가 +
(b) δ_eff scale 축소로 boundary 완화 두 효과 섞임. δ\* re-calibration
없이는 분리 불가.

**결정**: v4.2.4-**exp** suffix 유지, v4.1.1 default 유지. 다음 step:
1. `calibrate_v424_delta_star.py` — η 별 TIAGE-train δ\* re-calibration
2. SuperDialseg-train δ\* 별도 산출 (회귀 원인 분리)
3. multi-seed variance 측정
4. v4.2.3 / v4.2.4 cross-effect 실험

**근거**: 첫 mean Score 우위 + Dialseg711 의 oracle 천장 돌파는 codex P2
가설 ("frozen-pretrained-f 가 RNN 학습 환경 부족 보완") 강한 정량 증거.
다만 mechanism 분리 (DSE 정보 vs scale 부수효과) 미해결 + SuperDialseg
회귀 + single seed → exp 보존 합리.

**영향**: code 1 + script 1 + methodology doc 1 + REPORT 1 + decision-log 1줄.
v4.1.1 default 무변경.

## 2026-05-21 — v4.3.1-exp 신설: DialoGPT-small surprisal 이 δ_model 자리

사용자 + Claude 흐름: 외부 공격 "Hi-EM 의 prediction-by-production 은
진짜 다음 발화 예측이 아니지 않은가?" 에 대한 *방어 강화 카드*. v4.2.4
가 cosine-based DSE 로 δ_model 자리를 채운 패턴 동형으로, v4.3.1 은
**token-level autoregressive surprisal `−log P(u_t | u_{<t})`** 을
DialoGPT-small (Zhang+ 2020) 로 직접 산출하여 같은 slot 에 주입.

**알고리즘**:
- mpnet channel (δ_adj = a·δ_prev + (1-a)·δ_ctx): v4.1.1 default 그대로
  (m=2, ρ=0.7, a=0.5, δ*=0.5594)
- LM channel: δ_model(t) = mean-token NLL of u_t given u_{t-m_LM..t-1},
  m_LM=5, raw EOS-concat 포맷 (DialoGPT 학습 분포 보존), no normalization
- Blend: δ_eff² = η·δ_adj² + (1-η)·δ_model² (v4.1.1 식 그대로)
- δ* = 0.5594 (mpnet, raw NLL scale mismatch caveat). σδ² 그대로.
- use_rnn=False 강제.
- precompute 분리: `precompute_v431_nll.py` 가 LM forward, segmenter 는
  scalar nll_t 만 소비 (v4.2.4 의 DSE 임베딩 캐시 패턴).

**CSM (Xing & Carenini 2021; Zhou+ 2022) 와 구분**:
1. CSM 은 NSP head 를 *학습* → 그 score 가 boundary rule 자체.
2. v4.3.1 은 **frozen zero-shot** LM surprisal → SEM PE channel 로 흡수.
3. boundary 는 sCRP + local-MAP + B0 가 결정, cosine retrieval 미사용.

**SEM 계승 3-step**: PASS (conditional). v4.2.4 와 동일 논리 (SEM2
LinearEvent dynamics slot 의 frozen 대체). 새 mechanism 도입 아님.
"prediction-by-production 의 생성적 구현" 같은 강한 주장 금지, 단지
*surprisal 이 boundary 정보를 담는다는 경험 가설* 만 검증한다.

**산출**:
- `src/hi_em/sem_core_v431_exp.py` (HiEMSegmenterV431Exp(V411),
  assign_pair(s_topic, nll_t))
- `context/methodology/v4.3.1.md`
- `scripts/precompute_v431_nll.py` (DialoGPT-small NLL 캐시)
- `scripts/run_v431_smoke.py` (η sweep + REPORT)

**핵심 caveat (sweep 전 사전 명시)**:
- Raw NLL scale (~2~8 nats/token) >> δ_adj scale (~0.4~0.6) → η<1 에서
  δ_eff 가 surprisal 항에 급팽창 → 과분절 위험. v4.2.4 (DSE: 과합병)
  와 정반대 방향.
- 1차 sweep η ∈ {1.0, 0.99, 0.95, 0.9, 0.75, 0.5} (η→1 쪽 조밀).
- δ\* 재calibration 은 후속 (`calibrate_v431_*`) 보류.

**결정**: v4.3.1-**exp** suffix 유지. sweep 결과 본 후 promotion 검토.
v4.1.1 default 무변경.

**근거**: 외부 공격 (prediction-by-production 의 진정성) 에 대한
방어카드 + SEM2 frozen-f 슬롯의 또 다른 옵션 (LM-surprisal vs DSE-cosine)
. 두 방향 (cosine 응집성 vs token surprisal) 의 정량 비교로 SEM 의
dynamics slot 에 무엇이 더 informative 한지 측정.

**영향**: code 1 + scripts 2 + methodology doc 1 + decision-log 1.
v4.1.1 default 무변경.

## 2026-05-21 — v4.3.2-exp 신설: frozen ST5 + learned NextEmbedHead 가 δ_model 자리

사용자 + Claude 흐름: v4.3.1 (DialoGPT NLL) 의 자매판. 같은 외부 공격
("Hi-EM 의 prediction-by-production 의 진정성?") 에 대해 *embedding-space
에서 실제로 next-utterance vector 를 regression 으로 예측* 하는 답안.
PE 옵션 로드맵 (6)번 항목 — pretrained Sentence-T5 + 작은 학습 head.

**알고리즘**:
- mpnet channel (δ_adj = a·δ_prev + (1-a)·δ_ctx): v4.1.1 default 그대로
  (m=2, ρ=0.7, a=0.5, δ*=0.5594)
- ST5+head channel: \\hat{s}_t = head(s_st5[t-m_LM..t-1]),
  δ_model = 1 − cos(\\hat{s}_t, s_st5[t]),
  m_LM=5, left-zero-pad context, NextEmbedHeadMLP(3840 → 1024 GELU → 768 → L2-norm)
- Blend: δ_eff² = η·δ_adj² + (1-η)·δ_model² (v4.1.1 식 그대로)
- δ* = 0.5594 (mpnet 그대로; ST5 cosine-distance scale ≈ 0.3~0.7 로 mpnet 과 자연 양립)
- use_rnn=False 강제

**학습 (Colab)**:
- DailyDialog (HF `daily_dialog`, 11k train + 1k valid + 1k test)
- Frozen ST5 embedding 1회 캐시 → head training (no encoder grad)
- Loss = mean(1 − cos(\\hat{s}, s_t))
- AdamW lr=1e-3, wd=1e-4, linear warmup 5%, batch 256, epoch 10
- ckpt → `outputs/runs/_misc/next_embed_head_<tag>.pt` (사용자가 Colab→로컬 복사)

**CSM (Xing+ 2021; Zhou+ 2022) 와 구분**:
1. CSM = discriminative NSP ranking (positive/negative, hinge loss).
2. v4.3.2 = **continuous next-embedding regression** (cosine loss).
   negative sampling 안 함, target = 실제 다음 발화의 ST5 vector.
3. boundary 는 sCRP + local-MAP 가 결정 (CSM 처럼 score 직접 사용 안 함).

**SEM 계승 3-step**: PASS (conditional). SEM2 ``LinearEvent`` 의
learned dynamics 와 *가장 직접 대응*. 차이: per-event learning →
corpus-wide pretrained. v4.2.4 (frozen DSE) / v4.3.1 (frozen DialoGPT)
와 같은 방향성 (frozen pretrained-f). sCRP / Bayes / local-MAP / B0
무변경. δ scale 도 cosine-distance 라 mpnet 과 자연 양립.

**산출**:
- `src/hi_em/sem_core_v432_exp.py` (HiEMSegmenterV432Exp(V411),
  assign_pair(s_topic, delta_model_t))
- `src/hi_em/next_embed_head.py` (NextEmbedHeadMLP + pack_causal_window)
- `scripts/train_next_embed_head.py` (Colab 권장; HF `datasets` lazy import)
- `scripts/precompute_v432_delta.py` (로컬: ST5 encode + head inference → δ 캐시)
- `scripts/run_v432_smoke.py` (η sweep + REPORT)
- `context/methodology/v4.3.2.md`

**v4.3.1 와의 차이 (PE 옵션 로드맵상 (5) vs (6))**:
| | (5) v4.3.1 (DialoGPT) | (6) v4.3.2 (ST5+head) |
|---|---|---|
| δ_model 정의 | per-token NLL | 1 − cos(pred, target) |
| Range (raw) | 2~8 nats | 0.3~0.7 |
| mpnet scale 일치 | ✗ (큰 mismatch) | ✓ (자연 양립) |
| 학습 0회? | ✓ | ✗ (head 학습) |
| 외부 corpus | 없음 | DailyDialog |

→ scale 일관성은 v4.3.2 가 우수. 학습 의존성 / OOD generalization 위험은
v4.3.2 가 추가. 두 방향의 정량 비교는 두 sweep REPORT 비교로.

**결정**: v4.3.2-**exp** suffix 유지. ckpt 받아 sweep 후 promotion 검토.
v4.1.1 default 무변경.

**Colab/Local 분리 준수 (2026-05-21 보강)**: 사용자 결정으로 학습
데이터 source 를 HF `datasets.load_dataset('daily_dialog')` → **원본
`ijcnlp_dailydialog.zip` 자동 탐색 + 재귀 추출** (colab_csm_train.ipynb
[2] 셀과 동일 패턴) 으로 변경. HF datasets 의존 자체를 제거 →
pyproject.toml 무변경, train script 도 외부 dep 추가 0.

**codex 학습 setup 검토 반영 (2026-05-21)**: 1) scheduler 첫-step lr=0
버그 수정 ((step+1)/warmup), 2) target L2-normalize guard, 3)
eval_diagnostics 추가 (mean_baseline_loss / gain_vs_mean / delta_std /
pred_norm_mean) — codex 1위 위험 "regression-to-mean / output collapse"
의 epoch 별 자동 진단 + 경고, 4) optional batch-InfoNCE (`--nce-weight`,
default 0) — codex 권고대로 monitoring 우선, collapse 확인 시 조건부.
HP (lr/batch/epochs/wd/architecture) 자체는 적정 판정으로 변경 없음.

**Colab 흐름 별도 notebook 분리 (2026-05-21)**: 사용자 결정 — Colab
에서는 Hi-EM repo clone 등 어떤 사전 셋업도 없이 **완전 독립** 실행.
새 self-contained notebook `colab_train_next_embed_head.ipynb` (repo
root) 추가: 11 셀 (env + dep / zip 업로드 / zip 자동 추출 / parse /
NextEmbedHeadMLP inline / HP / ST5 encode / loaders+diag / train loop /
ckpt download). colab_csm_train.ipynb 의 IS_COLAB 분기 + zip 재귀 추출
패턴 그대로. Local CLI 흐름 (`scripts/train_next_embed_head.py`) 은
그대로 유지 — 두 진입점 동등.

**근거**: v4.3.1 / v4.3.2 두 사매판으로 'next-utterance prediction' 의
*확률* vs *벡터* 두 정의를 동시 측정. SEM2 learned dynamics 슬롯에
가장 직접적인 구현 ([[v4.2.4]] 의 frozen-DSE 보다 SEM2 LinearEvent 에
더 가까움).

**영향**: code 2 + scripts 3 + methodology doc 1 + decision-log 1.
v4.1.1 default 무변경. pyproject.toml 무변경.

## 2026-05-21 — v4.1.3 신설: graded boundary score + re-entry tracking (output-only)

사용자 요청 — paper 의 "graded boundary score" (Ben-Yakov & Henson 2018
hippocampal graded profile 동형) + "non-linear topic reentry" (schema
reinstatement) contribution bullet 정량 근거 확보.

**알고리즘**: v4.1.1 무변경. **output instrumentation 만 추가**.

**노출 API**:
- `seg.last_graded_score` = `δ_eff / δ*` per turn (boundary strength scalar)
- `seg.last_is_reentry` = boundary INTO previously-visited topic
- `seg.history()` / `reentry_turns()` / `graded_scores()` / `boundary_strength()`

**SEM 계승 3-step**: PASS trivial (algorithm 무변경, side-effect attribute 만).

**산출**:
- `src/hi_em/sem_core_v413.py` (HiEMSegmenterV413(V411))
- `context/methodology/v4.1.3.md`
- `scripts/run_v413_demo.py`
- `outputs/experiments/2026-05-21_v413_demo/REPORT.md`

**핵심 발견 (v4.1.1 의 봉인된 re-entry)**:

| dataset | f0_min_starts | bnd | reentries | rate |
|---|---:|---:|---:|---:|
| tiage | 2 (v4.1.1 default) | 431 | 0 | 0.0% |
| dialseg711 | 2 (v4.1.1 default) | 5202 | 0 | 0.0% |
| tiage | 1 (v4.2.1) | 696 | 603 | **86.6%** |
| dialseg711 | 1 (v4.2.1) | 8367 | 7854 | **93.9%** |

v4.1.1 의 `f0_min_starts=2` default 가 re-entry mechanism 을 사실상 봉인
(f0 centroid 가 ≥2 episode start 모여야 활성화되는데, 매 boundary 가 새
topic_id 생성하니 영원히 못 채움). [[v4.2.1]] 의 `f0_min_starts=1` 로 풀면
schema reinstatement 정상 작동.

**시사**: v4.1.3 + v4.2.1 조합이 paper 의 "non-linear topic reentry"
정량 근거. v4.2.1 의 marginal F1 효과 (codex 2026-05-19 예측) 와 함께
보면 — re-entry 가 **양적으론 dramatic 변화 (0% → 87-94%) 지만 정량
지표상 marginal**. 즉 segmenter 의 topic 구조가 질적으로 달라지지만
boundary 위치 자체는 비슷.

**Boundary strength bands** (graded_score 기반):
- `< 0.7` very weak (downstream 보류 권고)
- `0.7-1.0` weak (repeat 우세)
- `1.0-1.3` normal
- `≥ 1.3` strong (즉시 commit)

SeCom 등 downstream uncertainty-aware consumer 가 직접 활용 가능.

**영향**: code 1 + script 1 + methodology doc 1 + REPORT 1 + decision-log
1줄. v4.1.1 default 무변경. F1/Pk/WD 무변경 (algorithm 무변경).

## 2026-05-21 — v4.1.3 default 변경: f0_min_starts=1 흡수 (v4.2.1 통합)

원래 v4.1.3 = pure output instrumentation (V411 default f0_min_starts=2
상속). 검증 결과 v4.1.1 default 의 re-entry 가 0% (segmenter 가 매번 새
topic_id 생성 → f0 centroid 가 영원히 2 episode-start 못 채움 → 봉인).
"non-linear topic reentry" paper claim 데이터 0 → v4.1.3 자체 가치 약함.

**사용자 결정 (2026-05-21)**: v4.1.3 default 를 `f0_min_starts=1` 로 바꿔
[[v4.2.1]] 의 SEM2-faithful 변경을 v4.1.3 안으로 흡수. 관찰 도구 +
관찰 대상 동시 활성화.

**결과**: TIAGE 86.6% / Dialseg711 93.9% re-entries — paper 의
"non-linear topic reentry" contribution 정량 근거 default 로 즉시 확보.
explicit `f0_min_starts=2` 전달 시 V411 default 복원 (옛 ablation 참조용).

**[[v4.2.1]] 위치 변화**: 별도 클래스로 보존, default 변경의 명시적
이름 carrier. v4.1.3 가 그 default 를 흡수했으므로 *기능상 중복*. 다만
파일·class·이력 모두 그대로 유지 — 향후 ablation / decision-log 참조용.

**영향**: src/hi_em/sem_core_v413.py 의 `__init__` signature 변경
(`f0_min_starts=1` 명시), v4.1.3.md 결과 / 추천 사용 패턴 갱신, demo
재실행 (TIAGE+Dialseg711 default 표 갱신). 다른 코드 / v4.2.x 라인 무관.

## 2026-05-21 — v4.2.4-exp negative result 확정 (calibration artifact)

η sweep (2026-05-21_v424_eta_sweep) 의 핵심 우위 — η=0.75 mean Score
0.5084 (+0.0016 vs v4.1.1) 와 Dialseg711 η=0.5 의 0.6480 ("oracle 천장
돌파") — 가 δ\* 재calibration 후 모두 사라짐 (`2026-05-21_v424_calib`).

**Calibration 결과** (F1-best δ\*, tiage-train + superseg-validation):
- tiage-train η=0.5: δ\*=0.5260 (mpnet 단독의 0.5594 보다 -0.033)
- tiage-train η=0.75: δ\*=0.5209
- superseg-val η=0.5: δ\*=0.3801 (mpnet 보다 매우 낮음)
- superseg-val η=0.75: δ\*=0.4529

**Re-cal 후 mean Score**:
- v4.1.1: 0.5068
- v4.2.4 @η=0.5 re-cal: 0.4970 (-0.0098)
- v4.2.4 @η=0.75 re-cal: **0.4900 (-0.0168)**

**Dialseg711 의 0.6480 → 0.5690 (-0.079)**: prior δ\*=0.5594 가 mixed
signal 에서 우연히 Score-best 근방. F1-best 로 다시 잡으니 boundary
trigger 잦아져 Pk/WD 악화 → Score 떨어짐. v4.1.1.md 가 superseg val-cal
에서 미리 경고한 "F1-best 가 과분절 유발" 패턴 정확 재현.

**결정**: v4.2.4-exp **promotion 후보 탈락**, exp suffix 유지, v4.1.1
default 유지. 코드 보존 (Score-best δ\* / dataset-aware / CSM 학습 후
CSM-as-encoder v4.2.5 등 후속 연구 출발점).

**Important lesson**: F1-best δ\* ≠ Score-best δ\*. Train calibration
정책 (F1-best) 이 metric (Score) 과 misaligned 한 점은 v4.x 라인 전체에
적용되는 caveat. 후속 변형은 Score-best δ\* 직접 sweep 옵션 검토.

**산출**: scripts/calibrate_v424_delta_star.py, outputs/experiments/
2026-05-21_v424_calib/REPORT.md, v4.2.4.md "판정" 섹션 갱신. 또한
부산물로 cache: sds_emb_tiage_train_*.pkl (×2 encoder),
sds_emb_superseg_validation_*.pkl (×2 encoder) — 영구 보존, 미래 train/val
calibration 빠르게 가능.

## 2026-05-21 — v4.1.3 sub-type 분리 + per-band precision 정량 확보

`outputs/experiments/2026-05-21_v413_reentry_analysis` 분석 결과:

**Per-band precision (graded_score band 별 boundary prediction precision)**:

| dataset | very_weak | weak | normal | strong |
|---|---:|---:|---:|---:|
| TIAGE | 0.000 | 0.238 | 0.345 | **0.520** |
| Dialseg711 | 0.000 | 0.129 | 0.383 | **0.800** |

→ graded_score 가 진짜 calibrated boundary signal. 특히 strong band
(≥1.3) 의 precision 0.52~0.80 으로 paper 의 "score > 1.3 → 즉시 commit"
downstream 정책 정량 근거.

**Re-entry sub-type 분리** (case study 보고 두 mechanism 섞임 발견):
- `same_label_restart`: V411 `_is_restart` 경로, topic_id 유지, 같은 topic 의
  새 episode
- `cross_topic_reentry`: 옛 topic 으로 복귀, 진짜 non-linear

| dataset | total re | same_label (n, prec) | cross_topic (n, prec) |
|---|---:|---|---|
| TIAGE | 603 | 473 (78%), 0.309 | 130 (22%), 0.315 |
| Dialseg711 | 7854 | 6586 (84%), 0.319 | 1268 (16%), 0.221 |

**해석**:
- 78-84% 가 same_label_restart (within-topic resumption) — paper 의
  "non-linear" claim 과는 의미 다름.
- Cross_topic_reentry 가 22% / 16% — *진짜* non-linear, paper 가 강조해야
  할 type.
- Dialseg711 의 cross_topic precision (0.221) 이 same_label (0.319) 보다
  낮음 — 옛 topic 으로 복귀 결정에서 f0 의 discrimination power 낮은 듯.

**코드 변경** (`src/hi_em/sem_core_v413.py`):
- `last_is_same_label_restart`, `last_is_cross_topic_reentry` attribute 추가
- `same_label_restart_turns()`, `cross_topic_reentry_turns()` 메서드 추가
- `is_reentry` = OR (backward compat 유지)
- history dict 에 두 sub-flag 추가

**Paper framing 정정**:
- 이전: "TIAGE 87%, Dialseg711 94% boundaries are re-entries" (광의)
- 수정: "Cross-topic re-entries = **22% (TIAGE) / 16% (Dialseg711) of
  boundaries**, of which 32% / 22% match human-annotated boundaries."

**미해결**: cross_topic_reentry 의 precision 0.22 가 낮은 이유 — f0 centroid
의 quality 문제일 수 있음. 후속: f0_min_starts=3,4 등 다른 값에서 sweep,
또는 f0 centroid 의 outlier 제거 변형 검토.

**영향**: src/hi_em/sem_core_v413.py + scripts/analyze_v413_reentry.py +
v4.1.3.md "결과 / 산출 / 변경이력" 섹션 + outputs/experiments/
2026-05-21_v413_reentry_analysis/REPORT.md. v4.1.1 default 무관, F1/Pk/WD
무변경.

## 2026-05-21 — v4.1.3 default revert: f0_min_starts=2 (Score 보존)

같은 날 revision 1 (`f0_min_starts=1` 흡수, v4.2.1 통합) 후 직접 측정
(`/tmp/test_f0_effect`):

| dataset | f0=2 baseline | f0=1 (revision 1) | Δ |
|---|---:|---:|---:|
| TIAGE | 0.4675 | 0.3987 | **-6.9pp** |
| Dialseg711 | 0.5897 | 0.3940 | **-19.6pp** |
| SuperDialseg | 0.4631 | 0.3604 | **-10.3pp** |

**원인 진단**: `is_boundary` (label-change OR same-label-restart) 와 metric 의
boundary (label-change 만) 가 다른 정의. f0=1 활성 시 segmenter 가 옛
topic_id 재사용 → label-change count 감소 (TIAGE 431→223) → recall 폭락 →
F1/Score 폭락. v4.2.1 docstring 의 codex "marginal effect" 예측이 실제론
large drop. 측정 안 했을 뿐.

**SEM2 충실 vs Score trade-off**: 사용자 질문 "SEM2 가 가장 예측오차 적은
topic 으로 재진입하는 게 맞나?" → 답: 맞음. SEM2 의 `update_f0`/
`f0_is_trained` 가 첫 등장부터 활성. v4.1.1 의 `f0_min_starts=2` 는
Hi-EM-only 이탈 (보수). 즉 SEM2 원리는 v4.2.1/v4.1.3-analysis mode 가
충실. 다만 대화 도메인의 f0 단일 sample noise 가 큰 게 Score 폭락의
원인 — SEM2 원리 자체의 문제가 아니라 dialogue domain 의 f0 quality 문제.

**결정 (사용자)**: v4.1.3 default 를 `f0_min_starts=2` 로 되돌림.
- Performance mode (default): Score = v4.1.1 그대로
- Analysis mode (explicit `f0_min_starts=1`): re-entry 정량 분석 가능,
  Score 폭락 감수

**Paper framing 수정**: re-entry 는 "v4.1.3 가 expose 하는 sCRP mechanism
의 behavioral readout" 으로 명시. **default 에선 mechanism dormant**,
analysis mode 에서만 활성. "94% boundaries are re-entries" 표현 X.

**산출**: src/hi_em/sem_core_v413.py 의 `__init__(f0_min_starts: int = 2)`
복원, v4.1.3.md 의 "한 줄"/"추천 사용 패턴"/"변경 이력" 갱신.

## 2026-05-21 — v4.1.3 final: re-entry 기능 전체 제거 (graded_score 만)

같은 날 revision 2 (default revert to f0=2) 후 사용자 결정 "토픽 회귀
하지마" → v4.1.3 에서 **re-entry 관련 전체 제거**.

**제거된 항목**:
- `last_is_reentry`, `last_is_same_label_restart`, `last_is_cross_topic_reentry` attribute
- `reentry_turns()`, `same_label_restart_turns()`, `cross_topic_reentry_turns()` method
- `is_reentry`, `is_same_label_restart`, `is_cross_topic_reentry` history fields
- `analyze_v413_reentry.py` (script 자체는 보존 — historical record)

**유지 항목**:
- `last_delta_eff`, `last_graded_score`, `last_is_boundary` attribute
- `history()`, `graded_scores()`, `boundary_strength()` method
- Ben-Yakov & Henson 2018 graded profile 매핑
- per-band precision (TIAGE strong 0.52 / Dialseg711 strong 0.80)

**Paper contribution 정리**:
- ✅ "graded boundary score (Ben-Yakov & Henson 2018 mapping)" — v4.1.3
- ❌ "non-linear topic reentry" — paper 에서 제외

**v4.2.1 위치**: ablation 코드로만 보존 (HiEMSegmenterV421 클래스 유지).
실험·default·paper 어느 쪽에도 사용 안 함. 향후 재검토 필요 시 참조.

**Score 영향**: 0 (algorithm 무변경, output 만 추가). v4.1.1 그대로
0.4675/0.5897/0.4631 (test 평균 0.5068).

**산출**: src/hi_em/sem_core_v413.py 전면 재작성, scripts/run_v413_demo.py
재작성 (per-band precision + boundary strength 만), context/methodology/
v4.1.3.md 전면 갱신 (re-entry 섹션 제거, "제거된 기능" 섹션 추가).

## 2026-05-21 — v4.3.2-exp: CLEAR NEGATIVE (single η), no promotion

[[v4.3.2]] (frozen Sentence-T5 + DailyDialog-trained NextEmbedHead) 의
Colab 학습 + 3-dataset calibrated z-blend sweep + boundary signal 진단
종합 결과: **정직 비교 (single η, no test leak) 에서 clear negative**.
exp suffix 영구 유지.

**정직 비교 (single η = mean-best, 모든 dataset 동일 적용)**:

| | best single η | mean Score | vs v4.1.1 |
|---|---:|---:|---:|
| v4.1.1 | — | 0.5068 | — |
| v4.2.4 (raw, frozen DSE) | 0.75 | 0.5084 | +0.16pp (noise 의심) |
| **v4.3.2 (calibrated, TX head)** | **0.75** | **0.5023** | **−0.45pp** |

→ v4.3.2 는 어떤 single η 로도 v4.1.1 못 이김.

**Per-dataset best (⚠ test leak, ablation 분석용만 — promotion 근거 X)**:
- TIAGE η=1: 0.4675 (head 무용)
- Dialseg711 η=0.75: 0.5997 (+1pp, v4.2.4 의 그림자)
- SuperDialseg η=1: 0.4631 (head 무용)

⚠ 사용자 정직성 지적 (2026-05-21 후속): "per-dataset best η 는 test
leak". 그 전 entry 의 "mixed result (+1pp)" 표현은 ablation 정보로 명시,
판정은 single-η 기준 clear negative 로 정정.

**학습**: 6 ckpt 비교 (MLP / Transformer × NCE 0/0.03/0.05). TX head +
NCE=0 best (valid loss 0.1323, ACCEPTABLE). Transformer 가 MLP 대비
명확 우수 (pad-mask + pos-emb + self-attn). NCE auxiliary 는 모든 조합
에서 trade-off 나쁨.

**Segmentation sweep (3 dataset)**:
1. Raw blend (TIAGE만): η<1 에서 F1=0 폭락 (δ_model scale 0.171 <<
   δ_adj 0.5 → 과합병).
2. Calibrated z-blend (사용자 처방, v4.2.3 패턴 `z = δ/δ*`, 3 dataset):
   single η=0.75 (mean best) → mean 0.5023 < v4.1.1 0.5068 (**-0.45pp**).
3. Per-dataset best (⚠ ablation only, test leak): Dialseg711 만 +1pp
   약신호, TIAGE/SuperDialseg head 무용.

**Boundary signal 진단 (사용자 명령)**: TIAGE test 의 boundary-after
turn (n=315) vs non-boundary (n=1149) 의 δ_model 분포 비교:

```
boundary:    0.1799 ± 0.0293
non:         0.1692 ± 0.0312
diff:        +0.0107  (방향성 맞음)
t-test:      t=5.46, p=5.5e-08    (통계 유의)
Cohen's d:   +0.348               (small-to-medium, 분류로는 약함)
diff/σ_nb:   0.343 σ              (두 분포 ~86% 겹침)
```

→ head 가 boundary 정보를 *조금* 학습했지만 mpnet δ_adj 보다 *훨씬*
약한 신호. blend 가 mpnet 신호 dilute → 회귀.

**진단**:
- 학습 자체 ✓ (ACCEPTABLE), Architecture (TX) ✓, Calibrated z-blend ✓
- 그러나 head 의 boundary signal 본질적으로 약함 → segmentation 으로
  안 옮겨짐 = **task misaligned** (codex/사용자 1차 우려 적중).
- "Regression-to-mean" 정량 증거: δ_std plateau 0.025-0.036, head 의
  output 분포가 학습 corpus 평균 방향으로 수렴.

**산출 (negative)**:
- code 2 변형: `sem_core_v432_exp.py` 에 `calibrated` 옵션 (z-blend +
  internal δ\*=1), `next_embed_head.py` 에 `NextEmbedHeadTransformer` +
  `make_head` factory
- script: `run_v432_smoke.py --calibrated --delta-star-model auto`
- methodology v4.3.2.md "결과 / 판정" 섹션 채움 (negative result 정직
  기록)

**학술적 가치 (정직 보고)**:
1. DailyDialog → TIAGE domain generalization 실패의 정량 데이터
2. Regression-to-mean 의 직접 증거 (δ_std plateau, boundary signal
   weak)
3. Calibrated z-blend (v4.2.3 패턴) 의 safety 재확인 (raw 폭락 회피)
4. Continuous regression (v4.3.2) vs discriminative ranking (CSM) 비교
   에서 *부정적* 데이터 — CSM 류가 보조 channel 로 더 적합할 가능성
   시사 (단 직접 비교 별도)

**결정**: v4.3.2-exp 영구 보존, v4.1.1 default 무변경. 추후 GRU
head / TIAGE-train 직접 학습 / discriminative objective 등 후속
가능하나 본 결과로는 우선순위 낮음. [[v4.3.1]] (DialoGPT NLL) sweep
은 별도 (사용자 GPU 시간 결정).

## 2026-05-21 — v4.2.5-exp 신설: CSM (finetuned BERT-base) 이 δ_model 자리

사용자 + Claude 흐름: 사용자가 Colab 에서 CSM (Coherence Model, Xing &
Carenini 2021; lxing532/Dialogue-Topic-Segmenter) 학습 완료 (`cpt_277000.pth`,
state_dict). PE 옵션 로드맵의 (4) 번 항목 — finetuned BERT-base
discriminative coherence head.

**동기**: [[v4.3.2]] (continuous regression) 의 clear negative 후,
**NSP-supervised discriminative ranking 학습** 이 regression-to-mean
한계를 회피하는지 직접 비교. 같은 DailyDialog corpus, 다른 objective.

**알고리즘**:
- mpnet channel (δ_adj = a·δ_prev + (1-a)·δ_ctx): v4.1.1 default 그대로
- CSM channel: δ_model(t) = 1 − softmax(decoder([CLS] u_{t-1} [SEP] u_t [SEP]))[0]
  = incoherence probability
- coherence_decoder = `Sequential(Linear(768,768), ReLU, Dropout(0.1), Linear(768,2))`
- BERT backbone = bert-base-uncased (train_csm_hf.py default)
- Blend: raw (v4.1.1 식) 또는 calibrated z-blend (v4.2.3 패턴, `--calibrated`)
- use_rnn=False 강제

**v4.3.2 와 직접 비교 (예정)**:
| | v4.3.2 (TX head) | v4.2.5 (CSM) |
|---|---|---|
| 학습 목적 | continuous regression (cosine) | discriminative ranking (marginal hinge) |
| Output | similarity | probability |
| Regression-to-mean | 큼 (clear negative 확정) | 원리적으로 낮음 (TBD) |

**SEM 계승 3-step**: PASS (conditional). v4.3.2 와 같은 frozen-encoder
+ learned-head 패턴. dynamics slot 채우는 frozen-f. sCRP/Bayes/B0 무변경.

**산출**:
- `src/hi_em/sem_core_v425_exp.py` (HiEMSegmenterV425Exp(V411),
  assign_pair(s_topic, delta_model_t), calibrated 옵션)
- `scripts/precompute_v425_delta.py` (CSM forward, 3 dataset δ cache)
- `scripts/run_v425_smoke.py` (η sweep, raw/calibrated, REPORT)
- `external/Dialogue-Topic-Segmenter/` (clone, gitignored 추가)
- `context/methodology/v4.2.5.md`
- `.gitignore` external/ 추가

**진행 중**: precompute background (CPU 30분 추정). sweep 끝나면
v4.2.5.md "결과 / 판정" 채우고 decision-log 정리. 핵심 비교는
v4.2.5 (CSM, discriminative) vs v4.3.2 (TX, regression) 의 same-blend
mean Score.

**결정**: v4.2.5-**exp** suffix 유지, sweep 결과 본 후 promotion 검토.
v4.1.1 default 무변경. external/ tree 는 gitignored (vendored 안 함).

---

### 2026-05-21 (저녁) — v4.2.5 raw sweep 결과 + 사용자 중단

**진행**: precompute (CPU 9h+) 가 tiage / dialseg711 까지 완료, superseg
처리 중에 사용자 중단 요청 ("v4.2.5 실행 멈추고 싹다 저장하자").

**결과 (raw blend, single-η mean-best, 2 dataset)**:

| | best η | mean Score | vs v4.1.1 |
|---|---:|---:|---:|
| v4.1.1 (η=1) | — | 0.5286 | — |
| v4.2.5 raw (CSM) | 1.00 | 0.5286 | **−0.0000** |

→ **NEGATIVE** (mean-best 기준). η<1 → mean −8.3pp 폭락.

**데이터셋별 (test-leak, 참고)**:
- tiage: η=0 (CSM only) **+2.0pp** (F1 +8.5pp, Pk/WD 무변)
- dialseg711: η=1 (CSM off) best, CSM 섞으면 **−18.6pp 붕괴**

**v4.3.2 와 직접 비교**:
| | v4.3.2 | v4.2.5 |
|---|---:|---:|
| single-η mean-best | −0.45pp | −0.00pp |
| tiage best | 무유의 | **+2.0pp** |

→ "discriminative ranking 이 regression-to-mean 회피" 가설은 **부분
지지** (in-domain 데이터셋 한정). 도메인 shift 에는 v4.3.2 보다 *더*
취약. 두 head 모두 mean-best 기준 net-neutral/negative.

**원인 (잠정)**:
1. raw scale mismatch (δ_csm ≈ binary {0,1}, mpnet ≈ continuous) — CSM
   spike 가 δ_eff² dominate. calibrated z-blend 미검증.
2. dialseg711 (Wikipedia-style) 이 DailyDialog NSP 학습 corpus 와 매우
   멀음.

**미수행 (사용자 중단)**:
- superseg precompute (CPU 비용)
- calibrated z-blend sweep (raw scale 보정 효과 미확인)
- v4.3.2 와 같은 sample/η 에서 head-only 차이 분리

**보존 자산** (회수 가능):
- `outputs/runs/_misc/sds_v425delta_tiage_test_csm_bert_base.pkl`
- `outputs/runs/_misc/sds_v425delta_dialseg711_test_csm_bert_base.pkl`
- `outputs/runs/_misc/cpt_277000.pth` (CSM ckpt)
- `src/hi_em/sem_core_v425_exp.py`, `scripts/precompute_v425_delta.py`,
  `scripts/run_v425_smoke.py`, `external/Dialogue-Topic-Segmenter/`

**결정**: v4.2.5-exp **종결 (net-neutral)**. promotion 안 함. v4.1.1
default 무변경. 차후 calibrated sweep + superseg 완성 + v4.3.2 와
head-only ablation 시 재오픈 가능.

**산출**: `outputs/experiments/2026-05-21_v425_csm_2ds_raw/REPORT.md`
(해석/판정 채워짐), `context/methodology/v4.2.5.md` (결과/판정 채워짐).

## 2026-05-22 — v4.1.3 dead-code audit → Hi-OnTop (reduced form) main 모델 채택

**맥락**: v4.1.3 (`HiEMSegmenterV413(HiEMSegmenterV411)`) 가 paper 의 main segmenter 인데,
SeCom-swap latency profiling 중 `assign()` 시간의 77% 가 *안 쓰는* per-topic EventRNN
모듈 생성에 쓰임을 발견 (η=1 default → RNN 비활성). lazy-init 으로 수정 후 모든 HP 를
출력 변화로 전수 audit.

**audit 결과 (실증)**: v4.1.3 의 segmentation 출력에 영향을 주는 HP 는
`delta_star` / `ctx_window` / `ctx_decay` / `ctx_blend_a` 4개뿐 (+ `beta` 극단값만 marginal).
나머지는 default/canonical setting 에서 dead — 출력 byte-identical:
- EventRNN 전체 (`eta_prev=1` → δ_model weight 0)
- f0 / restart / re-entry (`f0_min_starts≥2` circular deadlock 으로 영구 봉인)
- SEM2 variance machinery (per-topic σ²_k, scaled-inv-χ² posterior — dead f0 로만 흘러감)
- sticky-CRP `alpha` 및 canonical/default `lmda=10` — `_fresh_baseline_for_prev`
  (codex 2026-05-18) 의 prior-cancel 설계로 repeat-vs-fresh argmax 에서 상쇄.
  단 후속 검증에서 `lmda=1` 같은 낮은 stickiness 는 non-prev f0 fallback 경로와
  상호작용해 일부 데이터에서 출력 차이를 낼 수 있음이 확인됨. 따라서 `lmda` 는
  전역 dead HP 가 아니라 default/canonical reduced-form parity 조건으로만 제외한다.
- `sigma_delta_c`, `var_likelihood_weight`, `pe_prior`, `cos_threshold`, `pe_threshold`,
  `hard_pe_fallback`, `min_transitions_for_pe` — 모두 dead

**환원 결과**: v4.1.3 의 실제 작동 알고리즘 =
`δ_eff = a·(1−cos(s_{t-1},s_t)) + (1−a)·(1−cos(causal_window, s_t)); boundary ⟺ δ_eff ≥ δ*`.
4단계 SEM2 파이프라인 (sCRP prior / RNN PE / σ²_k likelihood / Bayes posterior) 은
코드로 실행되나 출력 무영향.

**결정** (codex:rescue 위임 → A안 채택): main 모델을 `src/hi_em/hi_ontop.py` 의
`HiOnTop` (reduced form, dead code 0, ~270줄) 로 재정의. v4.1.3 와 byte-identical
검증 완료 (TIAGE/Dialseg711/SuperSeg 38,242 turn 0 mismatch). SEM2 full form
(`sem_core_v413.py`) 은 `archive/legacy_sem_ablation/` 으로 이동 — 삭제 아님,
paper 의 audit disclosure 재현용 ablation 증거물. `src/hi_em/sem_core_v411.py` 등
공유 인프라는 v4.2.x/v4.3.x 가 의존하므로 src 유지.

**근거 / SEM 계승 정합성**: codex 권고 — paper 는 Hi-OnTop 를 "full SEM2 구현" 이 아니라
"SEM 의 prediction-error boundary 직관의 minimal online realization" 으로 서술하고,
"SEM2-style RNN/sticky-CRP/variance 를 구현·audit 했으나 v4.1.3 default 에서 argmax
결정에 영향 없음" 을 명시 (audit disclosure). main claim 은 graded boundary score +
online O(1) latency + SeCom LLM-backend drop-in + latency 대폭 감소로 재구성.

**biology revival 검토 후 보류**: 사용자가 신경과학(해마-mPFC-PMN) 메커니즘을 실제
작동하게 되살리는 안 (mPFC reset / 해마 snapshot / re-entry, segment_id↔event_id 분리)
을 codex 와 설계까지 했으나 — "새 메커니즘 도입 말고 dead code 제거만" 으로 최종 결정.
revival 설계는 codex 스레드에 보존, 후속 버전 후보.

**영향**: `secom_adapter.py` → HiOnTop 로 re-point (byte-identical 이라 SeCom-swap 결과 불변).
`scripts/{run_v413_demo,run_v413_hp_sweep,analyze_v413_reentry}.py` 는 archived v413
import. `scripts/secom_swap/{03b,13}` 는 HiOnTop 로 re-point. methodology 갱신 필요.

**2026-05-22 후속 — `ctx_window` default 3 → 2 정정**: parity 검증
(`scripts/verify_hiontop_parity.py`, `outputs/experiments/2026-05-22_hiontop_parity/`)
중 발견 — v4.1.x 코드 default `ctx_window=3` 이 보고 수치(TIAGE 0.4675 / Dialseg711
0.5897 / SuperSeg 0.4631)를 낸 canonical TIAGE-cfg(`m=2`)와 어긋나 있었음. `HiOnTop`
는 main 모델이므로 default 를 보고 config 에 맞춰 `2` 로 정정. output parity 는 m 과
무관하게 구조적으로 성립(매칭 config 에서 38,242 turn diff 0 재검증). 산출:
`context/methodology/hi_ontop.md` 신설, `outputs/reports/hi_ontop_algorithm_walkthrough.{md,pdf}`
신규, `scripts/verify_hiontop_parity.py` 신규.

---

## 2026-05-23 — Hi-OnTop-v2: lexical-overlap 보정 변형 도입 (검증 대기, v1 유지)

**배경**: [[hi-ontop]] (`HiOnTop`) 의 알려진 실패 모드 — wording 은 비슷한데 topic 이
바뀌는 경계(δ≈낮은데 GT 경계)를 인접 임베딩 cosine 신호로는 못 잡음. 사용자 요청:
TextTiling 식 단어-빈도 겹침(lexical overlap) 신호를 δ_eff 에 결합해 v2 를 만들고
세 벤치에서 검증. 사용자 확정 방향 = **어휘 겹침↓ → δ_eff↑ / 겹침↑ → δ_eff↓**
(어휘 응집 시 경계 억제).

**설계 (codex:rescue 위임)**: 결합 형태·인과적 lexical 신호 정의·HP·SEM 계승
정당화 모두 codex 권고 채택.
`δ_eff_v2 = clip_[0,2]( δ_base + w_lex·r_t·(lexdist − μ_lex) )`. `δ_base` = HiOnTop
δ_eff 불변. `lexdist = 1 − cos_tf(L_{t-1}, subtf(u_t))`, L = 직전 m_lex turn 의
ρ_lex-감쇠 sublinear-TF 합. `r_t` = 짧은-turn 신뢰 게이트. `μ_lex` = train median
(residual centering). 가산형 채택 — 곱셈형은 δ_base 가 낮은 바로 그 실패 모드에서
힘이 약함. lexical 신호는 zero-lag·causal — streaming TextTiling 의 right-block
닫힘 lag 회피. `w_lex=0` 시 v1 과 byte-parity (검증됨).

**SEM 계승 3-step**: lexical overlap 은 SEM 에 **없음** — SEM 은 structured scene
dynamics 의 예측가능성으로 boundary 를 설명하지 단어 표면형 cohesion 을 추적하지
않음(추상화 수준 차이, 의도적 금지 아님). 충돌은 제한적 — lexical 항은 sCRP/local-MAP
가 아니므로 SEM 원형 충실 재현과는 이질적이나, Hi-OnTop-v2 목표가 SEM-inspired online
segmentation 이므로 충돌 없음. lexical 항을 **SEM core 가 아닌 domain-specific
보조 관측 feature** 로 기록. 우선순위: SEM 계승성(online·causal·local) > 도메인
적합성 > SEM 원형 충실 재현. CLAUDE.md "Retrieval 은 importance score 만" 규칙과는
무관(segmentation 영역).

**결과** (`outputs/experiments/2026-05-23_hiontop_v2/REPORT.md`, official SuperDialseg
Score, calib=train/validation/30%-tune, w_lex sweep): mean-3 v1 0.4927 → w_lex=0.30
0.4989 (+0.0062). dialseg711 은 w_lex 단조 증가(+0.005~+0.015, robust 양),
superseg 약한 양(+0.003~0.005), tiage noise(표본 100, 부호 불안정).

**결정**: **v1 대체 승격 보류**. mean-3 +0.6 Score point 는 약하고 tiage 가 noise
라 main 모델 교체 근거 부족. 단 dialseg711 단조 추세가 lexical 신호의 원리적 유효성을
보이므로 폐기도 아님 — `HiOnTopV2` 는 검증 대기 변형으로 보존. **현 main 모델 =
`HiOnTop` 유지.** 승격 재검토 전제: (a) multi-seed 유의성, (b) lexical HP 2차 grid,
(c) v1 실패 모드 turn case-level 검증.

**산출**: `src/hi_em/hi_ontop_v2.py` 신규, `scripts/run_hiontop_v2.py` 신규,
`context/methodology/hi-ontop-v2.md` 신규, `context/03-architecture.md` 갱신.

---

## 2026-05-23 — DTS 표 Ours latency 정직성 정정 (host-shared → realtime per-turn)

**배경**: `outputs/reports/dts_result.md` 의 Online "Ours (p70/p75/p80/oracle)" 4 행이
Pre. 0 · Seg. 0.2 ms 로 보고돼 있었음. caption 자체에 *"host-shared accounting (encoder
forward is amortized with the surrounding pipeline's representation step)"* 라고
명시 — 즉 encoder forward 비용이 표에서 **제외**된 채 보고됨. 사용자 정의("인코딩
캐시 따로 쓰지 않고 리얼타임 설정에서 매턴마다 계산해서 합치고 턴으로 나눈 값")
와 정반대. baseline 들(GreedySeg 13 ms 등)은 encoder forward 포함이라 한 표에서
비대칭.

**조치**: `scripts/measure_hiontop_latency.py` 신규 — Def-DTS bundle test split 에서
seed=0 으로 벤치당 500-turn budget subsample, **encoder 캐시 off + batch=1 + 매 turn
perf_counter (encode + assign)**, 첫 발화 제외(baseline 들과 동일 정책). encoder
= MPNet (`multi-qa-mpnet-base-dot-v1`) CPU. 결과 (`outputs/experiments/
2026-05-23_hiontop_latency_realtime/`):

| 벤치 | n turn | Pre. mean (ms) | Pre. p50 | Seg. mean | Seg. p50 |
|---|---:|---:|---:|---:|---:|
| tiage | 468 | 1128 | 881 | 0.42 | 0.25 |
| dialseg711 | 498 | 849 | 738 | 0.34 | 0.23 |
| superseg | 468 | 932 | 665 | 0.35 | 0.24 |
| **cross-bench** | **1434** | **967** | **747** | **0.37** | **0.24** |

**결정**: `dts_result.md` 의 Ours 4 행 Pre./Seg. 셀을 0/0.2 → **967/0.37** (cross-bench
mean) 으로 교체. caption 의 *host-shared accounting* 문장 폐기, *realtime per-turn,
no cache, ΣΔt/N* 정의로 교체. §4 TODO 의 *"Latency Pre./Seg. split 정밀 측정"* 항목
체크 처리. §6 해석 문단도 갱신 (segmentation 자체는 sub-ms 로 거의 free, 우세 비용은
encoder forward).

**한계**: CPU only, MPNet encoder 한정. GPU 또는 MiniLM/MiniLM-int8 인코더로 교체
시 Pre. 한 자릿수 ms 까지 떨어질 가능성 — 별도 측정 필요. 표본은 벤치당 500-turn
budget (전수 36k turn 측정 ~6 시간 회피).

**산출**: `scripts/measure_hiontop_latency.py` 신규, `outputs/experiments/
2026-05-23_hiontop_latency_realtime/{REPORT.md,latency.json}` 신규,
`outputs/reports/dts_result.md` 갱신.

### 2026-06-08: AMI 도메인 robustness 진단 + Hi-OnTop v3(granularity-adaptive) 라인 신설 + Hi-DoTS→Hi-OnTop rename

**1) AMI(회의 음성) 도메인 진단 — turn 단위 붕괴, 원인 = granularity 불일치**

AMI scenario meetings(NXT manual annotation, manifest 12미팅)로 Hi-OnTop 도메인
robustness 검증. **turn(=AMI segment) 단위로 완전 붕괴**:
- 같은 화제 인접 발화 cosine 중앙값 **0.167** (60%가 cos<0.2), δ_eff 경계 분리
  **AUC 0.567**(≈무정보), top-level F1 0.05~0.07, δ* p80=0.88·p98=0.98(거의 전 인접쌍이 경계처럼).
- 원인: 회의 발화 31%가 1단어 backchannel(yeah/okay/mm-hmm), median 4단어로 짧고
  다화자 반응형 → "같은 화제=비슷한 발화" 라는 방법 전제 불성립. 진짜 top 경계의
  31~44%가 ≤2단어 발화 위에 있어 **후처리(짧은 발화 drop/경계금지/defer) 전부 실패**
  (실측: F1 0.05→0.02, 신호 자체가 빔).
- **고정 시간 블록(전 화자 발화 이어붙임)으로 단위 교체 시 회복**: win sweep —
  30s AUC0.669, 45s 0.689, **60s 0.664/bestF1 0.379·bestScore 0.446**, 90s 0.629.
  같은화제 cos 0.167→0.44. **60s가 sweet spot이나 = AMI에 최적화된 magic number**
  (label-free 철학 위배). 산출: `scripts/ami_topic_eval.py`(turn), `ami_topic_block_eval.py`(블록),
  `outputs/experiments/2026-06-08_ami_topic_block_w{30,45,60,90}/`.

**2) 결정 — Hi-OnTop v3 = "granularity 적응" major 라인 신설, 경쟁 후보 2종**

적응 축 정리: v1=고정 / **v2=threshold 적응**(lexical 잔차가 turn별 effective δ*) /
**v3=granularity 적응**(무엇을 분절기에 넣을지). v3 아래 같은 문제의 경쟁 해법 2종:
- **v3.1 (B)** — 명시적 단위 적응: 시간 대신 content량(~N단어/토큰) 기준 적응 블록.
  단위 선택 층을 δ_eff 앞에 둠. (TextTiling baseline 과는 *역할* 이 다름 — baseline은
  최종 경계 산출기, v3.1은 unit 형성기.)
- **v3.2 (C)** — SEM 계승형: 고정 블록 폐기, turn 유지하되 각 발화를 **현재 segment
  누적 centroid μ_k**(=현재 event model)와 비교 + 짧은 발화 down-weight + sCRP
  stickiness(α)로 단일 blip 흡수 → segment 경계가 sequential MAP에서 emerge, magic 단위 없음.
  δ_ctx(고정 window m=2)의 *segment 전체-window + 경계 reset* 일반화.

**승자 = 정식 v3 승격** (.x 후보/ablation 보존). 결정 규칙(사후 cherry-pick 방지):
(a) AMI turn 단위 AUC 0.567/F1 0.05 대비 유의 상승, (b) DTS 3벤치(TIAGE/Dialseg711/
SuperSeg) v1/v2 무회귀, (c) magic number 없이 label-free/deployable 유지(남는 HP는
인코더·모델 속성, 도메인 의존 아님). 기존 experimental→승격 패턴(v3.3.10→v4.1.1) 계승.

**SEM 계승 3-step (v3.2 핵심)**: ① backchannel 처리는 SEM/SEM2 입력 도메인(풍부한
scene/단락) 밖이라 미구현 — 의도적 배제 아님. ② 누적 centroid 비교 = SEM `f_k.update(x_t)`
근사, α의 blip 흡수 = sticky prior 본래 역할 → SEM 메커니즘 직접 구현, 철학 충돌 없음.
③ 본 엔트리로 근거 기록. **v3.1(B)의 lexical/표면 cohesion은 SEM 비계승**(sCRP 연결고리
없음) — 실용 대안으로만.

**구현 순서**: v3.2(C) 먼저(`hi_ontop_v3_event_exp.py`/`HiOnTopV3Event`) → AMI+DTS 실측 →
v3.1(B)(`hi_ontop_v3_unit_exp.py`/`HiOnTopV3Unit`) → 비교표 → 승자 v3 승격.

**3) Hi-DoTS → Hi-OnTop 전면 rename (흔적 제거)**

Hi-DoTS는 Hi-OnTop의 옛 이름. 코드는 이미 마이그레이션 완료(`hi_ontop.py`/`HiOnTop`),
잔존하던 backward-compat shim(`hi_dots.py`/`hi_dots_v2.py`)·별칭(`HiDoTS=HiOnTop`)·파일명·
디렉토리·문서 흔적을 전부 제거. shim 2개 삭제, 파일/디렉토리 44개 rename, 텍스트 67개
파일 치환(Hi-DoTS→Hi-OnTop 등). DTS(task)·Def-DTS(외부레포)는 불변. 검증: 텍스트 흔적 0,
import smoke OK, 테스트 회귀 0(267 passed; 기존 실패 4건은 STM/locomo-data 무관 이슈).
**미해결**: figure PDF/PNG 내부 렌더링 텍스트(압축 스트림)는 재생성 필요, methodology
`hi_ontop.md`/`hi-ontop.md` 중복(구 `hi_dots.md`/`hi-dots.md`) 병합 보류.

---

## 2026-06-10 — AMI 분절: V_rel 상대신호 확립 + online reset 부트스트랩 병목 (Hi-OnTop)

**배경**: AMI 139미팅에서 기존 Hi-OnTop δ_eff ewma = ±2F1 0.151 / Score 0.203. LLM
full-context 0.543/0.640 대비 큰 격차. 원인 규명 + 신호 개선 (codex gpt-5.5 위임 2회).

**결정 1 — 경계 판정을 magnitude 임계치 → "active-event 설명 실패 + background 상대거리"로.**
진단: 가장 강한 δ_eff peak가 경계 아니라 noise(화자전환·단발 이상치). magnitude 단독 분리 불가.
신호 = **V_rel = r_active − λ·r_global** (r_active=1−cos(x, active prototype EWMA);
r_global=1−cos(x, global running centroid, g_rho=0.15); λ=0.6). gold-reset clean prototype +
이 신호 → oracle ±2F1 천장 0.687(>LLM 0.543). overfit 2-fold 격차 0.000 (knob 전 split 동일).

**SEM 계승 3-step**: ① r_global(background 대비)은 SEM new-event base distribution 비교의 거리공간
근사 — codex 설계 `log p(x|new)−log p(x|active)`의 reduction. 의도적 도입, SEM에 근거 있음.
② active prototype 누적·경계 reset = SEM event 모델 갱신/신규 event 개시. sCRP/local MAP 철학과 정합.
③ 본 엔트리 기록. λ·g_rho는 calibration(인코더·도메인 속성), magic number 아님(overfit 검증 완료).

**결정 2 — 병목은 임계치가 아니라 online reset 부트스트랩 (미해결, 다음 iteration).**
격차 분해: clean(gold-reset)+단순 μ+cσ = ±2F1 0.554(LLM급). 즉 신호·임계치 충분. 그러나
detected-reset deploy = 0.15 (오염→under-seg 악순환). robust·peak·anchor·refractory·EM 반복정제·
BOCPD top-K particle filter(codex 설계) 모두 oracle에 도달 못 함. BOCPD는 개수만 잡음(Score 0.305,
±2F1 0.069). **deployable 최선 = 단순 V_rel 적응임계치 Score 0.358** (기존·TextTiling·even-spacing
oracle 모두 상회).

**결정 3 — AMI 포지셔닝 = robustness/한계 도메인 (primary 아님).** drift+sparse+gold-offset 3중
난점, even-spacing oracle이 Pk/WD 지배(content 신호와 구조적 불일치). DTS(concat-seam) primary 유지.
codex 자문 일치.

상세 + 전체 수치: `outputs/experiments/2026-06-10_ami_vrel_localmap/REPORT.md`.
신규 스크립트: `ami_vrel_eval.py`, `ami_vrel2_eval.py`, `ami_bocpd_eval.py`, `ami_localmap_eval.py`.
**미해결**: V_rel을 정식 hi_ontop 모듈/버전으로 승격할지는 deploy가 oracle 격차 메운 뒤 결정 (현재 보류).
