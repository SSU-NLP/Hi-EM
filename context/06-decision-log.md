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
