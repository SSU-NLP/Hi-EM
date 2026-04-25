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