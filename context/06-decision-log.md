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

### 2026-04-23: Markov 확장 $P(e_n\mid e_{n-1}, s_{n-1})$ 철회

**근거**:
- 위 "사건 모델 옵션 A 확정"과 함께.
- 옵션 A의 likelihood가 이미 $s_n$을 centroid 대비 평가하고, 이력은 centroid 업데이트(Welford)에 implicit 반영됨.
- prior에 $s_{n-1}$ 의존 항을 추가하면 **double counting** + prior/likelihood 역할 혼합.

**결정**: prior은 SEM 원본 Eq 1 그대로 유지. scene-conditional 신호가 필요하면 likelihood(사건 모델)를 확장한다.

**영향 범위**: `context/01-hi-em-design.md §3`("철회" 기록), `02-math-model.md` 생성 모형 단순화.

**대안**: 확장 유지하되 double counting은 정규화로 상쇄 (기각: 해석 혼란, 이득 불명).