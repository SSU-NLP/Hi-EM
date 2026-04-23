# Hi-EM 설계 결정사항

---

## 확정 사항

### 1. Scene 임베딩
- **인코더**: `BAAI/bge-base-en-v1.5` (768dim)
- **입력**: 쿼리만 (응답 미포함)
- **정규화**: L2 normalize 후 저장
- **근거**: no fine-tuning 제약 + 실시간 처리 요구

### 2. Prior: sticky-CRP (SEM2와 동일 수식, 하이퍼파라미터 반전)

$$\Pr(e_n = k \mid e_{1:n-1}) \propto \begin{cases} C_k + \lambda \mathbb{I}[e_{n-1}=k] & k \leq K \\ \alpha & k = K+1 \end{cases}$$

- $\alpha = 1.0$, $\lambda = 10.0$ (초기값, 튜닝 대상)
- **근거**: topic 수 자동 결정 + switch-to-old 자연 처리

### 3. Markov 가정 확장

SEM:
- $P(e_n \mid e_{n-1})$
- $P(\mathbf{s}_n \mid \mathbf{s}_{n-1}, e_n)$

Hi-EM:
- $P(e_n \mid e_{n-1}, \mathbf{s}_{n-1})$ — 직전 쿼리 조건 추가
- $P(\mathbf{s}_n \mid e_n, \cdot)$ — 구체 형태는 **미확정**

### 4. 추론 방식: Local MAP approximation

$$\hat{e}_n = \arg\max_{e_n}\, \Pr(e_n \mid \mathbf{s}_{1:n}, \hat{e}_{1:n-1})$$

SEM2 `run()` 루프와 동일한 관점. TF/GRU 의존성만 제거.

### 5. Memory Reconstruction 폐기
Gibbs sampling 불필요. LTM에 원문 그대로 저장.

### 6. 학습 타이밍
- 매 턴 (online): 통계량 업데이트 (Welford)
- 매 라운드 (비동기): refinement, merge 검사, importance 계산

### 7. Cold Start
새 topic은 prior variance $\sigma_0^2$로 시작, $n_e \geq 3$부터 running variance로 전환.

### 8. 메모리 계층 (개념적 확정)
- **LTM**: 모든 턴 원문 + topic 메타 저장
- **STM**: importance 상위 topic만 상주

---

## 미확정 사항 (벤치마크 분석 후 결정)

### A. 사건 모델 (CRITICAL)

**Claude Code가 Phase 0 완료 후 결정.**

SEM에서 RNN을 썼던 $P(\mathbf{s}_n \mid e_n, \cdot)$을 Hi-EM은 어떻게 구현할지 미정.

고려할 옵션 (열린 목록):

| 옵션 | 형태 | 장점 | 단점 |
|---|---|---|---|
| A | Centroid only: $\mathcal{N}(\boldsymbol{\mu}_e, \sigma_e^2 I)$ | 최경량, cold start 문제 없음 | 순서 정보 소실 |
| B | Centroid + Momentum: $\mathbf{s}_{n-1} + \boldsymbol{\delta}_e$ | 순서 정보 포함 | 쿼리-토픽 순서 느슨하면 무의미 |
| C | Centroid + Entity set | Wiki QA류에서 강력 | Non-entity 대화에 약함 |
| D | Multi-signal ensemble | 다양한 신호 결합 | 가중치 튜닝 필요 |
| E | Small linear predictor | 순서 의존성 포착 | 작은 topic에서 과적합 |
| F | 새 제안 | — | — |

**결정 근거는 `outputs/benchmark-analysis.md`에 있어야 한다.**

### B. 메모리 계층 세부
- LTM 저장 형식: JSON vs SQLite vs Parquet
- STM 용량 $K_{\text{STM}}$: 고정값 vs 적응적
- KV cache paging 수준: 실제 vLLM 통합 vs stub

### C. Topic Importance 공식
개략 heuristic은 있지만 구체 가중치는 실험 후 튜닝.

### D. Topic Merge 기준
Centroid cosine threshold 등 구체 값 미정.

### E. 응답 생성 LLM
OpenAI API / Anthropic API / 로컬 HF 모델 중 선택.

---

## 변경 이력
`context/06-decision-log.md` 참조.