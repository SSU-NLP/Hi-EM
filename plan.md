# 구현 로드맵

> **Step 완료 규칙:** 각 Step의 하위 항목을 `[x]`로 표시하기 전,
> `python scripts/check_step_done.py`가 exit 0을 반환해야 한다.
> FAIL이 나오면 원인을 수정하고 재실행한다. 통과할 때까지 반복. 세부는 `CLAUDE.md` 참조.

## Phase 0: 자료 분석 및 설계 확정 (현재 단계)
### 0-1. SEM 논문 정독
- [x] `SEM-paper.pdf` 전체 정독 (pdftotext 산문 + 수식 10페이지 `pdftoppm` 이미지 직독)
- [x] 핵심 수식 1~24 이해 및 Hi-EM 계승/대체/폐기 분류 (context/00-sem-paper.md §2, §5)
- [x] `SEM/sem/sem.py` 실제 코드 검증 (`_calculate_unnormed_sCRP` L.144-159, `run()` L.161-354) — SEM2 기준 verbatim pseudocode 기록 (§4)

### 0-2. 벤치마크 데이터 분석
- [x] LoCoMo clone (`benchmarks/locomo/`)
- [x] TopiOCQA clone (`benchmarks/topiocqa/`)
- [x] LongMemEval clone (`benchmarks/LongMemEval/`)
- [x] LoCoMo `data/locomo10.json` 구조 분석 (10 conv × 27 sess × 22 turns, topic annotation 없음)
- [x] TopiOCQA 데이터 다운로드 및 분석 (dev 2514 turns, 205 conv, Wiki Topic ground truth)
- [x] LongMemEval 데이터 다운로드 및 분석 (oracle 500 Q, 6 question_types, 긴 chat)
- [x] `outputs/benchmark-analysis.md` 작성 (8239자, 3 벤치마크 + 옵션 A~F 증거 매트릭스 + Step 0-3 입력)

### 0-3. 사건 모델 설계 판단
- [x] 벤치마크 분석 결과에 근거해 사건 모델 형태 결정 (옵션 A — Centroid + diag variance)
- [x] `context/01-hi-em-design.md`의 미확정 섹션 채우기 (§4 옵션 A 확정, §3 Markov 확장 철회, B~E는 Phase 1/2로 위임)
- [x] `context/02-math-model.md`의 수식 확정 ($P(s_n|e_n=k)=\mathcal{N}(\mu_k, \mathrm{diag}(\sigma_k^2))$, PE = Mahalanobis)
- [x] `context/06-decision-log.md`에 판단 근거 기록 (옵션 A 선택 + Markov 철회 + 기각 옵션별 사유)

## Phase 1: 코어 모듈 구현 (src/hi_em/)
- [ ] 쿼리 임베딩 모듈 (bge-base-en-v1.5)
- [ ] Topic 클래스 (확정된 사건 모델 형태에 따라)
- [ ] sCRP prior 계산 (SEM/sem/sem.py 참조)
- [ ] Topic 배정 로직 (확정된 likelihood 형태에 따라)
- [ ] Online MAP inference 루프

## Phase 2: 메모리 계층
- [ ] LTM (저장 형식 결정 후)
- [ ] STM (in-memory)
- [ ] Topic importance 계산
- [ ] Topic merge 로직
- [ ] KV cache paging 인터페이스 (초기 stub)

## Phase 3: 오케스트레이션
- [ ] 매 턴 파이프라인
- [ ] 매 라운드 비동기 파이프라인

## Phase 4: 평가
- [ ] LoCoMo QA accuracy
- [ ] LongMemEval 5개 능력별 accuracy
- [ ] TopiOCQA topic shift F1
- [ ] Latency 측정
- [ ] KV cache 효율 측정

## Phase 5: 논문 실험
- [ ] Ablation study
- [ ] Baseline 비교 (MemGPT, RAG, sliding window)

## 각 Phase 완료 기준
`outputs/phase-N-results.md`에 측정 결과 기록.