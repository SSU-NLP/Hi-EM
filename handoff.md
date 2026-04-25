# Claude Code 핸드오프

> **이 파일은 매 세션 시작 시 첫 번째로 읽는 파일이다.**
> **매 세션 종료 시 반드시 업데이트된다.**

---

## 세션 시작 프로토콜 (매번 순서대로)

1. 이 파일(`handoff.md`) 전체 읽기
2. `plan.md`에서 현재 Phase와 미완료 체크박스 확인
3. `CLAUDE.md`의 작업 규칙 확인
4. 현재 Phase에 해당하는 `context/*` 문서 읽기
5. 마지막 commit 로그 확인: `git log -5 --oneline`
6. "다음 할 일" 섹션(아래)의 첫 항목부터 시작

---

## 세션 종료 프로토콜 (매번 순서대로)

작업을 멈추기 전에 **반드시** 아래를 수행:

1. 이 파일의 "현재 상태"와 "다음 할 일" 섹션 업데이트 (날짜 `마지막 업데이트` 오늘로 갱신)
2. 설계 결정이 있었다면 `context/06-decision-log.md`에 append
3. 설계 변경이 있었다면 `context/01-hi-em-design.md` 또는 `02-math-model.md` 갱신
4. **3-angle self-audit 수행** (CLAUDE.md "Step 완료 프로토콜 1단계" 참조): 구조/동작/설계 각도에서 최소 3 Q&A씩. 답 못 한 지점은 결과물 파일에 "검증 미해결" 섹션으로 기록.
5. **`python scripts/check_step_done.py` 실행 → exit code 0 받을 때까지 수정-재실행 반복**
6. `plan.md` 체크박스 갱신 (검증 통과한 항목만 `[x]`)
7. 커밋 명령어를 사용자에게 **제시**한다 (커밋 메시지는 CLAUDE.md 규칙 따름). **Claude Code는 직접 실행하지 않는다.**
8. 다음 세션이 바로 이어갈 수 있게 이 파일의 "다음 할 일" 첫 항목을 구체적으로 작성

---

## 현재 상태

**마지막 업데이트**: 2026-04-25
**현재 Phase**: Phase 1 — Topic 경계 감지 코어 + 평가 (1-1~1-5 완료, 1-6 종합 Gate **FAIL** → 사용자 결정 대기)
**진행률**: Phase 0 완료 + Phase 1 측정 완료, **종합 Gate FAIL로 Phase 2 진입 보류**

### 완료된 것
- 프로젝트 디렉토리 구조 생성
- 벤치마크 레포 4종 clone (`benchmarks/{locomo, topiocqa, LongMemEval, tiage}/` — 모두 gitignored)
- SEM2 레포 (`SEM/` ← `nicktfranklin/SEM2`, 과거 v1에서 교체)
- 설계 문서 (`context/*.md`)
- **Step 0-1 완료**: SEM 논문 전 35페이지 정독, SEM2 코드 검증, `context/00-sem-paper.md` (Hi-EM 관점 정리) + `context/sem-equations.md` (식 1~24 원본 LaTeX reference) 작성.
- **Step 0-2 완료**: 4 벤치마크 실데이터 분석 → `outputs/benchmark-analysis.md`. **평가 축 분리 결정** — 토픽 경계 감지(TopiOCQA + TIAGE) vs downstream QA(LongMemEval, LoCoMo).
- **Step 0-3 완료**: 사건 모델 **옵션 A** 확정, Markov 확장 철회.
- **Phase 1-1, 1-2 완료**: `src/hi_em/{embedding,topic,scrp,sem_core}.py` 구현 + `tests/` 18 tests passing.
- **Phase 1-3, 1-4 완료**: TopiOCQA dev. **Gate PASS (marginal)** — Hi-EM F1=0.471 vs cosine 0.467 (HP α=10, λ=1, σ₀²=0.1, SEM2 defaults). 7-iteration 탐색으로 bge intrinsic ceiling ~0.47 확인.
- **Phase 1-5 완료**: TIAGE test. **Gate FAIL** — Hi-EM persistence F1=0.317 / freq-shift F1=0.377, **둘 다 cosine baseline 0.421에 패배**. Latency 0.73 ms/turn은 PASS.
- **Phase 1-6 종합 Gate: FAIL** — TopiOCQA PASS + TIAGE FAIL → Phase 2 진입 자격 미충족.
- **2026-04-25 환경 복구 + TIAGE sweep 완료**: 로컬 `.venv` (uv-managed Python 3.11) 셋업, TopiOCQA F1=0.471 / TIAGE F1=0.317·0.377 재현 검증. **TIAGE 108-config grid sweep 신규 실행** (`run_tiage_sweep.py`) → **best Hi-EM F1=0.383 (α=10, λ=3, σ₀²=0.1), cosine 0.421 미달 + 0.4 floor 미달, Gate 두 조건 모두 FAIL**. "TopiOCQA만 sweep best, TIAGE는 두 점만"의 비대칭 해소 → 옵션 1(TIAGE HP sweep) 사실상 종료.
- **2026-04-25 옵션 5 (clustering quality) 완료**: V-measure/NMI/ARI 측정 (`run_clustering_quality.py`). **모든 metric에서 cosine 우위** — 원래 가설(Hi-EM 토픽 ID 묶기 우위) 반박. **새 발견**: Boundary F1 ↔ ARI **trade-off** — freq-shift HP (α=10): F1↑ ARI=0.187·0.314 / persistence HP (α=1): F1↓ ARI=0.398·0.397. **메모리 시스템엔 persistence HP가 적합** (completeness↑, 같은 토픽 복귀 cluster 보존 우선) → Phase 2 HP 선택 근거. `outputs/phase-1-clustering-quality.md`
- **Phase 2.5 폐기**: LongMemEval session=topic 가정이 잘못된 설계였음 (한 세션 내 subtopic 공존 정상). LongMemEval은 Phase 4 downstream QA용으로 재배치.
- **종합 보고서 작성**: `report.md` (Phase 0 시작 ~ Phase 1-5 시점, 12 섹션 + 부록).

### 미완료
- **Phase 1-6 FAIL 후 결정 분기** (5 후보) — `report.md §12` 참조
- Phase 2 (LTM + Memory window) — 1-6 결정 후 진입
- Phase 3 (오케스트레이션)
- Phase 4 (4-baseline QA accuracy: Sliding window / Full context / RAG / Hi-EM)
- Phase 5 (논문 실험)

---

## 다음 할 일 (세션 시작 시 여기서부터)

### 즉시 결정 필요: Phase 1-6 Gate FAIL 후 진로

5 후보 중 옵션 1 종료(2026-04-25 sweep). 권장 경로 = **옵션 5 → 옵션 3** 묶음.

1. ~~**TIAGE HP sweep**~~ ✅ **완료 (2026-04-25)** — 108 configs all-FAIL, best F1=0.383 < cosine 0.421. `outputs/phase-1-tiage-sweep.json`
2. **Hi-EM likelihood 교체** (옵션 A 변형) — `cosine(s, last_turn_in_topic)`로 likelihood 형식 변경, sCRP prior 유지 — 보류
3. **Phase 2 reframing 진입** ⭐ — "boundary F1 ≠ Hi-EM 핵심 가치"라 정직 기록 후 LTM/Memory window 설계 → Phase 4 downstream QA로 진짜 가치 검증
4. **옵션 D escalation** (multi-signal 재설계) — TopiOCQA에서 효과 약함 전례, 보류
5. ~~**Clustering 품질 추가 측정**~~ ✅ **완료 (2026-04-25)** — 가설 반박 + Boundary F1↔ARI trade-off 발견. 다음 = 옵션 3 진입.

**핵심 메타 질문** (`report.md §11`):
- Topic boundary F1 우위가 Hi-EM의 정의된 contribution인가? Yes → 1, 2, 4 / No → 3
- Phase 4 downstream QA에서 Hi-EM이 RAG를 이긴다고 합리적으로 기대 가능한가?

### Step 1-4 — Gate 판정 (Phase 1-3 후속, 완료됨)

- **PASS**: `Hi-EM F1 > cosine baseline F1` AND `Hi-EM F1 > 0.4` AND `latency +20% 이내` → Phase 2 진입
- **FAIL**: `06-decision-log.md` append → 옵션 A "번복됨" 마킹 → 옵션 D로 재시작

---

## 주의: 이전 대화의 편향된 결론을 맹신하지 마라

이전에 "Centroid + Entity set + Multi-signal" 같은 설계가 논의됐지만,
이는 TopiOCQA/TIAGE만 본 편향된 판단이었다.
LoCoMo/LongMemEval까지 직접 보고 다시 판단해라.

---

## Phase별 진입점

현재 Phase에 따라 다른 파일을 집중적으로 봐라.

### Phase 0 (현재): 자료 분석 및 설계 확정
- 주로 읽기: `SEM-paper.pdf`, `benchmarks/*/`, `context/00-sem-paper.md`, `context/04-benchmarks.md`
- 주로 쓰기: `outputs/benchmark-analysis.md`, `context/01-hi-em-design.md`, `context/02-math-model.md`, `context/06-decision-log.md`
- 완료 기준: 사건 모델 형태가 `context/01-hi-em-design.md`에 확정됨

### Phase 1: Topic 경계 감지 코어 + TopiOCQA sanity check
- 주로 읽기: `context/01-hi-em-design.md` §4, `02-math-model.md`, `SEM/sem/sem.py` `_calculate_unnormed_sCRP`/`run()`, `outputs/benchmark-analysis.md`
- 주로 쓰기: `src/hi_em/{embedding,topic,scrp,sem_core}.py`, `tests/test_{scrp,topic,sem_core}.py`, `scripts/run_topiocqa_segmentation.py`, `outputs/phase-1-topiocqa.md`
- 완료 기준 (1-4 Gate 모두 만족):
  - 단위 테스트 통과
  - `Hi-EM F1 > cosine baseline F1` (TopiOCQA dev)
  - `Hi-EM F1 > 0.4`
  - 턴당 latency 증가 +20% 이내
- FAIL 시: 옵션 D로 escalation, `06-decision-log.md` append 후 Phase 1 재시작

### Phase 2: 메모리 계층 (LTM + Memory window)
- LTM = SSD 파일 영속 저장, Memory window = 현재 라운드 prefill 대상 STM
- 저장 포맷, Memory window 크기·구성 정책, importance/merge 확정

### Phase 3 이후: `plan.md` 참조

---

## 벤치마크 데이터 준비 (Phase 0 Step 2에서 필요)

### LoCoMo
```bash
cat benchmarks/locomo/data/locomo10.json | jq '.[0]' | head -200
```

### TopiOCQA
```bash
cd benchmarks/topiocqa
python download_data.py --resource data.retriever.all_history.dev
cd ../..
```

### LongMemEval
```bash
cat benchmarks/LongMemEval/README.md
# HuggingFace에서 받음 - README 안내 따라감
```

---

## 환경 세팅 (아직 안 했으면)

```bash
cd /home/namchailin/Hi-EM

python --version  # 3.10+ 확인
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 환경 검증
python scripts/verify_env.py
```

Colab A100 환경은 `colab/README.md` 참조.

---

## 외부 레포 사용 규칙 (강제)

- `benchmarks/*` — **읽기 전용.** 수정 금지.
- `SEM/` (= SEM2, `nicktfranklin/SEM2` current build) — **참조 전용.** 알고리즘 구조만 참고, 코드 복사 금지. 원본 아카이브는 `ProjectSEM/SEM`.
- `SEM-paper.pdf` — 수정 불가.

---

## 전역 금지 사항 (CLAUDE.md와 중복이지만 재확인)

- TensorFlow/Keras 사용 금지 (PyTorch만)
- 외부 LLM fine-tuning 금지
- SEM2 코드 직접 복사 금지 (참조만 허용)
- 벤치마크 근거 없이 사건 모델 확정 금지
- `handoff.md` 업데이트 없이 세션 종료 금지
- `context/06-decision-log.md`에 기록 없이 설계 결정 금지

---

## 참고 문서 전체 목록

반드시 읽을 것:
- `brief.md` — 프로젝트 한 줄 요약
- `plan.md` — 전체 로드맵 + 체크박스
- `CLAUDE.md` — 코딩 규칙
- `context/00-sem-paper.md` — SEM 계승/폐기 정리
- `context/01-hi-em-design.md` — 확정/미확정 설계
- `context/02-math-model.md` — 수식
- `context/03-architecture.md` — 파일 구조
- `context/04-benchmarks.md` — 벤치마크 정보
- `context/05-open-questions.md` — 열려있는 질문
- `context/06-decision-log.md` — 결정 이력

필요 시:
- `SEM-paper.pdf` — Franklin et al. 2020 원본 논문
- `SEM/` — SEM2 코드 (참조, `nicktfranklin/SEM2`)
- `colab/README.md` — Colab A100 환경