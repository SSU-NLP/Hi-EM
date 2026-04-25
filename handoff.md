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
**현재 Phase**: Phase 4 — downstream QA 4-way baseline (Step 4-2/4-3/4-4 완료, **4-5 sanity 사용자 실행 대기**).
**진행률**: Phase 0/1/2/3 완료. Phase 4 Step 4-1 (clone, data 다운로드는 사용자) + 4-2 (preload_history, 2 tests) + 4-3 (run_longmemeval.py 4 method) + 4-4 (judge_longmemeval.py, Qwen judge) 완료. 전체 테스트 **51/51 PASS**. **사용자 실행: subset 30 sanity → 전체 500 → 분석**.

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
- **2026-04-25 Phase 2 진입 + Step 2-1 완료**: LTM 저장 포맷 확정 — **per-conversation JSONL (turn append-only) + `<conv_id>.state.json` (topic 상태 latest snapshot, overwrite)**. 디렉토리 `data/ltm/` (gitignored). Topic 분할 HP **persistence (α=1, λ=10, σ₀²=0.01)** 채택. 자세한 내용·trade-off: `context/01-hi-em-design.md §9.1` + `06-decision-log.md` 2026-04-25 entry.
- **2026-04-25 Step 2-2 완료**: `src/hi_em/ltm.py` (LTM API) + `tests/test_ltm.py` (8 tests). API: `append_turn / update_state / load_turns(topic_id?) / load_state / list_conversations`. validation 없음 (내부 모듈, schema는 §9.1 참조). 전체 테스트 회귀 **26/26 PASS**.
- **2026-04-25 Step 2-3 완료**: `src/hi_em/memory_window.py` — `select_memory_window(q, ltm, conv_id, k_topics, k_turns_per_topic)` baseline policy: cosine top-k topics × recency top-k turns/topic, flatten by turn_id ascending. `tests/test_memory_window.py` 8 tests. 전체 회귀 **34/34 PASS**. Step 2-4 (importance/merge/adaptive K)는 Phase 4 downstream 결과로 튜닝 — 미리 구현하면 over-engineering.
- **2026-04-25 Step 3-1 완료**: `src/hi_em/llm.py` — `OpenAIChatLLM(api_key, base_url)` + `chat(messages, model, **kwargs)`. **OpenAI-compatible** (OpenRouter / vLLM / OpenAI 본가 모두 동일 SDK). env var: `OPENAI_API_KEY` + `OPENAI_BASE_URL` (생성자 인자 우선). `requirements.txt` openai>=1.30 활성화 (실제 설치된 버전 2.32.0). `tests/test_llm.py` 5 tests (mock client). 전체 회귀 **39/39 PASS**. 백엔드 결정 근거: `memory/project_llm_backend.md`.
- **2026-04-25 Step 3-2 완료**: `src/hi_em/orchestrator.py` — `HiEM(conv_id, encoder, llm, model, ltm_root, alpha, lmda, sigma0_sq, k_topics, k_turns_per_topic, system_prompt?, **llm_kwargs).handle_turn(user_text) -> str`. 7단계 파이프라인 (embed → segment → snapshot → memory_window → messages → llm.chat → append user/assistant). 순서: select 시점에 user turn 미저장 → 직전 user 필터링 불요. assistant turn은 embedding=None, 직전 user의 topic_id 상속. `tests/test_orchestrator.py` 9 tests (FakeEncoder + mock LLM). 전체 회귀 **48/48 PASS**. **A→B→A 토픽 복귀 시 첫 A turn 자동 prefill 검증**. §9.1 schema 단순화: first/last_turn_id 제거 (실 사용처 없음). 세션 간 segmenter 상태 복원 미지원 (Phase 5 필요 시 추가).
- **2026-04-25 Step 3-3 완료**: `scripts/smoke_test_orchestrator.py` 실 LLM 검증. `.env` (python-dotenv) 자격증명 + `.env.example`. 환경: vLLM `http://210.222.65.89:50200/v1` + `Qwen/Qwen3-8B`. **결과 PASS** (`outputs/phase-3-smoke.md`): Turn 1·3 same topic_id=0, Turn 3 응답이 Turn 1 정보(가을 시기) 명시 인지. **`response_filter` 옵션 추가** (caller=raw, LTM=filtered) — Qwen `<think>` 블록 prefill 토큰 절약. test 1개 추가, 전체 **49/49 PASS**.
- **2026-04-25 Phase 4 진입 + Step 4-2/4-3/4-4 완료**: LongMemEval 벤치마크 평가 인프라 구축. (1) `HiEM.preload_history(turns)` 메서드 추가 — segmenter는 user만 통과, assistant는 직전 user의 topic 상속. 2 tests 추가 → **51/51 PASS**. (2) `scripts/run_longmemeval.py --method {sliding,full,rag,hi-em}` 4 baseline 통합 — sliding K=20 / full / rag bge-cosine top-K=10 / hi-em persistence HP. (3) `scripts/judge_longmemeval.py` — LongMemEval 6 prompt template 인용 (MIT, Copyright 2024 Di Wu), **judge model = Qwen/Qwen3-8B** (응답 생성과 동일 vLLM, 비용 0). question_type 5축 별 + abstention 분리 집계. **다음: 사용자 데이터 다운로드 + subset 30 sanity → 전체 500**.
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

### 다음: Phase 4 Step 4-5 sanity 실행 (사용자 명령 실행)

**준비**:
1. LongMemEval data 다운로드 (oracle 우선):
```bash
cd /Users/joseonghyeon/Desktop/Hi-EM/benchmarks/LongMemEval/data
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
cd /Users/joseonghyeon/Desktop/Hi-EM
```

**Subset sanity (30 questions × 4 method)**:
```bash
for m in sliding full rag hi-em; do
    uv run python scripts/run_longmemeval.py --method $m --limit 30 \
        --output outputs/phase-4-sanity-$m.jsonl
done
```

**Judge 4번**:
```bash
for m in sliding full rag hi-em; do
    uv run python scripts/judge_longmemeval.py outputs/phase-4-sanity-$m.jsonl \
        --ref benchmarks/LongMemEval/data/longmemeval_oracle.json
done
```

→ 결과 4 method × accuracy + question_type별 분리 알려주시면, 다음 단계 (전체 500 실행 또는 sanity 결과로 발견된 이슈 수정) 결정.

### Phase 4 plan (Step 4-1~4-7) — 자세히는 plan.md 참조

**목적**: Hi-EM의 진짜 가치 (boundary F1·ARI 모두 unsupervised metric으론 cosine 우위 → reframing 인정) 정량 검증. 4-way 비교에서 Hi-EM이 baseline 대비 의미 있는 개선을 보여야 Phase 5 (논문 실험) 진입 자격.

**4-way baseline**:
1. **Sliding window** (직전 K turn만 prefill)
2. **Full context** (모든 history prefill, 토큰 한계까지)
3. **RAG** (모든 turn 임베딩 후 cosine top-K 검색)
4. **Hi-EM** (현재 구현된 segmenter + memory_window)

**벤치마크**:
- LongMemEval (2025+, 5개 능력 별: single-session-user, single-session-assistant, temporal-reasoning, multi-session, knowledge-update)
- LoCoMo (snap-research, 10 conversations)

**HP 사용**: persistence (α=1, λ=10, σ₀²=0.01) — 옵션 5 결과의 cluster 보존성 우위 근거

**Step 4-1 (사용자 결정 필요)**:
- 어떤 벤치마크부터? (LongMemEval가 5축 평가라 풍부, LoCoMo는 단순)
- 어떤 model로? (smoke test의 Qwen3-8B 그대로? 다른 model?)
- 평가 metric: accuracy / F1 / GPT-judge-score (LongMemEval 기본은 GPT-judge)

**LongMemEval 권고** — 5개 능력 분리 측정으로 "어디서 Hi-EM이 강한가" 식별 가능. 단 평가 비용↑ (LLM judge 호출).

**Step 4-2~**: 평가 스크립트 작성, 실행, 결과 정리, Phase 5 진입 판정.

### Phase 1-6 결정 분기 진행 상황 (참고)

### Phase 1-6 결정 분기 진행 상황 (참고)

1. ~~**TIAGE HP sweep**~~ ✅ 완료 (2026-04-25). `outputs/phase-1-tiage-sweep.json`
2. **Hi-EM likelihood 교체** — 보류 (Phase 4 결과 후 재검토)
3. **Phase 2 reframing 진입** ✅ **채택, 진행 중** (Step 2-1 완료, 2-2부터 시작)
4. **옵션 D escalation** — 보류
5. ~~**Clustering 품질 측정**~~ ✅ 완료 (2026-04-25). `outputs/phase-1-clustering-quality.md`

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
uv sync   # .venv 생성 + deps 설치 + hi_em editable install (한 줄)
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