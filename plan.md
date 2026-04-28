# 구현 로드맵

> **Step 완료 규칙:** 각 Step의 하위 항목을 `[x]`로 표시하기 전,
> `python scripts/check_step_done.py`가 exit 0을 반환해야 한다.
> FAIL이 나오면 원인을 수정하고 재실행한다. 통과할 때까지 반복.
> 3-angle self-audit(구조/동작/설계)은 스크립트 실행 전 필수. 세부는 `CLAUDE.md` 참조.

## Phase 0: 자료 분석 및 설계 확정
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

---

## Phase 1: Topic 경계 감지 코어 + 최소 동작 sanity check (현재 단계)

**목적**: 옵션 A 기반 segmentation이 최소 벤치마크(TopiOCQA)에서 쓸만한지 먼저 확인. 이 gate를 통과해야 Phase 2(메모리 계층) 설계가 허구 위에 쌓이지 않는다.

**Gate 의미**: TopiOCQA는 Hi-EM 주 타깃(Claude-유사 대화)과 다르므로 "최소 동작 sanity check"로 이해. PASS ≠ 최종 성공, FAIL = 확실한 재설계 필요 (비대칭).

### 1-1. 최소 실행 가능 코어 (`src/hi_em/`)
- [x] `embedding.py` — `bge-base-en-v1.5` L2-norm wrapper (768dim, lazy torch import)
- [x] `topic.py` — Topic 클래스 (centroid μ + diag σ² + Welford + cold start $\sigma_0^2$ + $\sigma_\min^2$ floor)
- [x] `scrp.py` — `sticky_crp_unnormed(counts, prev_k, alpha, lmda)` (SEM2 동치)
- [x] `sem_core.py` — `HiEMSegmenter.assign(s)` → `(k, is_boundary)`. 옵션 A centroid independence 근거로 SEM2 restart-vs-repeat 분기와 λ/2 halving은 **미포팅** (docstring 명시)

### 1-2. 단위 테스트 (18/18 passing, 0.89s)
- [x] `tests/test_scrp.py` — 7 tests: 첫 턴, stickiness, Hi-EM α/λ 반전, SEM2 default, 용량 가득, 입력 불변
- [x] `tests/test_topic.py` — 6 tests: Welford vs `np.mean`/`np.var`, cold start 복귀값, variance floor, log-lik 최대, PE zero@centroid, 입력 불변
- [x] `tests/test_sem_core.py` — 5 tests: 첫 턴 topic 0, stickiness 유지, 3 cluster 복원, boundary flag, 재현성

### 1-3. TopiOCQA dev 측정
- [x] `scripts/run_topiocqa_segmentation.py` — dev 2514 turns / 205 conv 전체 예측
- [x] **Metric: topic shift F1** — Ground truth `Topic` 필드 변화만, `Topic_section`은 noise(FP)
- [x] **Baseline 3종 비교**: (a) all-boundary / (b) cosine threshold sweep / (c) Hi-EM
- [x] **Latency 측정**: 턴당 추가 시간
- [x] 결과 기록: `outputs/phase-1-topiocqa.md`
- [x] **한계 명시**: TopiOCQA 12턴 평균 → variance 학습 기회 거의 없음, centroid 부분만 실측

**실측 결과 (HP 튜닝 후)**:
- all-boundary F1 0.451 / cosine(θ=0.70) F1 0.467 / **Hi-EM F1 0.471** (α=10, λ=1, σ₀²=0.1)
- Latency: 19.60 (embed) + 0.29 (assign) ≈ 19.89 ms/turn
- 탐색 이력 3 iter (`scripts/run_topiocqa_{sweep,variants,multisignal}.py`) → `outputs/phase-1-topiocqa.md §탐색 이력`
- **발견**: TopiOCQA는 frequent-shift regime → SEM2 defaults 유리. Hi-EM persistence 기본값은 LongMemEval 등에서 유지, regime-split은 `context/02-math-model.md`에 기록.

### 1-4. Gate 판정 + 분기
- [x] **PASS 조건**: Hi-EM F1 > cosine (0.471 > 0.467) AND F1 > 0.4 AND overhead ≤ 200ms
- [x] **Gate 결과: PASS (marginal)** — TopiOCQA는 frequent-shift regime, Hi-EM 주 타깃 아님
- [x] hyperparam regime split 확정 → `06-decision-log.md` 2026-04-24 entry

### 1-5. TIAGE dev/test topic-shift 평가 (Phase 1-3 확장)

**근거**: LongMemEval는 turn-level topic label이 없어 topic 경계 감지 평가엔 부적합 (04-benchmarks.md 참조). TopiOCQA는 factoid QA로 편향. **PersonaChat 기반 TIAGE**를 추가 평가해 Hi-EM segmentation의 chit-chat 대화에서의 성능을 독립 벤치마크로 검증한다.

- [x] `scripts/run_tiage_segmentation.py` — test 100 conv / 1564 turns 예측
- [x] **Metric: topic shift F1** — label `'1'` = shift, `'0'` = continue (인간 annotated, Cohen's Kappa 0.48)
- [x] **Baseline 3종 비교**: (a) all-boundary / (b) cosine threshold sweep / (c) Hi-EM
- [x] **HP 두 regime 병행**: persistence (α=1, λ=10, σ₀²=0.01) vs freq-shift (α=10, λ=1, σ₀²=0.1)
- [x] 결과 기록: `outputs/phase-1-tiage.md`
- [x] **Gate 조건**: TopiOCQA와 동일 (baseline 대비 우위 AND F1 > 0.4 AND latency +20% 이내)

**실측 결과 (2026-04-25, Colab A100)**:
- all-boundary F1=0.354 / **cosine(θ=0.525) F1=0.421** / Hi-EM persistence F1=**0.317** / Hi-EM freq-shift F1=**0.377**
- Latency: 0.73 ms/turn (overhead PASS)
- **Gate FAIL**: Hi-EM 두 HP 모두 cosine baseline에 명확히 패배 (Hi-EM F1 < 0.421, F1 < 0.4)

### 1-6. Phase 1 종합 Gate

- [x] TopiOCQA gate: 1-4 기준 PASS (marginal, F1=0.471 vs cosine 0.467)
- [x] TIAGE gate: 1-5 기준 **FAIL** (Hi-EM ≤ cosine, F1 < 0.4)
- [ ] **두 gate 모두 PASS** → Phase 2 진입 — **불충족**
- [x] 둘 중 하나라도 FAIL → `06-decision-log.md` append, 옵션 A 재검토 or regime-split HP 확장 — **현재 결정 대기 (report.md §7 옵션 A~E 참조)**

**종합 Gate: FAIL.** 후보 5종(`report.md §12`) 중 #1 종료, 권장 경로 = #5 → #3:
1. [x] TIAGE HP sweep (108 configs, 2026-04-25) — **best F1=0.383 (α=10, λ=3, σ₀²=0.1) < cosine 0.421, 두 Gate 조건 모두 FAIL**. → "어떤 HP로도 못 넘는다" 결정적 증거. `outputs/phase-1-tiage-sweep.json`
2. [ ] Hi-EM likelihood를 cosine-vs-last-turn으로 교체 (옵션 A 변형) — 보류
3. [ ] Phase 2 reframing 진입 — boundary F1 ≠ Hi-EM 핵심 가치, downstream QA로 정직 검증 ⭐
4. [ ] 옵션 D escalation (multi-signal 재설계) — TopiOCQA 전례상 효과 약함, 보류
5. [x] Hi-EM clustering 품질 측정 (V-measure / NMI / ARI, 2026-04-25) — **가설 반박**: 모든 metric에서 cosine 우위. **새 발견**: Boundary F1 ↔ ARI trade-off — persistence HP (α=1) ARI=0.398/0.397 > freq-shift HP (α=10) ARI=0.187/0.314. **메모리 시스템 관점에선 persistence HP 적합** → Phase 2 HP 선택 근거 확보. `outputs/phase-1-clustering-quality.md`

---

## Phase 2: 메모리 계층 (LTM + Memory window)

**핵심 구도**:
- **LTM** = SSD 파일 공간. 모든 턴 원문 + topic 메타 영속 저장.
- **Memory window** (= STM) = 현재 라운드에서 LLM 호출 전 prefill할 턴들. LTM에서 선별해 승격.

### 2-1. LTM 저장 포맷 결정 (2026-04-25 완료)
- [x] **확정: per-conversation JSONL (turn append-only) + `.state.json` (topic latest snapshot, overwrite)**
- [x] 디렉토리: `data/ltm/<conv_id>.{jsonl,state.json}` (gitignored)
- [x] Turn 스키마: `{turn_id, ts, role, text, embedding[768], topic_id, is_boundary}`
- [x] Topic state 스키마: `{conv_id, n_turns, topics: [{topic_id, centroid, variance, count, first_turn_id, last_turn_id}]}`
- [x] **HP 채택: persistence (α=1, λ=10, σ₀²=0.01)** — 옵션 5 ARI/completeness 우위 근거
- 자세한 trade-off 분석: `context/01-hi-em-design.md §9.1`

### 2-2. LTM write/read API (2026-04-25 완료)
- [x] `src/hi_em/ltm.py` — `LTM.{append_turn, update_state, load_turns(topic_id?), load_state, list_conversations}`
- [x] `tests/test_ltm.py` — 8 tests passing (missing/append-order/topic-filter/state-overwrite/state-turns-independence/conv-isolation/list/lazy-mkdir)
- [x] 전체 테스트 회귀 26/26 PASS

### 2-3. Memory window 구성 (2026-04-25 완료)
- [x] `src/hi_em/memory_window.py` — `select_memory_window(q, ltm, conv_id, k_topics, k_turns_per_topic)` baseline policy: cosine top-k topics × recency top-k turns/topic, flatten by turn_id
- [x] `tests/test_memory_window.py` — 8 tests passing (empty/no-topics/single/top-k cosine/recency truncation/k>available/cross-topic order/topic isolation)
- [x] 전체 테스트 회귀 34/34 PASS

### 2-4 (대기, 우선순위 낮음 — Phase 4 결과로 튜닝)
- [ ] Topic importance 계산 (사용 빈도·최근성·cross-reference 가중치)
- [ ] Topic merge 로직 (centroid cosine threshold 기반, LTM 압축)
- [ ] Memory window 크기 정책 $K_{\text{window}}$ (고정 vs 적응적)
- 이유: 현재 baseline policy(cosine + recency)가 단순해 보일 수 있으나, **Hi-EM의 진짜 차별화는 Phase 4 downstream QA에서 4-way baseline 비교(Sliding/Full/RAG/Hi-EM) 시 측정**. 미리 importance/merge 튜닝하면 over-engineering. Phase 4 결과로 어떤 정책 조정이 ROI 높은지 판단 후 진행.

### 2-Full (2026-04-27 구현 완료) — STM + Round + Importance
- [x] **P2F-1**: `src/hi_em/topic_importance.py` (4 작용: 강화·빈도·망각·연결) — 13 tests
- [x] **P2F-2**: `MemoryWindow` 클래스 (`src/hi_em/memory_window.py`) — topic-atomic, threading.RLock — 20 tests
- [x] **P2F-3**: `RoundProcessor` (`src/hi_em/round_processor.py`) — async daemon thread, mention log + neighbor weights — 13 tests
- [x] **P2F-4**: `HiEM(use_stm=True, round_size=10, ...)` STM-first 분기 + round trigger + in-sync turn append — 10 tests + 회귀 13/13
- [x] **P2F-5/6**: 통합 smoke (`scripts/smoke_test_full_pipeline.py`) — 25 turn, A↔B 토픽 인터리브, atomicity / 트리거 / cap / revisit-hit invariant pass
- [x] Sanity: stratified 30Q × `hi-em-full` × LongMemEval oracle — error 0/30, 78s, revisit_hit 0.40
- [ ] **P2F-7**: 5-method (sliding/full/rag/hi-em/hi-em-full) sanity 30 비교 → 결정 분기 (full 500 / HP sweep / 다른 dataset / 정직 reframing)

---

## Phase 3: 오케스트레이션

### 3-1. LLM adapter (2026-04-25 완료)
- [x] `src/hi_em/llm.py` — `OpenAIChatLLM(api_key, base_url)` + `chat(messages, model, **kwargs) -> str`
- [x] OpenAI-compatible (OpenRouter / vLLM / OpenAI 본가 모두 동일 SDK)
- [x] env var: `OPENAI_API_KEY` + `OPENAI_BASE_URL` (생성자 인자 우선, env fallback)
- [x] `tests/test_llm.py` — 5 tests passing (mock OpenAI client; 실 API 호출은 Step 3-3 smoke test에서)
- [x] requirements.txt: `openai>=1.30` 활성화. 백엔드 결정 근거: `memory/project_llm_backend.md`

### 3-2. orchestrator (2026-04-25 완료)
- [x] `src/hi_em/orchestrator.py` — `HiEM(conv_id, encoder, llm, model, ltm_root, alpha=1, lmda=10, sigma0_sq=0.01, k_topics=3, k_turns_per_topic=5, system_prompt=None, **llm_kwargs)`
- [x] `handle_turn(user_text) -> str` 7단계: embed → segment → snapshot → memory_window → messages → llm.chat → append user/assistant
- [x] 순서 결정: select 시점에 user turn은 LTM에 미저장 → 직전 user 필터링 불필요. user/assistant turn은 LLM 응답 후 함께 append. assistant는 embedding=None, 직전 user의 topic_id 상속.
- [x] system_prompt 옵션 인자 (caller가 매 턴 messages에 끼우지 않아도 됨)
- [x] `tests/test_orchestrator.py` — 9 tests passing (FakeEncoder + mock LLM):
  single-turn write / topic-change boundary / state snapshot / first-turn messages /
  system_prompt prepend / second-turn prefill includes first / **A→B→A 토픽 복귀 시 첫 A turn 복귀** /
  llm_kwargs forwarding / ltm files at root
- [x] **48/48 PASS**. 토픽 복귀 prefill이 단위 레벨에서 검증됨 → Hi-EM 핵심 가치(같은 토픽 메모리 호출) 작동 확인.

### 3-3. End-to-end smoke test (2026-04-25 완료)
- [x] `scripts/smoke_test_orchestrator.py` — A→B→A 시나리오 (Kyoto·pasta·Kyoto), `.env` (python-dotenv) 자격증명, `--model` CLI 인자, `<think>` strip 옵션
- [x] **vLLM 로컬 + Qwen/Qwen3-8B로 PASS** (`outputs/phase-3-smoke.md`)
  - Turn 1·3 same topic_id=0, Turn 2 boundary 정확
  - Turn 3 LLM 응답이 Turn 1 정보(가을 시기) 명시 인지 → memory window 작동 확인
- [x] **`response_filter` 옵션 추가** (`HiEMSegmenter.handle_turn` → `HiEM.__init__`): caller에 raw 응답, LTM에 filtered 저장. Qwen-style `<think>` 블록이 prefill 토큰 낭비 안 됨. test 1개 추가, 전체 49/49 PASS.
- [x] tokenizers fork 경고 silence (smoke_test 스크립트에서 `TOKENIZERS_PARALLELISM=false`)

### 3-N (대기)
- [ ] 비동기 라운드 처리(merge · importance 재계산) — Phase 4 결과로 우선순위 결정
- [ ] 세션 간 segmenter 상태 복원 (현재 미지원: state.json 쓰기만 하고 읽기는 없음 — Phase 5 필요 시 추가)

---

## Phase 4: 전체 평가 — QA accuracy 4-way baseline 비교

**실제 downstream 유용성 평가**. Hi-EM이 단순 retrieval보다 낫다는 증거 수집. **Phase 1-6 reframing의 정량 검증 단계** — boundary F1·ARI 모두 cosine에 패배했어도 downstream QA에서 우위면 Phase 5 진입.

### 4-1. LongMemEval 데이터 (사용자 다운로드)
- benchmark clone (2026-04-25): `benchmarks/LongMemEval/` (gitignored)
- HF data 다운로드 명령 사용자에게 전달, oracle 우선 (subset sanity)

### 4-2. `HiEM.preload_history` (2026-04-25 완료)
- [x] orchestrator에 메서드 추가 — history user/assistant 미리 LTM 주입 (segmenter는 user만 통과, assistant는 직전 user의 topic 상속)
- [x] `tests/test_orchestrator.py` 2 tests 추가 (preload + preload→handle_turn 통합) → **51/51 PASS**

### 4-3. 4-baseline 통합 스크립트 (2026-04-25 완료)
- [x] `scripts/run_longmemeval.py --method {sliding,full,rag,hi-em}`
  - sliding: 직전 K turn (default K=20)
  - full: 전체 history 그대로
  - rag: bge cosine top-K (default K=10), chronological 정렬
  - hi-em: `preload_history` + `handle_turn`, persistence HP (α=1, λ=10, σ²=0.01)
- [x] `<think>` strip 적용 (Qwen3-8B 등 reasoning model 대응)
- [x] 출력: hypothesis jsonl `{question_id, hypothesis, method, model}`

### 4-4a. W&B logging (2026-04-26 완료)
- [x] `src/hi_em/eval_logging.py` — `WandbRun` (no-op if `WANDB_API_KEY` 미설정), `count_prefill_tokens` (Qwen tokenizer chat template, 정확), `aggregate_summary` (avg/p50/p95 + by_qtype + error_rate)
- [x] `HiEM.handle_turn(return_debug=True)` — prefill/messages 외부 노출 (Phase 4 메트릭 계산용)
- [x] `run_longmemeval.py` baseline 함수 4개 모두 `(response, messages, extras)` tuple 반환. `count_prefill_tokens(messages, model)`로 정확한 token count. Hi-EM은 `topic_revisit_hit` 추가.
- [x] `judge_longmemeval.py` sidecar `<output>.wandb-run-id`로 같은 run에 accuracy resume 추가.
- [x] **codex review 반영**: 빠진 metric (`topic_revisit_hit_rate`, `error_or_empty_rate`) 추가, over-designed metric 4개 (`boundary_count`, `n_topics`, `query_assigned_topic_count`, `selected_in_window`) drop, latency histogram → scalar p50/p95, run-per-method 구조.
- [x] `tests/test_eval_logging.py` 4 tests + `tests/test_orchestrator.py` 1 test (return_debug). 전체 **56/56 PASS**.

### 4-4. Judge 스크립트 (2026-04-25 완료)
- [x] `scripts/judge_longmemeval.py` — LongMemEval 6 prompt template 인용 (MIT License Copyright 2024 Di Wu, `benchmarks/LongMemEval/LICENSE`)
- [x] **Judge model = Qwen/Qwen3-8B** (응답 생성과 동일 vLLM endpoint, 비용 0)
- [x] `temperature=0`, `max_tokens=20`, judge raw에서 `<think>` strip 후 첫 token이 `yes`면 정답
- [x] question_type별 + abstention별 accuracy 분리 집계

### 4-5 (대기). Subset sanity (사용자 실행)
- [ ] 30 questions × 4 method × oracle → 8 명령 (run × 4 + judge × 4)
- [ ] 응답 형식 확인, judge 정확성 sanity, latency 측정

### 4-6 (대기). 전체 평가 (사용자 실행)
- [ ] oracle 500 + s/m (필요 시 추가 다운로드) × 4 method
- [ ] question_type 5축 별 결과 표
- [ ] `outputs/phase-4-summary.md` 작성

### 4-7 (대기). Phase 5 진입 판정
- [ ] Hi-EM이 4-baseline 중 우위 (특정 능력에서? 토큰 효율?)
- [ ] report.md 갱신, 결정 분기 (Phase 5 논문 vs 추가 개선)

---

## Phase 4-Re: research-experiment-infrastructure 적용 (2026-04-27)

**목적**: Phase 4 baseline 결과 (Hi-EM 0.562 < baseline) 기반으로 다음 실험들 (HP sweep / 추가 method / 다른 dataset)이 **resumable + atomic** 인프라 위에서 진행되도록. SKILL `research-experiment-infrastructure` (`~/.claude/skills/...`) 적용.

### Skill → Hi-EM 매핑
- Experiment = (method, HP, dataset) 평가 1회
- Round = N questions (default 50, oracle 500 → 10 rounds)
- Phase = (1) run hypothesis, (2) judge accuracy
- working_state = stateless (Phase 2-Full STM 도입 시 LTM/STM 추가)
- Replay 인프라 = **폐기** (정적 dataset + stateless eval, SKILL §7 분기 조건으로 불필요)

### R-1 ~ R-11

- [x] **R-1 Archive** — `archive/2026-04-26-baseline/{outputs, ltm, README.md}`. 4 HP × sanity/full 표 + sample noise + lost 측정 명시.
- [x] **R-2 atomic_io** — `src/hi_em/atomic_io.py` (save_json, load_json, append_jsonl, load_jsonl). utf-8 surrogate-safe.
- [x] **R-3 experiment lifecycle** — `src/hi_em/experiment.py`. `ExperimentMeta`, `create_experiment` (idempotent), `mark_round_complete` (summary→checkpoint 순서), `mark_experiment_complete`, `find_resumable_experiment`, `sanity_check_summary`, `Session`. 17 unit tests.
- [x] **R-4 resume** — `find_resumable_experiment` + `completed.json` 체크 분리. SKILL §5/§9.7 따름.
- [x] **R-5 run_experiment.py** — 단일 entry. round 단위 atomic + resume + session 지원 + experiment-level summary 디스크 저장 + wandb sidecar resume + sanity check 자동.
- [x] ~~**R-6 replay**~~ — 폐기 (SKILL §7 분기로 정적 dataset에선 불필요).
- [x] **R-7 자동 resume invariant test** — `tests/test_run_experiment.py` 5 tests. **SKILL §10 #13 reference vs interrupt+resume primary_metric 정확 일치 invariant 자동화**.
- [x] **R-9 실 vLLM smoke test** — `--exp-id smoke-resume-kill` 5 questions × 2 rounds. round 사이클 + idempotent 재호출 + wandb sidecar 동작 확인.
- [x] **R-10 docs cascade** — handoff/plan/README/decision-log/03-architecture 갱신.
- [x] **R-11 인프라 재현성 검증** (2026-04-27 sanity 단계 통과) — sanity 30 × 4 method 결과 archive set #2와 모든 cell-level Δ가 sample noise (`temperature=0.7`, 5/qtype) 영역 안. Overall Δ ≤ -0.10. 4 method **상대 ranking 그대로** (full > sliding ≈ rag > hi-em) + Hi-EM **multi-session 약점 패턴 그대로** (0.00~0.20). 인프라 systematic bias 없음 확정. Full 500 재현은 시간 절약 (sanity 검증 충분 판정).

### Metric 수집 정책 (Phase 4-Re)

매 experiment 종료 시 디스크에 **2 level summary 저장**:
1. **Round level**: `rounds/round_NNN/summary.json` (그 round의 questions만 aggregate, sanity_check 자동 호출)
2. **Experiment level**: `{exp_dir}/summary.json` (모든 round의 judged 합쳐 aggregate, wandb summary와 동일)

User 비교 표 형식 그대로 stdout 출력 (Overall + 6 question_type accuracy + token/latency p50/p95 + topic_revisit_hit_rate).

### W&B 통합 (SKILL §8 sidecar pattern)
- run name = exp_id (sidecar `wandb-run-id.txt`로 resume 시 같은 run 이어짐)
- name/group은 resume 시 다시 보내지 않음 (덮어쓰기 회피)
- step axis = round_num (define_metric 처리)
- 자격증명: WANDB_API_KEY env or `wandb login` (~/.netrc) 둘 다 인정

4 baseline QA accuracy 비교 (LoCoMo · LongMemEval):
- **Baseline 1. Sliding window**: 최근 N개 세션만 LLM에 투입. 오래된 정보 놓침.
- **Baseline 2. Full context**: 모든 세션 투입. Context length 초과·비용 폭발.
- **Baseline 3. RAG (embedding retrieval)**: 쿼리와 cosine-similar top-K 턴. Topic 구조 없는 flat retrieval.
- **Hi-EM**: Topic 단위 조직화 + 관련 topic 턴을 Memory window로 승격 → compact context.

- [ ] LoCoMo QA accuracy (4 baseline × 전체 197 QA/conv)
- [ ] LongMemEval 5개 능력별 accuracy (IE / Multi-session / Temporal / Knowledge update / Abstention; oracle + _s + _m)
- [ ] Latency 누적 측정 (LLM 호출 포함 총 overhead)
- [ ] Memory window 크기 / prefix 토큰 수 효율 측정 (재사용률 등)

---

## Phase 5: 논문 실험 + 정직 reframing

**현재 상황 요약** (2026-04-27 R-11 종결 후):
- Phase 1: boundary F1 → cosine baseline에 패배
- Phase 1-6 (옵션 5): ARI/V-measure → cosine baseline에 패배
- Phase 4: downstream QA (LongMemEval oracle 500 nothink) → **Hi-EM 0.562, 4 method 중 꼴찌** (sliding 0.658 / full 0.712 / rag 0.692)
- Phase 4 sanity (×2 새 infra 재현) → 결과 일관, 인프라 검증 ✓
- Hi-EM **유일 강점**: ssp (single-session-preference) **0.97** — 4 method 중 1위 (full 0.93보다 +0.04)
- Hi-EM 약점: multi-session 0.23, knowledge-update 0.51, temporal 0.46

→ Hi-EM의 **광범위 contribution 가설 반증 완료**. 남은 길은 (a) **좁은 contribution으로 정직 정리** + (b) **다른 시나리오 (긴 history, 다른 도메인) 탐색**.

### 5-A. 정직 reframing 보고서 (시급, 1~2일)
- [ ] `report.md` 갱신 — Phase 0~Phase 4 종합 결론. "Hi-EM의 contribution은 ssp 정도. 다른 axis는 baseline 못 넘음" 정직 기록
- [ ] `outputs/phase-4-final.md` (신규, archive로 이동 가능) — 4 method × 6 qtype × 4 HP × sanity/full 모든 결과 종합 표 + 분석
- [ ] decision-log: Hi-EM 가치 정의 좁히기 결정 entry
- [ ] 논문 plan 결정: (a) ssp 좁은 contribution으로 short paper, (b) 추가 실험 필요, (c) negative result 보고서

### 5-B. 다른 dataset 탐색 (선택, 사용자 결정)
LongMemEval oracle은 짧은 history (evidence sessions only). Hi-EM의 진짜 시나리오는 긴 history + 토픽 복귀. 측정 안 한 곳:
- [ ] LongMemEval s_cleaned (115k token history) — 다운로드 명령 `wget .../longmemeval_s_cleaned.json -P benchmarks/LongMemEval/data/`
- [ ] LongMemEval m_cleaned (500 sessions) — Hi-EM 가장 빛날 시나리오
- [ ] LoCoMo (snap-research) — 197 QA/conv, 다른 도메인

각 dataset에 새 session으로 4 method 비교 — `scripts/run_session.py --data ... --session-id ...`. ±2~6시간/dataset.

### 5-C. Ablation studies (Phase 5-A 후 결정)
- [ ] sCRP prior 기여도 (uniform prior로 교체 vs sticky-CRP)
- [ ] centroid + variance vs centroid-only (variance 기여도)
- [ ] memory_window selection (cosine top-k vs random top-k vs recency only)
- [ ] response_filter (`<think>` strip 효과 정량)

### 5-D. Baseline 비교 (선택, scope creep 위험)
- [ ] MemGPT 비교 (시간 큼)
- [ ] LongMemEval 논문 baseline 재현
- [ ] 우리 4 method (sliding/full/rag/hi-em) 만으로 충분할 수 있음

### 5-E. 결정 분기
Phase 5-A 정직 reframing 후:
- **(a) Short paper / workshop**: ssp 좁은 contribution으로 작성. 빠름.
- **(b) 추가 실험 필요**: 5-B (다른 dataset) → 새 winning region 발견 시 long paper.
- **(c) Negative result + alternative**: Hi-EM은 보류, 다른 algorithm 탐색 (Phase 2-Full STM, 알고리즘 변경 등).
- **(d) Hi-EM 폐기 + insights only**: Phase 1~4 lessons → workshop 또는 thesis chapter

---

## 각 Phase 완료 기준
`outputs/phase-N-results.md` 또는 세부 파일(`phase-1-topiocqa.md`, `phase-1-tiage.md` 등)에 측정 결과 기록.
