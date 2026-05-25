# 코드 아키텍처

## 프로젝트 루트
`/home/namchailin/Hi-EM`

## src/hi_em/ (구현 대상, 사건 모델 확정 후 최종 레이아웃 결정)

예상 구조 (사건 모델에 따라 일부 변경 가능):
src/hi_em/
├── init.py
├── config.py              # 하이퍼파라미터
├── embedding.py           # bge encoder wrapper
├── topic.py               # Topic 클래스 (사건 모델 형태에 따라 필드 결정)
├── scrp.py                # sticky-CRP prior
├── boundary.py            # boundary score 계산 (사건 모델에 따라 구현 달라짐)
├── sem_core.py            # online MAP inference 루프 (v2 본진)
├── sem_core_v33{1..8}*.py # 버전별 segmenter (methodology/README 계보 참조).
│                          #   v3.3.5 f_is_trained gating / v3.3.6 persistence+replay+seed
│                          #   (topic_v336.py) / v3.3.7 map_variance σ² / v3.3.8 pe_prior
├── importance.py          # topic importance
├── merge.py               # topic merge
├── ltm.py                 # 장기 메모리
├── stm.py                 # 단기 메모리
├── memory_window.py       # STM 구성 — LTM에서 현재 라운드 prefill 대상 턴 선별/승격
└── orchestrator.py        # 매 턴 파이프라인
# 사건 모델이 엔티티/cue/qtype 등을 사용하는 옵션으로 결정되면 추가:
├── entity.py            # spaCy NER wrapper
├── cue_phrase.py        # regex cue detector
└── question_type.py     # rule-based qtype classifier

## scripts/ (Phase 1 현재 실재)
scripts/
├── check_step_done.py             # Step 완료 검증 게이트 (CLAUDE.md "Step 완료 프로토콜" 2단계)
├── run_topiocqa_segmentation.py   # Phase 1-3 메인 평가 (TopiOCQA dev F1)
├── run_topiocqa_sweep.py          # 108-config HP grid (α × λ × σ₀²) — Phase 1-4 best HP 탐색
├── run_topiocqa_variants.py       # 5가지 구조 변형 비교 (gauss-origin/global/self, vMF-origin/const)
├── run_topiocqa_anchors.py        # 옵션 A 변형: anchor turn 기반 likelihood
├── run_topiocqa_bigencoder.py     # bge-large 인코더 시도 (Phase 1 추가 탐색)
├── run_topiocqa_contextualized.py # contextualized embedding 시도
├── run_topiocqa_multisignal.py    # 옵션 D escalation 탐색 (multi-signal)
├── run_tiage_segmentation.py      # Phase 1-5 TIAGE test 평가 (persistence + freq-shift 두 점)
├── run_tiage_sweep.py             # Phase 1-6 TIAGE 108-config grid (TopiOCQA sweep mirror)
└── run_clustering_quality.py      # Phase 1-6 옵션 5: V-measure/NMI/ARI 측정 (cosine vs Hi-EM 두 HP)

# Phase 2 진입 (2026-04-25). 신규 모듈:
#   src/hi_em/ltm.py             ✅ Step 2-2 (LTM read/write API, per-conv JSONL + state.json, §9.1)
#   src/hi_em/memory_window.py   ✅ Step 2-3 (select_memory_window: cosine top-k topics × recency top-k turns)
#   src/hi_em/llm.py             ✅ Step 3-1 (OpenAIChatLLM — OpenRouter/vLLM/OpenAI 본가 OpenAI-compatible)
#   src/hi_em/orchestrator.py    ✅ Step 3-2 (HiEM.handle_turn — 7단계 파이프라인 + response_filter 옵션)
#   tests/test_ltm.py            ✅ 8 tests
#   tests/test_memory_window.py  ✅ 8 tests
#   tests/test_llm.py            ✅ 5 tests (mock OpenAI client)
#   tests/test_orchestrator.py   ✅ 10 tests (FakeEncoder + mock LLM, 토픽 복귀 + response_filter 검증)
#   scripts/smoke_test_orchestrator.py ✅ Step 3-3 (실 LLM A→B→A; vLLM Qwen3-8B PASS, outputs/phase-3-smoke.md)
#   scripts/run_longmemeval.py   ✅ Step 4-3 (4 baseline: sliding/full/rag/hi-em → hypothesis jsonl)
#   scripts/judge_longmemeval.py ✅ Step 4-4 (LongMemEval prompt 인용, Qwen judge)
#   src/hi_em/orchestrator.py    ✅ Step 4-2 (preload_history 메서드 추가, 51/51 tests)
#   .env.example                 ✅ 협업자 안내 (.env는 gitignored, python-dotenv)
# Phase 4 진행 중 — Step 4-5/4-6 사용자 실행 대기 (subset → 전체)
# 추가 예정:
#   Step 2-4: importance / merge / adaptive K_window (Phase 4 결과 후 튜닝)
# LTM 데이터 위치: outputs/experiments/<name>/<run_label>/results/working_state/ltm/<conv_id>.{jsonl,state.json}
#                  (sweep 안 self-contained, gitignored). 2026-05-08 이전엔 top-level data/ltm/ — 삭제됨.
# LLM 백엔드: memory/project_llm_backend.md (OpenAI-compatible, OpenRouter/vLLM)

# Phase 2-Full (2026-04-27 구현 완료) 신규 모듈:
#   src/hi_em/topic_importance.py   ✅ P2F-1 (compute_importance: 4 작용 강화·빈도·망각·연결)
#   src/hi_em/memory_window.py      ✅ P2F-2 (MemoryWindow class 추가 — topic-atomic STM, threading.RLock)
#   src/hi_em/round_processor.py    ✅ P2F-3 (RoundProcessor — async daemon thread, mention log + neighbor weights)
#   src/hi_em/orchestrator.py       ✅ P2F-4 (HiEM.use_stm 옵션 + STM-first + round trigger + in-sync turn append)
#   src/hi_em/config.py             ✅ configs/hiem.json loader (segmenter / memory_window / topic_importance / stm / round / evaluation)
#   tests/test_topic_importance.py  ✅ 13 tests
#   tests/test_memory_window_class.py ✅ 20 tests (atomicity invariant 강제)
#   tests/test_round_processor.py   ✅ 13 tests
#   tests/test_orchestrator_stm.py  ✅ 10 tests (STM-first + round trigger + in-sync update)
#   scripts/smoke_test_full_pipeline.py ✅ P2F-5/6 통합 smoke (vLLM, 25-turn invariant pass)
#   scripts/run_longmemeval.py / run_experiment.py — `--method hi-em-full` 추가 + STM/round/importance HP CLI

## tests/ (Phase 1 현재 실재, 18 tests passing)
tests/
├── test_topic.py        # Topic 클래스 (Welford 온라인 update + Gaussian likelihood)
├── test_scrp.py         # sticky-CRP prior (SEM2 `_calculate_unnormed_sCRP` 수치 매칭)
└── test_sem_core.py     # HiEMSegmenter MAP 할당 루프 (prior×likelihood argmax + boundary flag)
# Phase 2+ 진입 시 추가 예정: test_orchestrator, test_ltm, test_memory_window

## 진입점 (예상)
```python
from hi_em.orchestrator import HiEM

hi_em = HiEM(config="default", llm_callable=my_llm_fn)
response = hi_em.handle_turn(user_query="...")
```

`orchestrator.handle_turn`은 context 구성만, LLM 호출은 주입된 callable에 위임.

---

## Phase 4-Re 인프라 (2026-04-27 추가, research-experiment-infrastructure skill 적용)

```
src/hi_em/
├── atomic_io.py         # save_json / load_json / append_jsonl / load_jsonl (utf-8 surrogate-safe)
└── experiment.py        # ExperimentMeta · create_experiment · mark_round_complete
                         #   · find_resumable_experiment · sanity_check_summary · Session

scripts/
├── run_experiment.py    # 신규 단일 entry. round 단위 atomic checkpoint + resume + session
└── (legacy) run_longmemeval.py / judge_longmemeval.py / run_phase4_all.py — 점진 deprecation

configs/
└── hiem.json            # 모든 알고리즘 HP single source-of-truth (segmenter / memory_window /
                         # topic_importance / stm / round / evaluation 6 섹션)

archive/2026-04-26-baseline/   # 기존 결과 영구 보존 (outputs/ + ltm/ + README.md)
results/
├── experiments/{exp_id}/      # 새 실험: rounds/ + checkpoints/ + experiment.json + summary.json
└── sessions/{session_id}/     # HP sweep / multi-method 묶음 (tracked, common config)

tests/
├── test_experiment.py        # 17 tests — atomic / lifecycle / sanity_check / Session
└── test_run_experiment.py    # 5 tests — round cycle / mid-crash / resume / idempotent /
                              # SKILL §10 #13 reference vs interrupt+resume invariant
```

**Round = 50 questions** (oracle 500 → 10 rounds). Resume granularity.
**Phase**: (1) run hypothesis (2) judge accuracy. atomic save per phase, checkpoint after both.
**Metric**: per-method × per-qtype accuracy + prefill_tokens/latency p50/p95 + error_rate + topic_revisit_hit_rate.
## methods/ (2026-05-20 신설)

baseline 의 원본(offline)·Hi-EM 수정본(online, prefix-causal) 진입점 정리.
범위: TextTiling, BayesSeg. 방식 A(wrapper, benchmarks 무복사·read-only 유지).
- `methods/texttiling/{offline,online}.py`, `methods/bayesseg/{offline,online}.py`, `methods/README.md`
- 이후 확장: `methods/greedyseg/online_delay2.py`, `methods/graphseg/online_window.py`
- offline=전체대화(원본 알고리즘 호출), online=`scripts/run_*_prefix.py`(검증본) 실행 진입점
- 동일 harness: Def-DTS 번들 데이터 + autoseg Pk/WD/F1 + Score. online=AUXILIARY(codex). 산출 outputs/experiments/<name>/REPORT.md
- **`methods/RoBERTa/{offline/train.py, online/segment.py}`** (2026-05-23 신규)
  — supervised RoBERTa 분절기 (Coldog2333/SuperDialseg EMNLP 2023 Table 3
  `RoBERTa` 충실 재현). `offline` = 학습+평가, 경계마다 미래 포함 ~20 윈도우
  logit 평균. `online` = offline 체크포인트 재사용, 추론만 strict causal —
  경계 (t-1,t) 를 turn t 시점 causal 윈도우 하나로 1회 결정 (미래 0).
  harness 예외 — SuperDialseg 번들 데이터 + 논문 official Pk/WD metric.
  결과 → `outputs/experiments/2026-05-23_roberta_{supervised,online}/`.
decision-log 2026-05-20 참조.

## src/hi_em/ segmenter 모델 (2026-05-23 갱신)

```
src/hi_em/
├── hi_dots.py       # HiDoTS — 현 main DTS 모델 (v4.1.x reduced form, commit 326b86b)
└── hi_dots_v2.py    # HiDoTSV2(HiDoTS) — lexical-overlap 보정 변형 (검증 대기, 2026-05-23)
```

- `HiDoTSV2` = `HiDoTS` + TextTiling 식 단어-빈도 겹침 보정항. `w_lex=0` 시 v1 과
  byte-parity. 설계·결과 → `context/methodology/hi-dots-v2.md`, decision-log 2026-05-23.
- 실험 entry: `scripts/run_hidots_v2.py` → `outputs/experiments/2026-05-23_hidots_v2/`.
- 현재 main 모델은 `HiDoTS` 유지 — `HiDoTSV2` 는 v1 대체 승격 보류 (검증 대기).
