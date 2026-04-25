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
├── sem_core.py            # online MAP inference 루프
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

# Phase 2 진입 (2026-04-25), 추가 예정:
#   src/hi_em/ltm.py           — LTM read/write API (per-conv JSONL + state.json, §9.1)
#   src/hi_em/memory_window.py — 현재 query → topic 선별 → prefill prefix
#   src/hi_em/orchestrator.py  — 매 턴 파이프라인 (Phase 3)
#   tests/test_ltm.py
#   tests/test_memory_window.py
# LTM 데이터 위치: data/ltm/<conv_id>.{jsonl,state.json} (gitignored)
# Phase 2+ 진입 시 추가 예정: orchestrator/LTM/memory_window 등

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