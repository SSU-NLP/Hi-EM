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
├── kv_cache.py            # KV cache 인터페이스 (초기 stub)
└── orchestrator.py        # 매 턴 파이프라인
# 사건 모델이 엔티티/cue/qtype 등을 사용하는 옵션으로 결정되면 추가:
├── entity.py            # spaCy NER wrapper
├── cue_phrase.py        # regex cue detector
└── question_type.py     # rule-based qtype classifier

## scripts/
scripts/
├── analyze_benchmarks.py  # Phase 0 벤치마크 분석
├── prepare_benchmarks.py  # 데이터 전처리
├── run_locomo.py
├── run_topiocqa.py
├── run_longmemeval.py
└── analyze_results.py

## tests/
tests/
├── test_topic.py
├── test_scrp.py
├── test_sem_core.py
└── test_orchestrator.py

## 진입점 (예상)
```python
from hi_em.orchestrator import HiEM

hi_em = HiEM(config="default", llm_callable=my_llm_fn)
response = hi_em.handle_turn(user_query="...")
```

`orchestrator.handle_turn`은 context 구성만, LLM 호출은 주입된 callable에 위임.