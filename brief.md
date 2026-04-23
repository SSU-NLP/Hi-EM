# Hi-EM: Human-inspired Episodic Memory for LLM Conversations

## 목적
Transformer 기반 LLM에 fine-tuning 없이 붙는 **실시간 대화 메모리 관리 시스템**.
인지과학의 Structured Event Memory (SEM, Franklin et al. 2020)를 쿼리-토픽 구조로
재해석하여, 긴 대화에서 **토픽 단위 STM/LTM 관리** 및 **KV cache paging**을 구현한다.

## 핵심 차별점
- **No fine-tuning** — 어떤 Transformer LLM에도 add-on으로 붙음
- **Online topic segmentation** — 매 턴 실시간 판정
- **Sticky-CRP 기반 topic 수 자동 결정**
- **Topic 단위 KV cache 재사용** — latency + memory 동시 최적화

## 평가 벤치마크 (우선순위순)
1. **LoCoMo** — 장기 메모리 (메인, `benchmarks/locomo/`)
2. **LongMemEval** — Chat assistant 장기 기억 (메인, `benchmarks/LongMemEval/`)
3. **TopiOCQA** — 토픽 감지 정밀도 검증 (`benchmarks/topiocqa/`)
4. **TIAGE** — 필요 시 추가 clone

## 참조 자료
- `SEM-paper.pdf` — 원본 논문 (Franklin et al. 2020, Psych Review)
- `SEM/` — **SEM2** (`nicktfranklin/SEM2`) current working build. 알고리즘 참조용, 코드 직접 복사 금지

## 제약
- 메인 LLM fine-tuning 금지
- 턴당 추가 latency +10~20% 이내
- Python 3.10+, PyTorch (TensorFlow/Keras 금지)

## 설계 상태
- **확정**: no fine-tuning, sticky-CRP prior, LTM/STM 계층, memory reconstruction 폐기
- **미확정**: 사건 모델의 구체적 형태 (벤치마크 분석 후 결정)