# 미확정 질문

## 최우선: 사건 모델 (Phase 0 완료 후 결정)
`context/01-hi-em-design.md`의 "미확정 사항 A" 참조.

---

## 설계 차원

1. Multi-signal 가중치 튜닝 방법 (사건 모델에 신호 앙상블 포함 시)
   - A: grid search on dev set
   - B: 학습 가능한 linear layer
   - C: 고정값 시작 + 실험 조정

2. Entity extractor (사건 모델이 엔티티 사용 시)
   - spaCy `en_core_web_sm` (가벼움 ~5ms)
   - spaCy `en_core_web_trf` (정확 ~20ms)
   - 간단한 noun phrase chunker

3. Memory window → LLM 전달 수준
   - vLLM/SGLang 등 prefix caching API 실제 통합
   - Hi-EM은 Memory window(선별 턴 리스트)만 반환, 실제 prefill은 downstream이 담당
   - 프로파일링·시뮬 측정만 (실제 prefill 생략)

4. Topic importance 학습화 여부
   - 지금은 heuristic
   - 나중에 RL or 사용자 feedback

---

## 기술 차원

5. LTM 저장 형식
   - JSON file
   - SQLite
   - Parquet

6. 응답 생성 LLM
   - OpenAI API
   - Anthropic API
   - 로컬 HF 모델

---

## 평가 차원

7. Baseline 비교 대상
   - 단순 sliding window
   - MemGPT
   - RAG with cosine-only retrieval
   - 단순 sCRP (boundary score 없이)

---

## 마감 기한
Phase 1 코드 작성 전 최소 사건 모델 + 5 + 6은 결정 필요.