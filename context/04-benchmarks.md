# 벤치마크 정리

## Tier 1 (메인): 장기 대화 메모리

### LoCoMo
- 경로: `benchmarks/locomo/`
- 데이터: `benchmarks/locomo/data/locomo10.json` (레포 내 포함)
- 구조: 10개 대화, multi-day 세션 구성
- 평가: QA accuracy, event summarization
- 논문: ACL 2024
- 레포: https://github.com/snap-research/locomo

### LongMemEval
- 경로: `benchmarks/LongMemEval/`
- 데이터: HuggingFace (레포 README 참조)
- 구조: 500 질문, 5개 메모리 능력
  1. Information extraction
  2. Multi-session reasoning
  3. Temporal reasoning
  4. Knowledge updates
  5. Abstention
- 평가: QA accuracy (GPT-4o judge)
- 논문: ICLR 2025
- 레포: https://github.com/xiaowu0162/LongMemEval

## Tier 2 (토픽 감지 검증)

### TopiOCQA
- 경로: `benchmarks/topiocqa/`
- 데이터: `python download_data.py --resource data.retriever.all_history.dev`
- 구조: 3920 대화, 평균 13턴, 4 topic
- Topic 정의: Wikipedia document 경계
- 평가: topic shift detection F1
- 레포: https://github.com/McGill-NLP/topiocqa

### TIAGE (옵션)
- 레포: https://github.com/HuiyuanXie/tiage
- 구조: 500 annotated 대화, shift 3.5회/대화
- Cohen's Kappa = 0.48
- 필요 시 추가 clone

---

## 데이터 준비 체크리스트
- [x] LoCoMo clone
- [x] TopiOCQA clone
- [x] LongMemEval clone
- [ ] LoCoMo 데이터 확인 (레포 내 포함)
- [ ] TopiOCQA data download
- [ ] LongMemEval HF data download
- [ ] TIAGE clone (옵션)

---

## 데이터 분석 체크리스트 (Phase 0 완료 기준)

각 벤치마크별 필수 분석:
- [ ] 샘플 1~2개 JSON 구조 직접 확인
- [ ] 평균/중간값/최대값: 세션 수, 턴 수, 쿼리 토큰 길이
- [ ] Topic 전환 패턴 (명시 annotation 여부, 전환 빈도, 유형)
- [ ] Claude-유사 대화(코딩/글쓰기/브레인스토밍)와의 유사성 평가
- [ ] 해당 벤치마크에서 **어떤 사건 모델이 유리할지** 판단

분석 결과 → `outputs/benchmark-analysis.md`

---

## 주의: 벤치마크별 bias

각 벤치마크는 서로 다른 대화 성격을 가진다:
- **TopiOCQA**: Wiki 기반 정보 검색 QA. Named entity 풍부.
- **TIAGE**: PersonaChat 기반 chit-chat. Cue phrase 많음.
- **LoCoMo**: Multi-day 긴 대화. 세션 경계 명확.
- **LongMemEval**: Chat assistant 시뮬레이션. Needle-in-haystack 구조.

**한 벤치마크의 특성만 보고 설계를 고정하지 마라.** 여러 벤치마크에서 공통으로 유효한 신호를 찾거나, 벤치마크별 적응형 설계를 고려할 것.