# 벤치마크 데이터 분석 (Phase 0 Step 0-2)

**목적**: 세 벤치마크(LoCoMo / TopiOCQA / LongMemEval)의 실제 데이터 구조·통계·topic 전환 패턴을 직접 열어 확인하고, 이를 근거로 Hi-EM의 사건 모델 옵션(A~F)을 Step 0-3에서 선택하기 위한 증거를 수집한다.

데이터 출처
- LoCoMo: `benchmarks/locomo/data/locomo10.json` (레포 내 포함)
- TopiOCQA dev: `benchmarks/topiocqa/download_data.py --resource data.topiocqa_dataset.dev`로 다운로드 → `benchmarks/topiocqa/downloads/data/topiocqa_dataset/dev.json`
- LongMemEval oracle: HuggingFace `xiaowu0162/longmemeval-cleaned` → `benchmarks/LongMemEval/data/longmemeval_oracle.json`

---

## 1. LoCoMo

### 구조
- top-level: list[dict], 길이 10 (10개 long-term 대화)
- 각 conversation의 키:
  - `conversation`: 세션들의 dict. key 패턴 `session_N`, `session_N_date_time` (N=1..) — **세션 경계는 날짜 기반**
  - `qa`: list of {question, answer, evidence, category}
  - `event_summary`, `observation`, `session_summary`, `sample_id`
- turn 구조: `{speaker, dia_id, text}`. `dia_id`는 `"D1:3"` 형태 (세션:턴 인덱스)
- **명시적 topic annotation 없음** — topic 경계 ≈ session 경계(날짜) 로 간접 정의

### 통계 (10개 대화 aggregate)
| 지표 | 값 |
|---|---|
| conversations | 10 |
| sessions 합계 | 272 (평균 27.2/대화, min 19, max 32) |
| turns 합계 | 5882 (평균 588/대화) |
| turns/session 평균 | 20~24 |
| turn text 길이 | 평균 108~142자 (chat style) |
| QA | 총 1986 (평균 199/대화) |
| QA category 분포 | 1: 282, 2: 321, 3: 96, 4: 841, 5: 446 |

### 특징
- **Multi-day 대화**: 세션별 `date_time` 필드. 날짜 간격 = 자연스러운 사건 경계 후보.
- **긴 누적 문맥**: 대화당 평균 588턴, 최대 689턴. SEM이 originally 비디오 30~100 scene 수준을 다룬 것과 규모 차이 큼.
- **QA evidence**: `evidence` 필드가 `["D1:3"]` 같은 dia_id 포인터 → 특정 턴이 답 근거.
- **QA 카테고리 4가 약 42%**: 한 카테고리가 dominant — 벤치마크의 평가 편향을 인지해야.

### Hi-EM 시사점
- 명시적 topic annotation이 없으므로 **unsupervised segmentation**이 그대로 유효. sCRP + MAP 루프 적용 자연스러움.
- 세션(날짜) 경계 ≈ 강한 prior 변화 신호. Hi-EM의 $\lambda$(stickiness) 큰 값과 **session boundary 외부 트리거**를 결합하면 인간이 라벨링한 날짜 경계를 자동으로 회수 가능한지 실험 가치 있음.
- 턴 길이가 LongMemEval보다 짧고 TopiOCQA보다 김 → 일상 대화 수준. **centroid 기반 사건 모델(옵션 A)이 과도하게 단순하지 않은지 검증 필요**.

---

## 2. TopiOCQA

### 구조
- top-level: list[dict], 길이 2514 (dev 턴 수)
- turn 구조:
  - `Conversation_no`, `Turn_no`
  - `Question`, `Answer`, `Rationale`
  - **`Topic`**: Wikipedia document 이름 (e.g., "Dunkirk (2017 film)")
  - **`Topic_section`**: 섹션 (e.g., "Release")
  - `Context`: 이전 턴들 요약
  - `is_nq`: Natural Questions 기원 여부
  - `Additional_answers`: 평가용 대체 답안 여러 개

### 통계 (dev aggregate)
| 지표 | 값 |
|---|---|
| turns 총합 | 2514 |
| conversations | 205 |
| turns/conv 평균 | 12.3 (min 10, max 16, median 12) |
| distinct topics/conv 평균 | 3.7 |
| topic shifts/conv 평균 | 3.3 |
| section shifts/conv 평균 | 7.7 |
| Question 길이 | 평균 37자 (factoid) |
| unique topic docs | 734 (Wikipedia 페이지) |

### 특징
- **명시적 topic ground truth**: Wikipedia document 단위로 topic 정의 → **topic shift F1 측정 가능**. Hi-EM의 sCRP 기반 판정을 직접 검증할 수 있는 유일한 벤치마크.
- **짧은 대화, 잦은 전환**: 12턴 평균에 3.3 shift → 약 3~4턴마다 topic 변경. 다른 벤치마크보다 transient.
- **Factoid 쿼리**: 37자 평균. 임베딩 공간의 anchor가 "topic document" 하나에 강하게 묶임 → centroid 기반 분리도 잘 될 것으로 예상.
- **section 전환이 topic 전환보다 2배 많음**: 같은 topic 내 세부 hop이 자주 일어남 → Hi-EM이 "section" 수준 shift를 topic shift로 오인할 위험 있음.

### Hi-EM 시사점
- TopiOCQA는 **sCRP prior 자체의 segmentation 성능 검증용**. $\alpha$, $\lambda$ 튜닝 타겟 1순위.
- 사건 모델이 entity/document 중심으로 잘 구분되므로 **옵션 A(centroid only)도 TopiOCQA에선 충분**할 가능성.
- 단 Wiki QA는 Claude-유사 대화와 성격이 다름 → TopiOCQA만 보고 사건 모델 고정하면 편향 (handoff.md 경고 그대로).

---

## 3. LongMemEval (oracle)

### 구조
- top-level: list[dict], 길이 500 (test questions)
- question 구조:
  - `question_id`, `question_type`, `question`, `answer`
  - `question_date`: 질문 시점 (e.g., "2023/04/10 (Mon) 23:07")
  - `haystack_dates`: evidence session 날짜 리스트
  - `haystack_session_ids`: session ID 리스트
  - **`haystack_sessions`**: 각 session = list of turns `{role: user|assistant, content, has_answer}`
  - `answer_session_ids`: 답을 포함하는 session ID

### 통계 (oracle aggregate)
| 지표 | 값 |
|---|---|
| questions | 500 |
| sessions/question 평균 | 1.9 (min 1, max 6) — oracle은 evidence sessions만 |
| turns/session 평균 | 11.6 (min 2, max 32) |
| turn content 길이 | 평균 **1206자** (chat assistant 긴 응답) |
| user/assistant 비율 | 5479 / 5481 (1:1) |
| has_answer=True 턴 | 896 / 10960 (8.2%) — sparse evidence |
| 날짜 span/question | 평균 21일, max 256일 |

### question_type 분포 (500)
| type | 개수 |
|---|---|
| temporal-reasoning | 133 |
| multi-session | 133 |
| knowledge-update | 78 |
| single-session-user | 70 |
| single-session-assistant | 56 |
| single-session-preference | 30 |
| abstention (id suffix `_abs`) | 30 |

### 특징
- **LongMemEval_S / _M은 별도 파일**: oracle은 evidence 세션만 포함된 버전. 전체 haystack(40 sessions 이상)은 `longmemeval_s_cleaned.json` 등 다른 파일.
- **Chat assistant 스타일**: 1206자 평균 content → LoCoMo(138자), TopiOCQA(37자)와 큰 차이. **Claude-유사 사용 패턴과 가장 근접**.
- **Question_type = semantic tag**: topic annotation은 아니지만 질문의 인지적 능력 축을 표시. 5개 능력별 accuracy 개별 측정 요구.
- **Multi-session reasoning 133 questions**: 여러 세션을 걸친 정보 통합 필요 → **Hi-EM의 LTM/STM 계층 설계가 직접 검증되는 축**.
- **Knowledge update 78**: 동일 주제가 시간에 따라 사실이 바뀌는 케이스 → topic merge/update 로직 필요.
- **Abstention 30**: 근거 없는 질문에 "모름" 답해야 → retrieval이 잘못된 topic을 가져오면 실패.

### Hi-EM 시사점
- **Claude-유사 장기 대화의 benchmark 근사치로 가장 적합**. topic annotation이 없는 점도 실서비스와 일치.
- 긴 response → feature embedding에 노이즈 많음. centroid만으로는 구분 약할 가능성 → **옵션 A 단순형은 LongMemEval에서 한계 예상**.
- `knowledge-update` 유형 존재 → **같은 topic의 centroid도 시간 따라 drift**해야 한다 → Welford online update 필수.
- Multi-session reasoning 축이 세어서 Hi-EM의 LTM 설계가 결정적.

---

## 4. 세 벤치마크 비교

### 정량 비교표

| 지표 | LoCoMo | TopiOCQA | LongMemEval (oracle) |
|---|---|---|---|
| 단위 | 대화 | 대화 | 질문 |
| 단위 수 | 10 | 205 | 500 |
| 턴/단위 평균 | 588 | 12.3 | 22 (1.9 sess × 11.6 turns) |
| turn 길이 (chars) | 108~142 | 37 (factoid) | 1206 (chat) |
| Topic annotation | **없음** (session=날짜) | **명시** (Wiki doc) | **없음** (question_type=능력 태그) |
| Topic shift 검증 | 간접 (session 경계) | **직접 F1** | 간접 (session 경계) |
| 대화 style | Chit-chat / personal | Factoid QA | Chat assistant |
| 다루는 주 task | QA + event summary | Topic shift detection + QA | QA across sessions |
| Claude-유사도 | 중 | 낮음 | **높음** |

### 사건 모델(옵션 A~F) 선호도 증거

| 옵션 | LoCoMo 적합도 | TopiOCQA 적합도 | LongMemEval 적합도 | 종합 |
|---|---|---|---|---|
| A (Centroid only) | 중 (session 경계 회수 가능성) | **높음** (entity-rich Wiki) | 낮음 (긴 response로 centroid 불안정) | 단독으로는 부족 |
| B (Centroid + Momentum) | 중 | 중 | 중 | 대화 비순차로 효과 약함 |
| C (Centroid + Entity set) | 중 (인물/장소 유효) | **높음** (Wiki entity 풍부) | 중 (엔티티 많지만 non-entity 턴도 다수) | TopiOCQA bias 위험 |
| D (Multi-signal ensemble) | 높음 | 높음 | 높음 | 복잡성 대비 이득 분석 필요 |
| E (Small linear predictor) | 낮음 (cold start 약함) | 낮음 | 낮음 | 작은 topic에서 과적합 |
| F (새 제안) | — | — | — | — |

### 판단 기준
- **필수 충족**: LongMemEval에서 쓸 만해야 함 (Hi-EM의 주 타깃 시나리오와 가장 가깝기 때문).
- **추가 검증**: TopiOCQA로 topic shift F1 수치 validate.
- **편향 회피**: TopiOCQA만 보고 옵션 C로 고정하지 않는다 (handoff.md 경고).

---

## 5. Step 0-3 입력으로 넘길 핵심 관찰

1. **Topic annotation이 있는 건 TopiOCQA 하나**. 나머지는 session(날짜) 경계가 유일한 간접 ground truth. → Hi-EM은 unsupervised segmentation이 default, evaluation 시 TopiOCQA만 F1 직접 측정.
2. **턴 길이 분포가 벤치마크 간 30배 차이** (37 / 138 / 1206자). 임베딩 벡터의 variance 분포가 크게 달라져 **$\sigma_0^2$ prior**를 벤치마크별로 재설정할 필요 가능성.
3. **Knowledge update 78개 (LongMemEval)** → topic centroid가 시간에 따라 drift해야. Welford online update가 설계 필수.
4. **Multi-session / temporal reasoning 총 266개 (53%)** → LTM의 multi-session cross-reference가 평가 축.
5. **TopiOCQA의 section shift가 topic shift의 2배** → Hi-EM의 $\lambda$ 값이 너무 작으면 section 수준에서 과분할. tuning 대상 1순위.

### 다음 Step (0-3)에서 결정할 것
- 옵션 A~F 중 하나 선택 (또는 새 옵션 F 제안). LongMemEval·LoCoMo에서 centroid-only의 한계를 실험 전 추정 가능한 만큼 기술하고 의사결정.
- $\alpha, \lambda$ 초기값 재검토: 현재 `α=1.0, λ=10.0` (TopiOCQA section 2배 빈도 고려 시 $\lambda$ 더 키워야 할 수도).
- Welford online centroid/variance 업데이트 확정 (이미 확정이지만 knowledge-update 시나리오 실험 설계 필요).

---

## 6. 검증 미해결

- **LongMemEval_S / _M (115k / 500sess)는 이번 분석에서 생략**. oracle만 본 상태 — 실제 needle-in-haystack retrieval 성능은 _S로 검증해야 Hi-EM의 STM/LTM 설계가 충분한지 판정 가능. Phase 1 구현 후 재검토.
- **LoCoMo QA category 숫자(1~5)의 정확한 정의**는 레포 README/논문에서 아직 못 찾음. 분포만 확인. Step 0-3 전 재확인 가치 있음.
- **TopiOCQA의 `Context` 필드 형식**과 `is_nq` 영향 미분석 — factoid vs open 대화 구분 가능 여부 확인 미완.
- **턴 길이 평균 계산에서 `text`/`content` 필드만 사용** — metadata 길이는 제외. 실제 임베딩 입력 길이와 다를 수 있음.

이러한 미해결 지점은 사건 모델 옵션 선택(Step 0-3)에 결정적 영향이 없다고 판단하나, Phase 1 구현 전에 재검토한다.
