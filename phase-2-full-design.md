# Phase 2-Full: STM + Round + Importance 구현 계획

> **목표**: 현재 baseline (LTM + stateless memory_window 함수)을 **2-tier 메모리 시스템** (장기 메모리 LTM = SSD File / 단기 메모리 STM = Memory Window, 라운드 내 STM 고정) + **라운드 비동기 처리** + **Topic importance 4작용**으로 확장. Phase 4 sanity 결과의 multi-session 약점(0.20)이 baseline의 **stateless re-selection**과 **fixed budget**에서 기인함이 확인됨 → 본 설계는 정확히 그 두 axis를 풀어냄.
>
> Hi-EM의 winning thesis(적은 token으로 baseline 동등)를 회복하는 path. 단 본 설계 도입 후에도 RAG 못 넘으면 정직 reframing.

---

## 0. 현재 구현 vs 목표 (사용자 표 그대로)

### 0.1 메모리 계층

| 항목 | 현재 (src/hi_em/) | 목표 |
|---|---|---|
| LTM 위치 | `data/ltm/<conv>/{state.json, *.jsonl}` | 동일 |
| LTM 내용 | turn 원문 + topic_id + turn_id + embedding + topic centroid | 동일 |
| LTM 쓰기 시점 | 매 턴 sync | 동일 |
| **STM 존재** | ❌ (memory_window 함수만) | ✅ **MemoryWindow 클래스 (in-process, RAM 보유)** |
| **STM 내용** | — | importance ≥ threshold인 topic의 turn 전체 |
| **STM 상태 보존** | — | 라운드 사이 유지, 매 턴 누적/eviction |
| **Eviction** | — | 가득 차면 lowest importance topic 통째 제거 |

### 0.2 매 턴 처리

| 항목 | 현재 | 목표 |
|---|---|---|
| 분절 신호 | sCRP centroid distance ✓ | 동일 |
| Retrieval source | LTM stateless 재선별 | **STM 우선, miss 시 LTM→STM 승격** |
| Retrieval 단위 | topic top-3 × 마지막 5 turn | **STM 내 topic 전체** |
| 라운드 내 STM 변동 | (해당 없음 — 매 턴 stateless 재선별) | **라운드 시작 시 STM 고정**, 라운드 내는 STM miss 1건 + 직전 발화 append만 변동, promotion/eviction은 라운드 경계에서만 |
| Cold start | 빈 LTM이면 no prefill | LTM 직접 디코딩 |
| 응답 후 | LTM append | 동일 |

### 0.3 라운드 처리 (신규)

| 항목 | 현재 | 목표 |
|---|---|---|
| Round 개념 | ❌ | ✅ **10 turn = 1 round** |
| 실행 시점 | — | IDLE 시간, async |
| LTM 재구성 | append-only | **topic_id 기준 정렬** (단 turn_id는 보존) |
| STM 갱신 | — | importance 기반 promotion |

### 0.4 Topic Importance (신규)

| 작용 | 효과 | 공식 (잠정) |
|---|---|---|
| **강화 (turn 수)** | turn 많을수록 ↑ | $\log(1 + n_t)$ |
| **강화 (빈도)** | 자주 언급될수록 ↑ | round별 mention count의 EMA |
| **망각 (recency)** | 오래될수록 ↓ | $\exp(-\lambda_r \cdot (\text{round}_\text{now} - \text{round}_\text{last}))$ |
| **연결 (인접/통합)** | 인접 topic + 통합 신호 강한 topic의 importance 가중 합 | $\sum_j w_{ij} \cdot I_j$ |

종합:
$$I_t = \alpha_1 \log(1+n_t) + \alpha_2 \mathrm{EMA}(\text{freq}) + \alpha_3 \exp(-\lambda_r \Delta\text{round}) + \alpha_4 \sum_j w_{tj} I_j$$

가중치 $\alpha_{1..4}$ 초기값은 모두 1.0 (uniform), Phase 4 sweep 결과로 튜닝.

---

## 1. 구현 단계 (Step 별)

### Step P2F-1: `topic_importance.py` (단위 함수, 1일)
- `compute_importance(topics_state, round_now, mention_log) -> dict[topic_id → float]`
- 4 작용 각각 분리 함수 → 종합 → unit tests 8~10개
- **검증**: 4 작용 각각 isolation test + edge case (빈 history / 최근 1 turn / 통합 신호 없음)

### Step P2F-2: `MemoryWindow` class (단기 메모리, 2일)
- 모듈 `src/hi_em/memory_window.py` 확장 (현재 함수만 → class로)
- API:
  ```python
  class MemoryWindow:
      def __init__(self, max_topics: int, max_turns: int): ...
      def get(self, topic_id: int) -> list[dict] | None  # STM hit / miss
      def promote(self, topic_id: int, turns: list[dict]) -> None  # LTM → STM
      def evict_lowest_importance(self, importance: dict[int, float]) -> int | None
      def all_turns(self) -> list[dict]  # current STM 전체 turn (chronological)
      def current_topics(self) -> set[int]
  ```
- 기존 `select_memory_window` 함수 deprecate (Phase 2 baseline 평가용으로 유지하되 internal로)
- **검증**: STM hit/miss, eviction policy, capacity boundary, 라운드 사이 state 유지

### Step P2F-3: `RoundProcessor` (3일)
- 모듈 `src/hi_em/round_processor.py`
- API:
  ```python
  class RoundProcessor:
      def __init__(self, ltm: LTM, stm: MemoryWindow, threshold: float, alpha: list[float], lambda_r: float): ...
      def process(self, conv_id: str) -> RoundResult
          # 1. mention log 갱신 (이번 라운드의 turn 분포)
          # 2. compute_importance (모든 topic)
          # 3. STM promotion: importance ≥ threshold인 topic 모두 STM에 (없으면 promote)
          # 4. STM eviction: capacity 초과 시 lowest importance 제거
          # 5. (선택) LTM 재구성: state.json topic_id 기준 정렬 (단 jsonl은 append-only 유지)
  ```
- async 실행: `asyncio` 또는 `concurrent.futures`. 단일 process이므로 단순 thread 충분.
- **트리거**: `HiEM.handle_turn`에서 turn count % 10 == 0일 때 round_processor.process 호출 (background thread).
- **검증**: 10 turn 후 자동 발동 / mention log 정확성 / promotion threshold 동작 / eviction 후 STM 크기 보장

### Step P2F-4: `HiEM.handle_turn` 수정 (1일)
- 매 턴 흐름 변경:
  ```python
  def handle_turn(self, user_text: str) -> str:
      q = self._encoder.encode([user_text])[0]
      topic_id, is_boundary = self._segmenter.assign(q)

      # NEW: STM 우선
      stm_turns = self._stm.get(topic_id)
      if stm_turns is None:                              # STM miss
          ltm_turns = self._ltm.load_turns(self.conv_id, topic_id=topic_id)
          self._stm.promote(topic_id, ltm_turns)
          stm_turns = ltm_turns

      # NEW: STM 전체 chronological prefill (라운드 내 안정)
      prefill = self._stm.all_turns()
      messages = [...]  # 동일

      response = self._llm.chat(messages, model=self._model, **self._llm_kwargs)
      self._ltm.append_turn(...)  # user
      self._ltm.append_turn(...)  # assistant

      # NEW: 라운드 트리거
      if self._next_turn_id % (2 * 10) == 0:  # 10 user+assistant pair
          self._round_processor.process_async(self.conv_id)

      return response
  ```
- `return_debug` 그대로 유지 (Phase 4 metric용 — STM hit/miss flag 추가)
- **검증**: 10 unit tests (STM hit / STM miss → LTM promote / 라운드 트리거 / cold start / 라운드 내 STM 고정 invariant는 통합 smoke test)

### Step P2F-5: 라운드 내 STM 고정 invariant 검증 (1일, 검증 위주)
- **설계 invariant**: 라운드 시작 시 STM snapshot이 고정됨. 라운드 내 매 턴은 그 STM에 직전 user/assistant 발화만 append (`maybe_append_turn`) 후 응답 생성. STM의 promotion / eviction은 `RoundProcessor`가 라운드 경계에서만 수행.
- **예외 (STM miss)**: 현재 turn의 topic이 STM에 없을 때만 LTM에서 그 topic을 promote. 이 경우 외 라운드 내 STM 구성은 변하지 않음.
- 이는 LLM 런타임 내부 mechanism에 의존하는 feature가 아니라 **Hi-EM 측 설계 불변량**. P2F-2/3/4가 이 불변량을 implement, P2F-5는 그것을 명시적으로 test.
- **검증**:
  - (a) unit: 라운드 내 매 턴마다 `stm.current_topics()` snapshot 비교 — STM miss로 인한 promotion 외 변화 없음
  - (b) unit: `RoundProcessor.process`는 라운드 경계 (`turn_id % (2*round_size) == 0`)에서만 호출
  - (c) integration smoke (P2F-6): 25-turn trace에서 STM 변경 시점이 (라운드 경계 ∪ STM miss)뿐임을 확인

### Step P2F-6: 통합 smoke test (1일)
- `scripts/smoke_test_full_pipeline.py`: A→B→A→B→A 5턴 시나리오, STM hit/miss + 라운드 발동 trace 출력
- 응답 시간 비교: baseline (현재 stateless) vs full pipeline (STM 고정 + 라운드 처리)

### Step P2F-7: Phase 4 재실행 (반나절)
- `run_longmemeval.py`의 hi-em method를 새 `HiEM`으로 자동 교체 (HiEM API 안 바뀌면 코드 변경 없음)
- sanity 30 nothink + sanity 30 think 비교
- 결과로 D/E 결정 (Phase 4 reframing)

---

## 2. 의존 그래프

```
P2F-1 (importance) ──┐
                     ├─→ P2F-3 (RoundProcessor) ─┐
P2F-2 (MemoryWindow) ┘                           │
                                                  ├─→ P2F-4 (handle_turn 수정) ──→ P2F-5 (STM 고정 invariant) ──→ P2F-6/7
                                                  │
                                  Phase 4 sanity ──┘ (검증 trigger)
```

총 추정 시간: **8~10일** (1~2주).

---

## 3. 호환성 / 회귀 위험

### 3.1 보존해야 할 기능
- `HiEM.handle_turn(user_text)` 시그니처: caller (Phase 4 평가) 변경 없이 동작
- `preload_history(turns)`: 기존 동작 (segmenter 통과 + LTM 저장) 유지하되 STM에는 promote 안 함 (또는 별도 옵션 — caller 결정)
- 58 unit tests: 기존 4 tests (test_orchestrator의 single-turn / first-turn / second-turn / topic-revisit)는 **반드시 PASS**

### 3.2 Risk
- **STM 상태가 conv_id 사이 격리 안 되면 leak**: per-conversation STM 인스턴스 강제
- **async 라운드 처리 race**: Python `threading.Lock`으로 STM/LTM 보호 (encoder처럼)
- **importance 가중치 잘못 설정 시 thrashing**: P2F-1 unit test로 monotonicity 검증 + Phase 4 sweep으로 튜닝
- **라운드 내 STM이 비의도적으로 mutation (STM miss / 라운드 경계 외)**: P2F-5 invariant test로 검출 (`maybe_append_turn` / eviction 트리거 시점 점검)

### 3.3 Phase 4 재실행 시 비교 baseline
| 라벨 | 의미 |
|---|---|
| `hi-em` (현재) | stateless re-selection, k_topics=3 × k_turns_per_topic=5 |
| `hi-em-full` (P2F 후) | STM 라운드 고정 + round 처리 + importance |
| 가설 | hi-em-full이 multi-session에서 0.20 → 0.40+ |

---

## 4. 비검증 사항 (P2F 종료 후에도 미해결)

1. **Phase 5 논문 실험·재현**: 본 설계는 평가 인프라 강화. 논문 수치 재현은 Phase 5.
2. **m_cleaned (500 sessions)** 평가: oracle은 짧음. STM의 진짜 가치(긴 history + 토픽 복귀)는 m_cleaned에서만 정량 가능.
3. **importance 4 작용 가중치 최적값**: P2F-7 sweep 후 확정. 본 plan은 baseline 가중치만.

---

## 5. 다음 단계 (이 plan 채택 시)

1. 본 md를 codex review에 제출 → design 결함 / 누락 / overengineering 지적 받기
2. review 반영 후 P2F-1부터 순차 implement
3. 각 step 별 commit (Phase 2 commit 패턴 유지)
4. P2F-7 (sanity 비교) 결과로 Phase 4 reframing 최종 결정

## 5'. 구현 진행 (2026-04-27)

| Step | 상태 | 결과 |
|---|---|---|
| P2F-1 (`topic_importance.py`) | ✅ 2026-04-26 | 13 tests pass |
| P2F-2 (`MemoryWindow` class) | ✅ 2026-04-27 | 20 tests pass — topic-atomic invariant 강제 (turn-level slicing API 부재) |
| P2F-3 (`round_processor.py`) | ✅ 2026-04-27 | 13 tests pass — async + per-instance RLock |
| P2F-4 (`HiEM.handle_turn` STM-first) | ✅ 2026-04-27 | 10 tests pass + 회귀 13/13 — `use_stm=True` 옵션, in-sync turn append |
| P2F-5 (라운드 내 STM 고정 invariant) | ✅ 부분 — 명시적 invariant unit test 미작성 | smoke test 25-turn trace에서 STM 변동이 라운드 경계 + STM miss뿐임 육안 확인 |
| P2F-6 (smoke test) | ✅ 2026-04-27 | `scripts/smoke_test_full_pipeline.py` 25-turn invariant pass |
| P2F-7 (Phase 4 sanity 비교) | 🔲 사용자 실행 대기 | `--method hi-em-full --limit 30 --stratify` 1 question OK / 30Q OK (0/30 error, 78s) |

**핵심 추가**: `MemoryWindow.maybe_append_turn(topic_id, turn)` — STM 내 topic만 in-sync 갱신. STM에 없는 topic은 partial 안 채움 (atomicity 보존). 이로써 라운드 안에서도 STM 내 topic의 prefill이 stale하지 않으면서 STM 자체의 promotion / eviction은 라운드 경계까지 발생하지 않음 → 라운드 내 prefill 구성이 안정적으로 유지됨.

**확인된 invariants** (smoke + sanity):
- topic atomicity — STM 내 모든 topic의 turn 수가 LTM full count와 일치
- round trigger — `next_turn_id % (2*round_size) == 0`에 정확히 1회 발동
- max_topics 캡 준수, max_turns soft 캡 (단일 topic이 cap 초과 시는 atomicity 우선)
- A→B→A 패턴에서 STM revisit hit (smoke 4/5)

---

## 6. 결정 분기 (사용자 확인 필요)

| 항목 | 권고 | 대안 |
|---|---|---|
| **Round 크기 (turn)** | **10** (사용자 제안) | 5 / 20 / adaptive |
| **STM capacity** | max_topics=10, max_turns=200 | 측정 후 조정 |
| **Importance threshold** | 0.5 (uniform 가중치 가정) | 측정 후 |
| **Round async lib** | `threading.Thread` (단순) | `asyncio` (orchestration 늘어나면) |
| **LTM 재구성 범위** | state.json만 (jsonl append-only 유지) | jsonl도 재정렬 (디스크 I/O↑, 지양) |
| **라운드 내 STM 고정 invariant 검증** | unit test (snapshot 비교) + smoke test trace | (선택) 응답 latency 분산 측정으로 안정성 정량 |

---

## 7-pre. Codex Review (2026-04-26) 핵심 결함 + 사전 fix

### 🔴 가장 큰 risk
**core hypothesis가 wrong일 수 있음** — persistence HP가 46-turn history를 한 topic으로 collapse하는 게 root cause인데 STM은 그 bad partition을 cheaper하게 만들 뿐 better하게 안 만듦. 단 사용자 결정: "구현 후 실측으로 검증". HP regime 재시도는 P2F 종료 후 분기.

### 구현 시작 전 30분 fix (이 design에 반영)
| Codex 지적 | Fix |
|---|---|
| Q5: Round 트리거 모순 (10 vs 2*10) | **10 user+assistant pair = 20 raw turn** 트리거로 통일. P2F-4 코드 sketch의 `% (2*10) == 0` 그대로 의미: "10 라운드 turn = 20 jsonl row" |
| Q2: EMA half-life 미지정 | **`λ_freq = 0.5`** (round당 frequency EMA decay) 기본값. Phase 4 sweep으로 튜닝 |
| Q3: Concurrency lock granularity | **per-conversation `threading.RLock`** — STM/LTM 둘 다 같은 lock으로 보호. eviction은 lock 잡고 promotion 끝날 때까지 대기. 단순함 우선, lock 경합은 측정 후 재검토 |

### 실측으로만 풀리는 것 (P2F-7 후 판단)
- Q1/Q7: STM이 multi-session 0.20을 0.40+로 풀어내는가, 아니면 cheaper-only인가
- Q6: MVP만으로 충분한지 — 본 design 풀 패키지 구현 후 ablation으로 측정 가능

### 보류 / 후속
- Importance 재귀 (Iⱼ)의 fixed-point: 1차 구현은 **Σⱼ wᵢⱼ Iⱼ를 직전 round의 importance**로 평가 (재귀 1단계만, 수렴 보장 위해 iteration 제한 N=1). Phase 4 결과로 수렴 알고리즘 필요한지 결정
- Recency floor: quiet-but-important topic starve 회피 위해 `min_importance_floor = 0.1`로 cap. P2F-1 unit test에 포함

---

## 7. 검증 미해결 (구현 후 P2F-7에서 판정)

1. **Topic importance 공식의 monotonicity가 4 작용 모두에서 보장되나?** — round count 증가 시 망각이 다른 작용을 압도하는 케이스 등.
2. **STM eviction 후 다시 같은 topic 등장 시 promotion 비용** — 매번 LTM에서 다 읽어 RAM에 올리면 효과 무의미. incremental 적재 필요한가?
3. **Round async가 매 턴 흐름과 deadlock 안 함을 보장**할 lock 그래프?
4. **본 design이 Phase 4 multi-session 0.20을 실제로 풀 수 있나?** — 즉 budget 늘림 + STM 영속화가 정답 누락 문제를 직접 푸는가, 아니면 같은 문제 단순 transparent하게 만드는가?
5. **라운드 내 STM 고정이 응답 정확도에 영향 없는가?** — 라운드 중간에 새 topic으로 shift하면 STM miss로 즉시 promote (P2F-4). 그 외에는 STM이 라운드 경계까지 라운드 시작 시점 구성을 유지 — 그 사이에 다른 topic의 새 turn 정보가 답변 결함을 일으키는가.
