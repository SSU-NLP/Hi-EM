# Hi-EM 실험 가이드

`scripts/experiment.py` 사용법 + 결과 해석 요약. 더 깊은 알고리즘 정의는 `context/methodology/README.md`.

---

## 1. `scripts/experiment.py` 가 만드는 출력물

새 sweep / ablation / comparison 은 **`scripts/experiment.py`** 한 entry 로 돌린다 (Claude Code 에 "이런 method 들로 돌려줘" 라고 부탁). 이 스크립트가 method × HP 조합마다 자식 프로세스 (`run_experiment.py`) 를 띄우고, 끝나면 결과를 모아 `REPORT.md` 한 장으로 만든다.

전체 산출물은 **`outputs/experiments/<name>/`** 안에 self-contained 로 모인다:

```
outputs/experiments/<name>/
├── REPORT.md                          ← (1) 전체 비교 표 ★ committed
└── <label>/                           ← method × HP 1 조합 마다 한 폴더
    ├── exit_code.txt                  ← (2) 자식 종료 코드 ("0"=성공, resume key)
    ├── run.log                        ← (3) 자식 stdout (디버그용)
    ├── stm_topk.json                  ← (4) STM 통계 (hi-em-full-* 만)
    └── results/experiments/<exp_id>/
        ├── summary.json               ← (5) 이 method 의 전체 metric (REPORT 의 한 행 원본)
        └── rounds/round_NNN/          ← 질문 200 개씩 한 round
            ├── hypothesis.jsonl       ← (6) LLM 답변 raw
            └── judged.jsonl           ← (7) judge 결과 + per-Q metric
```

**각 파일의 의미**:

1. **`REPORT.md`** — 모든 method 끝나면 자동 생성. 각 row 의 metric 은 (5) `summary.json` + (4) `stm_topk.json` 에서 뽑아온 값. **이 파일만 커밋**, 나머지 raw 는 모두 `.gitignore`. 이미 끝난 sweep 의 표만 다시 그리고 싶으면 Claude Code 에 "aggregate-only 로 다시 만들어줘" 하면 됨.
2. **`exit_code.txt`** — `0` 이면 그 method 완료. 같은 `<name>` 으로 재실행하면 `0` 인 method 는 skip → 중간에 죽어도 안전하게 이어 돌림.
3. **`run.log`** — 자식 프로세스 stdout. 실험이 죽거나 이상하면 여기 본다.
4. **`stm_topk.json`** — hi-em-full-\* method 만. round 마다 STM 안의 가장 큰 topic 1·2·3 의 turn 수를 push. REPORT 의 STM 컬럼 (`T1μ` 등) 원본.
5. **`summary.json`** — 그 method 의 전체 metric (정확도 + qtype 별 + 평균 latency + retrieval). REPORT 의 한 행 원본.
6. **`hypothesis.jsonl`** — 질문 1 개당 한 줄, LLM 답변 + 메타데이터 (어떤 turn 들을 retrieved 했는지 등). 답변 자체를 보고 싶을 때.
7. **`judged.jsonl`** — `hypothesis.jsonl` 에 judge LLM 의 yes/no 평가 + per-question retrieval metric (H@k / R@k / P@k) 를 붙인 것. 잘 못 맞춘 질문 case 분석할 때.

**참고 — 다른 outputs/ 디렉토리**:

```
outputs/
├── experiments/   ← 위 트리 (sweep 들의 본진)
├── runs/          ← run_experiment.py 단독 호출 (디버그/smoke). gitignored.
├── reports/       ← 사람이 쓴 독립 분석 MD. committed.
└── design/        ← 설계 문서. committed.
```

`archive/` 는 의도적으로 폐기한 실험 (`README.md` 에 폐기 사유 기록). 실험 데이터는 모두 gitignored.

---

## 2. REPORT.md 표 — 컬럼 설명

`REPORT.md` 의 행 = method × HP 1 조합. 컬럼은 4 그룹:

### (a) 정확도 (LLM judge yes/no, 0~1)

| 컬럼 | 의미 |
|---|---|
| `accuracy_overall` | 전체 평균 정확도 |
| `multi-hop` | 여러 turn 의 정보를 합쳐야 답하는 질문 (cat 1) |
| `single-hop` | 한 turn 만 보면 되는 질문 (cat 4) |
| `temporal-reasoning` | 시간 추론 질문 (cat 2) |
| `adversarial` | "Not mentioned in conversation" 류 (cat 5, multi-choice) |
| `open-domain` | 자유 답변 질문 (cat 3) |

### (b) latency / 비용

| 컬럼 | 의미 |
|---|---|
| `gen_p50(s)` | per-question LLM generation 시간 (중앙값) |
| `wall` | method 전체 wall-clock time |

### (c) STM 통계 (hi-em-full-* 만)

`stm_topk.json` 에서 매 round 마다 STM 안에서 가장 큰 topic 1·2·3 의 turn 수를 기록.

| 컬럼 | 의미 |
|---|---|
| `T1μ` / `T2μ` / `T3μ` | round 평균 — STM 1·2·3위 topic 의 turn 수 |
| `T1max` / `T2max` | 전 round 중 1·2위 topic 최대 turn 수 (mega-topic 진단) |
| `T1var` | round 간 1위 topic 크기 변동 |
| `STM_n_topics` | round 평균 STM 안 topic 개수 |

→ 진단 활용: `T1μ` 가 100+ 면 mega-topic 의심, `STM_n_topics` 가 1 이면 single-topic 으로 collapse.

### (d) Retrieval (정답 turn 회수율, LoCoMo evidence 기준)

LoCoMo 는 질문마다 정답이 들어있는 turn 의 `dia_id` 리스트 (`evidence`) 를 제공. method 가 LLM 에 prefill 한 turn 의 `dia_id` 와 비교.

`k` = **method 가 LLM 에 prefill 한 turn 수** (method 마다 다름):

| method | k |
|---|---|
| `hi-em-full-*` | STM 안 모든 turn (≈ 100–200) |
| `rag` / `rag-summary` / `rag-observation` | `rag_k` = 10 |
| `sliding` | `sliding_k` = 20 |
| `full` | 전체 history |

| 컬럼 | 의미 |
|---|---|
| `H@k` | Hit@k — prefill 안에 정답 turn 이 1개 이상 있는 비율 |
| `R@k` | Recall@k — 정답 turn 들 중 몇 % 가 prefill 에 있는가 |
| `R-multi-hop@k` | multi-hop 질문만 따로 본 R@k (모든 evidence 가 들어가야 답 가능) |
| `P@k` | Precision@k — prefill 중 정답 비율 (낮으면 noise 多) |

→ 진단 활용: hi-em 의 `P@k` 가 RAG 의 1/22 (0.003 vs 0.07) — k 가 hi-em 은 ~150, RAG 는 10 이라 noise 차이 큼. topic 단위 prefill 의 본질적 한계를 보여주는 지표. acc bottleneck 진단의 핵심.

---

## 3. methodology 디렉토리 안내

알고리즘 정의는 모두 **`context/methodology/`** 에 모여 있다.

```
context/methodology/
├── README.md           ← ★ 먼저 보는 곳. 버전 계보 + 한 줄 요약 + HP 매트릭스
├── infrastructure.md   ← 버전 무관 인프라 (cache, encoder lock, REPORT 컬럼 정의 등)
├── v1.md               ← Gaussian likelihood baseline
├── v2.md               ← v1 + STM round-clear / sessioned preload
├── v3.1.1.md           ← Gaussian → bounded cosine
├── v3.2.1.md           ← sticky-CRP count 에 sub-linear (C+1)^β
├── v3.3.1.md           ← centroid → per-topic GRU 예측
├── v3.3.2.md           ← + SEM2 surprise hard PE boundary
├── v3.3.3.md           ← + SEM2 f0 / restart 분기 복원
└── v3.3.4.md           ← hard PE → per-topic σ²_k calibrated likelihood
```

**언제 어디를 보는가**:

- 새 sweep 돌릴 때 어떤 method 가 무슨 HP 를 쓰는지 알고 싶다 → **`README.md` 의 HP 매트릭스 한 표**.
- 특정 버전의 식 / SEM 계승 / 알려진 한계 / 최근 sweep 결과를 보고 싶다 → 해당 **`vX.Y.Z.md`**.
- 캐시 / encoder lock / 새 버전 추가 절차 등 인프라 질문 → **`infrastructure.md`**.
