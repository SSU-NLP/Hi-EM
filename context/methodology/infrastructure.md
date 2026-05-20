
# 공용 Infrastructure (cross-cutting design)

버전 간에 공유되는 *non-version-specific* 인프라/구현 설계. 사소한 cache 정책, locking, encoding 단위 같은 항목도 여기 누적.

각 항목은 다음 형식:
- 무엇 (정의)
- 어디 (코드 위치)
- 왜 (도입 동기)
- 행동 영향 (어느 method 가 받는가)
- 알려진 한계 / 변형 후보

---

## 1. `EncoderCache` — RAG 계열용 임베딩 캐시

- **무엇**: `(sample_id, kind)` 키로 conversation 별 corpus 임베딩을 캐싱. `kind ∈ {"history", "summary", "observation"}`.
- **어디**: `scripts/run_experiment.py` 의 `class EncoderCache`.
- **왜**:
  - LoCoMo 한 conversation 에 ~200 질문이 같은 600-turn history 를 공유. 캐시 없으면 질문마다 600 turn 을 재인코딩 → 200×.
  - encoder 는 thread lock 으로 직렬화돼 있어 (`QueryEncoder.encode` 의 global Lock), worker 수를 늘려도 인코딩 비용은 그대로.
- **행동 영향**:
  - `rag` / `rag-summary` / `rag-observation` baseline 의 retrieval 단계.
  - 의미 자체는 안 바꿈 — 동일한 임베딩을 한 번만 계산.
- **Locking 패턴**: per-key build lock 을 outer `_dict_lock` 안에 둠. concurrent worker 가 같은 sample 에 대해 중복 인코딩 안 하도록.
- **알려진 한계 / 변형 후보**:
  - 현재 sample 단위로 인메모리. conversation 간 공유 캐시 없음. 다른 sample 이 같은 텍스트 chunk 를 갖고 있어도 재인코딩.

---

## 2. `HiEMConvCache` — Hi-EM 인스턴스 캐시 (per-conversation post-build state)

- **무엇**: `(sample_id, method)` 키로 *대화 segmentation 이 끝난 시점의 HiEM 인스턴스* 를 캐시. 질문은 read-only `eval_query()` 로만 인스턴스를 사용.
- **어디**: `scripts/run_experiment.py` 의 `class HiEMConvCache`.
- **왜**:
  - LoCoMo / LongMemEval 스타일에서 같은 600-turn history 에 200 질문을 던짐. 질문마다 LTM jsonl + segmenter centroid + STM round-promote 를 재구축하면 200× preload 비용.
  - 한 번 build → 모든 질문이 공유.
- **행동 영향**:
  - `hi-em` 와 모든 `hi-em-full-vX.Y.Z` — `is_hi_em_method()` regex helper (`^hi-em(-full-v[\d.]+)?$`) 가 prefix 매칭하므로 새 버전 자동 활성화. 별도 set 갱신 불필요 (2026-05-09 리팩터).
  - segmentation 결과 자체는 변경 없음. 단지 같은 결과를 재계산하지 않을 뿐.
  - **2026-05-09 버그 사례**: 도입 당시 explicit set 에 v3.3.3 / v3.3.4 누락 → 매 질문마다 600-turn 대화 재 preload (200x 낭비). regex helper 로 전환해 재발 차단.
- **Lazy build + per-sample lock**: 첫 worker 가 sample 잠금을 잡고 build, 동일 sample 의 다른 worker 는 대기 후 캐시 공유. 다른 sample 들은 병렬로 build.
- **반드시 read-only 사용**: 질문 처리는 `HiEM.eval_query` 로만 — 인스턴스 상태(centroid, count, prev_k) 를 mutate 하지 않음. mutate 하면 질문 순서가 결과를 바꿔버림.
- **알려진 한계 / 변형 후보**:
  - 디스크 캐시 없음 — 프로세스 재시작 시 재build.
  - `read-only` 보장은 코드 규약. 명시적 lock 으로 enforce 안 됨.

---

## 3. Encoder lock (단일 thread 직렬화)

- **무엇**: `QueryEncoder.encode` 가 global `threading.Lock` 으로 인코딩 호출을 직렬화.
- **어디**: `src/hi_em/embedding.py` (간접 — `EncoderCache` docstring 에 기록).
- **왜**: 임베딩 모델 자체가 multi-thread 안전하지 않음 (HuggingFace tokenizer + torch model). 한 thread 만 동시에 forward.
- **행동 영향**:
  - `--workers N` 늘려도 *인코딩 단계* 는 직렬. LLM 호출만 병렬.
  - Hi-EM 의 segmentation 단계는 conversation 단위 직렬 → encoder 가 직렬이라도 큰 문제는 안 됨.
  - RAG 계열은 `EncoderCache` 가 빌드된 뒤엔 인코딩이 거의 없으므로 영향 적음.
- **알려진 한계**:
  - 한 sample 의 인코딩 시간이 wall-clock 직선적으로 들어감.
  - GPU encoding 으로 가도 lock 은 그대로 (일관성 보장).

---

## 4. LLM `--no-thinking` (reasoning bypass)

- **무엇**: OpenAI-compatible `extra_body` 로 두 가지 키를 동시에 보냄:
  ```
  extra_body = {
      "chat_template_kwargs": {"enable_thinking": False},  # vLLM
      "reasoning": {"enabled": False},                     # Crts / OpenRouter
  }
  ```
- **어디**: `scripts/run_experiment.py` 의 judge 호출 + main chat 호출 두 곳에서 같은 `extra_body` dict 를 LLM kwargs 에 주입.
- **왜**:
  - qwen / DeepSeek / Crts proxied 모델들은 reasoning mode 가 default ON 이면 답을 `message.reasoning` 또는 `<think>...</think>` 로 보냄.
  - Hi-EM 의 LLM 어댑터는 `message.content` 만 추출 → reasoning 모델이면 `content=None` 이 와서 모든 hypothesis 가 빈 문자열.
  - 2026-05-07 sweep 사고: `--no-thinking` 빠진 채 v3.3.1 비교 sweep 돌려서 모든 run 의 `error_rate=1.00`. 결과 무의미.
- **행동 영향**:
  - 모든 method (Hi-EM 라인 + RAG 계열 + sliding) 의 LLM 호출.
  - hyperparameter 가 아니라 *환경 호환성 플래그*.
- **권장**: 외부 OpenAI-compatible endpoint (Crts / OpenRouter) 사용 시 *항상 켤 것*. 새 sweep script 작성 시 누락하지 않도록 기본 포함.

---

## 5. Retrieval policy — Hi-EM (선택된 topic 안에서)

- **무엇**: `HiEM.eval_query` 가 질문 임베딩으로 top-K topic 을 고른 뒤, 각 topic 에서 *최근 N turn* 을 가져옴.
- **어디**: `src/hi_em/memory_window.py` (`select_memory_window`), `src/hi_em/orchestrator.py` (`HiEM.eval_query`).
- **왜 / 한계**: 현재 retrieval 정책이 evidence-aware 가 아님. selected topic 이 정답 turn 을 포함하더라도 그 안의 어느 turn 이 답변에 결정적인지 모름 — *최근 N turn* 또는 *centroid 와 가까운 turn* heuristic 으로만 추리는 단계. v3.x 모든 버전이 이 정책을 그대로 상속. (decision-log 2026-05-09 entry 의 P@k 0.003 결과가 직접 증거 — H@k 는 0.42 (정답 retrieved) 이지만 P@k 는 0.003 (그 외 noise turn 다수).)
- **행동 영향**: 모든 Hi-EM 변형의 QA 결과 — 답이 "오래된 turn" 또는 "centroid 와 떨어진 turn" 에 있으면 못 찾음.
- **변형 후보** (아직 미적용):
  - 선택된 topic 안에서 turn-단위 cosine rerank
  - MMR (다양성)
  - 시간 필터
  - 질문 유형별 retrieval 전략 분리

### 5b. v3.3.3-4 retrieval atomicity (`(topic_id, episode_id)` 단위)

- **무엇**: v3.3.3-4 만 사용. STM 의 모든 turn 을 `(topic_id, episode_id)` 그룹으로 묶고 episode 단위로 score → top-K episode 안 turn 을 통째로 prefill.
- **어디**: `src/hi_em/orchestrator.py` `HiEM._episode_rerank_prefill()`. `retrieval_mode == "episode_rerank"` 일 때 `eval_query` 가 호출.
- **score**: `R(ep) = a·max_i cos(q,s_i) + b·S_topic + e·S_episode − g·z̄PE − r·recency` (defaults `a=1.0, b=0.10, e=0.35, g=0.05, r=0.03`, `episode_top_k=3`).
- **dormant LTM safety net**: `dormant_ltm_top_n=8` (default for v3.3.3-4) — STM 못 들어간 LTM topic 중 query cosine 상위 N turn 을 추가로 prefill. promotion miss backstop.
- **promotion_threshold default override**: v3.3.3-4 는 0.5 → 0.3 (evidence-bearing topic 의 STM 진입을 늘림).
- **다른 method 에 대해**: `retrieval_mode == "stm_all_turns"` (기본) 로 폴백 — 기존 동작 변경 없음. 이 retrieval 변경은 v3.3.3-4 전용.
- **한계**: token budget 으로 turn cut 시 atomicity 깨짐 (현재 fallback 동작). dormant LTM 도 LTM 분포 자체가 promotion 이력 종속이라 backstop 효과는 audit 으로 확인 필요.

---

## 6. STM topic atomicity

- **무엇**: STM 에 topic 이 들어갈 때 *통째로* 들어감. soft turn cap (`max_turns`) 보다 atomicity 가 우선.
- **어디**: `src/hi_em/memory_window.py` 의 `MemoryWindow` class docstring (불변식 정의) + `promote()` 메소드 (전체 turn list 강제 유입).
- **왜**: 한 topic 의 의미를 깨지 않으려는 설계.
- **한계**: 큰 topic (200 turn 짜리) 이 통째로 들어가면 prefill 폭주 (LoCoMo 19k token 사례, decision-log).
- **STM cap HP (HiEM common, sweep 시 함께 고려)**:
  - `stm_max_topics` (default 10) — STM 안 동시 유지 topic 수 cap. evict 정책에 영향.
  - `stm_max_turns` (default 200) — STM 전체 turn 수 cap. **prefill 길이를 직접 결정** → LLM latency / token cost 직결.
  - `promotion_threshold` (default 0.5) — round-end 시 STM 진입할 topic 의 importance 임계.
  - `importance_alpha`, `lambda_r`, `lambda_freq`, `min_floor` — importance 함수 파라미터 (round_processor).
  - 이 5 종은 segmenter version 과 직교하지만 retrieval 분포·acc 에 큰 영향. v3.3.x sweep 시 segmenter HP 와 함께 검토.
- **변형 후보**: summary 수준에서만 atomicity 유지, 내부 turn 은 bounded chunk 만.

---

## 7. Experiment harness — `scripts/experiment.py` (canonical)

- **무엇**: 단일 entry point 로 임의의 experiment (sweep / ablation / comparison) 를 실행 + REPORT.md 자동 생성.
- **어디**: `scripts/experiment.py`.
- **용어**: 한 *experiment* = 이름 붙은 여러 *run* 의 집합. 각 run = method × HP 한 조합. sweep, ablation, comparison 모두 같은 형태.
- **왜**: 이전엔 experiment 마다 별도 shell + 별도 aggregator 를 작성. 코드 중복 + 결과 형식 불일치. 통합 harness 로 대체.
- **사용**:
  ```
  uv run python scripts/experiment.py \
    --name <experiment-name> \
    --benchmark locomo \
    --data benchmarks/locomo/data/locomo10.json \
    --workers 100 --no-thinking \
    --method "method[:label][@k=v,k=v,...]" \
    [--method ...] \
    [--limit N --stratify]
  ```
- **Method spec 문법**: `method[:label][@k1=v1,k2=v2,...]`
  - `method`: `run_experiment.py --method` 값 (e.g. `hi-em-full-v3.3.2`, `rag`, `sliding`, `full`)
  - `label`: 출력 subdir 이름. 미지정 시 method (`.` → `p`, `-` → `_`).
  - `k=v`: `run_experiment.py` 의 `--k v` 로 forward. HP override.
- **Resume**: 같은 `--name` 으로 재실행하면 `<run_dir>/exit_code.txt = 0` 인 run 은 자동 skip. 중간에 죽어도 안전.
- **출력**: `outputs/experiments/<name>/<label>/` 에 per-run run.log + results/ + stm_topk.json (hi-em-full-* 만). 모든 run 끝나면 `outputs/experiments/<name>/REPORT.md` 자동 생성.
- **새 hi-em-full-vX.Y.Z 버전 추가 체크리스트** (2026-05-09):
  1. `src/hi_em/sem_core_vXYZ_*.py` (segmenter) + `topic_vXYZ_*.py` (필요시) 신규.
  2. `src/hi_em/orchestrator.py` 의 `version == "vX.Y.Z"` elif 분기 추가 + 새 HP 시그니처에 추가.
  3. `scripts/run_experiment.py` 세 곳: (a) argparse `--method choices`, (b) `HiEMConvCache._build` 의 `elif method == "..."` HP 매핑, (c) 질문 dispatch 의 `seg_version = "vX.Y.Z"`. 새 HP 는 argparse + `v3_extra` dict + `run_hi_em_full` kwargs 에 propagate.
  4. **자동 적용**: hiem_cache enable / encoder needed / STM topk recording 은 prefix `hi-em(-full-v\d+\.\d+\.\d+)?` 로 매칭해 자동 활성화 (`is_hi_em_method` 헬퍼). 별도 set 갱신 불필요.
  5. `tests/test_vXYZ_*.py` 추가 + `context/methodology/vX.Y.Z.md` (한 줄 정의 / 수식 / SEM 계승 / HP / 한계 / 변형) + `decision-log` entry.

- **REPORT.md 컬럼** (2026-05-09~): `method | notes | accuracy_overall | multi-hop | single-hop | temporal-reasoning | adversarial | open-domain | H@k | R@k | R-multi-hop@k | P@k | T1μ/T2μ/T3μ | T1max/T2max/T1var | STM_n_topics | gen_p50(s) | wall`. 모든 run 이 동일 데이터·limit 으로 돌므로 `n_questions` 는 표 위 header 한 줄로 분리. `notes` 는 method 식별의 일부이므로 method 바로 옆에 둠 — 해당 method 가 실제로 사용하는 HP 한 줄 요약 (hi-em-full-* 는 segmenter+memory_window, rag* 는 rag_k, sliding 은 sliding_k) + override 가 있으면 끝에 `· override: k=v` 로 append. retrieval 지표 4종 (H@k/R@k/R-multi-hop@k/P@k) 은 LoCoMo `evidence` (정답 turn `dia_id` 리스트) 와 method 가 prefill 한 turn `dia_id` 의 교집합으로 계산. `experiment.json` 의 `config` 에서 HP 를, `summary.json` 에서 metric 을 추출.
- **Aggregate-only 모드**: `--aggregate-only` 로 실험 안 돌리고 REPORT.md 만 재생성.
- **legacy**: `scripts/legacy/` 에 옛 per-experiment shell 스크립트 + per-aggregator 보존됨 (참고용).

---

## 7b. Dormant evidence audit (importance-policy 진단)

- **무엇**: 각 hi-em-full 질문에 대해 STM 에 못 들어간 LTM topic 들 중 정답 evidence 가 모인 비율을 측정.
- **어디**: `scripts/run_experiment.py` `compute_dormant_evidence_audit(entry, hi)`. round summary 에 합쳐져 `summary.json` → `eval_logging.aggregate_summary` 가 `dormant_ev_rate`, `n_topics_with_ev`, `top_ev_topic_promoted` 로 노출.
- **언제 emit**: `is_hi_em_method()` true + LoCoMo entry 에 `evidence` (dia_id 리스트) 존재할 때만.
- **메트릭**:
  - `dormant_ev_rate` = (LTM-only topic 안 evidence turn 수) / (전체 evidence turn 수). 1 에 가까울수록 정답이 dormant 에 몰려있다는 뜻 → importance score 가 evidence-bearing topic 을 STM 못 올림.
  - `top_ev_topic_promoted` = evidence 가 가장 많이 몰린 topic (전체 기준) 이 STM 안에 있는지 (0/1).
  - `n_topics_with_ev` = evidence 가 1+ 개 들어있는 topic 수 (다항 evidence 분포 가늠).
  - **(2026-05-11 추가, dormant 안 집중도 직접 측정)**
  - `n_dormant_topics_with_ev` = dormant 안 evidence-bearing topic 수.
  - `dormant_top_topic_n_ev` = dormant topic 중 evidence 가 가장 많이 몰린 topic 의 evidence 개수.
  - `dormant_top_topic_share` = `dormant_top_topic_n_ev / dormant_ev_count`. **1.0 → 한 dormant topic 에 dormant evidence 가 다 몰림** (사용자 가설 강력 지지, promotion fix 만으로 회수 가능). **낮음 → dormant 안에서도 흩어짐** (segmentation 단계 cause 가능성 부각).
- **왜**: importance score / promotion threshold 정책 자체에 문제가 있는지 직접 진단. 사용자 가설 ("정답 topic 이 STM 못 올라가서 retrieval 미스") 검증 도구. `dormant_top_topic_share` 가 가설의 "집중도" 를 직접 측정.
- **행동 영향**: v3.3.3-4 의 `promotion_threshold=0.3` + dormant LTM safety net 이 설계 변경의 직접 motivation. 모든 hi-em-full method 에 자동 적용되어 비교 가능.

---

## 7c. Evidence→topic concentration audit (segmentation 진단)

- **무엇**: LoCoMo 의 question 별 evidence dia_id 들이 *몇 개의 Hi-EM topic 으로 흩어졌는지* 를 post-hoc 으로 측정. importance / retrieval policy 와 분리된 *순수 segmentation 품질* 신호.
- **어디**: `scripts/analyze_evidence_topics.py` (CLI). `scripts/experiment.py` 의 `run_method` 가 hi-em 류 run 완료 직후 자동 호출.
- **언제 emit**: `working_state/ltm/<conv>_<method>/<conv>.jsonl` (dia_id → topic_id 매핑) 이 존재하는 모든 hi-em run. 비-hi-em (rag/sliding/full) 은 silently no-op.
- **산출물 (per run)**:
  - `<run_dir>/per_question_evidence_topics.csv` — **n_ev≥2 question 만** (n_ev=1 은 segmentation 평가 trivial: 단일 evidence 는 항상 1 topic). LoCoMo 1986 question 중 약 423 행 (multi-hop / 일부 sh/temp/od/adv 의 multi-evidence 케이스만). 컬럼: `qid, cat, n_ev, n_ev_found, n_topics_used, evidence_dia_ids, evidence_topics, topic_breakdown`. **2026-05-14**: 이전엔 n_ev=1 포함 1986 행이었음. 변경 이유 — n_ev=1 은 trivially `n_topics_used=1` 이라 overall mean topics/q 를 1 쪽으로 끌어내려 segmentation 신호 희석. `evidence_topic_summary.json` 의 qtype 별 stats 는 원래부터 n_ev≥2 필터였음. 기존 24개 CSV 도 일괄 정리됨.
  - `<run_dir>/evidence_topic_summary.json` — qtype 별 (mh / sh / temp / od / adv) `_topics_per_q` (= 평균 `n_topics_used`, n_ev≥2 question 만), `_all1_pct`, `_mean_n_ev`, `_n_q`.
- **REPORT 노출**: `mh_topics/q`, `sh_topics/q`, `temp_topics/q`, `od_topics/q` 컬럼. 작을수록 segmenter 가 evidence 를 적은 수 topic 에 co-locate.
- **해석**:
  - **mh_topics/q** 가 ~3 부근 (evidence 가 거의 다른 topic) + acc 낮음 → multi-hop evidence 가 본질적으로 다른 session/dia 분산. segmentation HP 로 추가 개선 여지 작음 → importance policy 가 책임.
  - **sh_topics/q** 가 1 에서 멀어짐 → 같은 dia 인접 evidence 가 다른 topic 으로 갈리는 *false-positive boundary*. segmentation HP (cos_threshold, sigma0_sq 등) 로 직접 개선 가능.
  - same-dia split vs cross-dia split 분리는 CSV `evidence_dia_ids` 로 사용자 측에서 추가 분석.
- **왜**: HP sweep (segmentation) ↔ importance policy 의 *책임 귀속* 분리. 이전엔 acc / H@k 로 두 가지가 합성되어 보였음.
- **행동 영향**: 모든 hi-em-full method 에 자동 적용. 새 sweep 마다 별도 명령 없이 REPORT 에 컬럼 노출.

---

## 8. 산출물 디렉토리 구조

2026-05-08 통합 후 — top-level **2개**:

```
outputs/                  # 모든 active/historical generated
├── experiments/             # experiment.py 산출 (sweep / ablation / comparison). self-contained.
│   └── <date>_<label>/<run_label>/{run.log, exit_code.txt, stm_topk.json, results/experiments/<exp_id>/}
├── runs/                    # standalone run_experiment.py 데이터 (default --results-root).
│   ├── <date>/<exp_id>/     # 옛 sweep 들의 raw exp 데이터 (날짜별 누적)
│   └── _misc/               # smoke / scratch
├── reports/                 # 독립 분석 MD (committed)
└── design/                  # 설계 문서 (committed)

archive/                  # 의도적으로 버린 것 (Phase 4 era — RAG 에 패배 후 폐기)
├── 2026-04-26-baseline/     # Phase 4 reference (RAG 0.62 vs Hi-EM 0.56)
├── 2026-04-28/              # Phase 4 follow-up (24 runs)
├── 2026-04-29/              # (14 runs)
├── 2026-04-30/              # (1 run)
└── README.md                # "왜 버렸는지"
```

**원칙**:
- 새 experiment = `scripts/experiment.py` → `outputs/experiments/<name>/` (self-contained)
- standalone debug = `run_experiment.py` 직접 호출 → `outputs/runs/`
- 시간 흐른 experiment 데이터 = `outputs/runs/<date>/` 에 누적 (정리 X — 자료)
- `archive/` = *의도적 폐기* 만. "오래됐다" 가 아니라 "더 이상 쓰지 않는다고 결정함" 의 표시. baseline + Phase 4 era 만 현재.

**없어진 것** (2026-05-08 정리):
- top-level `results/` → `outputs/runs/` 로 이동
- `archive/legacy_runs/` → 5월 데이터는 `outputs/runs/`, Phase 4 데이터는 `archive/<date>/` 로 분리
- `outputs/legacy/` → orphan smoke jsonl 삭제 (가치 없음)

git 정책: `outputs/experiments/<name>/REPORT.md`, `outputs/reports/*.md`, `outputs/design/*` = committed. `outputs/runs/`, sweep 안의 jsonl 데이터 = gitignored.

---

## 9. SEM2 cold-start gating / dynamics / σ² / fresh-baseline 계열 (v3.3.5~8, 2026-05-17)

- **무엇**: idx374 segmentation 진단에서 파생된 4개 cross-cutting 메커니즘.
  - `f_is_trained` gating (v3.3.5): untrained topic(transition_count<min_transitions_for_pe) = fresh slot 과 동일 L0 → likelihood 동률, prior(λ) 결정. SEM2 복원.
  - persistence+replay dynamics (v3.3.6, `TopicV336`): untrained=직전임베딩(identity), per-topic 독립 EventRNN, topic history 전체 replay(n_epochs), `rnn_ready` ≠ `f_is_trained`.
  - map_variance σ² (v3.3.7): `σ²=(ν₀var₀+n·v)/(ν₀+n+2)`, n≥2 즉시. `pe_var_min_samples` gate 폐기.
  - fresh-baseline `pe_prior` (v3.3.8): L0 가 cos_threshold 아닌 pe_prior(chance PE)에서 도출. non-prev topic = f0-likelihood(SEM2 `k0≠k_prev`).
- **어디**: `src/hi_em/sem_core_v33{5,6,7,8}.py`, `src/hi_em/topic_v336.py`, orchestrator `version` dispatch + HP(`min_transitions_for_pe`,`rnn_n_epochs`,`rnn_ready_min_transitions`,`rnn_max_history`,`seed`,`pe_var_df0`,`pe_var_window`,`pe_prior`), `tests/test_sem_core_v33{5,6,8}.py`, `scripts/inspect_longmemeval_segmentation.py --version`.
- **왜**: methodology v3.3.5~8.md + decision-log 2026-05-17. 핵심 = v3.3.4 의 young-topic centroid 처벌(chicken-and-egg) → SEM2 충실 복원 연쇄.
- **행동 영향**: v3.3.5~8 전부. v3.3.8 default `pe_prior=1.0`(원칙값)은 idx374 mega-collapse — **작동값은 벤치마크 calibration 대상, N=1 production default 금지**.
- **한계**: v3.3.7 은 #14|15 경험적 반증(보존). v3.3.8 fresh-baseline 은 embedding-공간 의존 HP(SEM2 단일상수 불가). non-prev f0 는 generic-opener 취약.

## 10. 재현성 — segmenter seed (필수, 2026-05-17)

- **무엇**: EventRNN/per-topic 모델 random init 이 v3.3.4/5 까지 unseeded → 동일 입력 다른 분절(앞선 v3.3.4 REPORT 수치 불일치의 정체). v3.3.6+ 는 `seed`(per-topic `manual_seed(seed·100003+topic_id)`, RNG snapshot/restore) 로 결정적.
- **어디**: `TopicV336.__init__`, `HiEMSegmenterV33{6,7,8}(seed=...)`, orchestrator `seed` param(default 0).
- **왜**: CLAUDE.md "모든 randomness seed 고정". 논문 재현성 필수.
- **행동 영향**: v3.3.6~8. v3.3.4/5 는 unseeded(결과 해석 시 명시). 실험 시 seed 보고 의무.
- **한계**: v3.3.4/5 소급 적용 안 함(별 버전).

## 11. evidence_recall@K — retrieval 평가 metric (evidence_cohesion 폐기, 2026-05-17)

- **무엇**: `evidence_cohesion=1`(전 evidence 단일 topic) **폐기**. 대체 = `evidence_recall@K` = ★turn 담은 모든 topic 이 importance 상위 K 에 드는가. **Primary = topic-level**, 보조 = session-level(LongMemEval label 호환). 보조 진단 = `topic_precision@K`, prefill token cost.
- **어디**: 평가 스크립트(향후), `scripts/inspect_longmemeval_s.py`(증거/distractor 분리: `answer_session_ids`+`has_answer`).
- **왜**: evidence 가 여러 scene 에 본질 분산(idx374) → 단일 topic 강요 = mega-merge = SEM atomicity 위배. decision-log 2026-05-17.
- **행동 영향**: 모든 LongMemEval segmentation 평가. longmemeval_s 의 session_id 는 SEM 재발견 target 아니라 외생 hard boundary(단 v3.3.9 는 emergent 방향 — session_id 미사용).
- **한계**: topic↔session 단위 불일치 시 두 지표 병행 보고.

## 13. Streaming baseline segmenter — `StreamingTextTiling` (2026-05-20)

- **무엇**: `src/hi_em/baselines/texttiling_streaming.py` 의 `StreamingTextTiling`
  class. `push(utterance) → list[int]` (이번 호출에서 새로 확정된 boundary
  utterance index 1-based), `flush() → list[int]` (대화 종료시 잔여 gap 처리).
  block-cosine + Welford running mean/std threshold + one-sided causal depth +
  min_gap suppression. per-turn 실질 O(w).
- **어디**: `src/hi_em/baselines/`, 진입점 runner `methods/texttiling/online_streaming.py`
  (Def-DTS 번들 jsonl 3 dataset 로드 — tiage/dialseg711/superseg, segeval 직접 호출).
- **왜**: 기존 `run_texttiling_prefix.py` 는 매턴 nltk fresh 호출 = O(t)/turn
  → online baseline 의 "latency 비교값" 정책 위배. 진짜 streaming 버전을 별도
  method 명 (`TextTiling-online-streaming`) 으로 신설 (codex:rescue 2026-05-20,
  decision-log 참조).
- **행동 영향**: TextTiling 비교 시 *3종 구분 강제* — offline (NLTK 전체), prefix-
  recompute (NLTK 매턴 fresh), streaming (자체 구현). 같은 표에 섞지 말 것.
- **한계**: 원본 NLTK TextTiling 점수 재현 안 함 (causal running threshold ≠
  offline global threshold). Pk/WD/F1 = INDICATIVE; latency 가 비교값. class
  default w=10/k=6 (NLTK 호환), runner default w=5/k=3 (tiage 짧은 대화 대응).

## 12. LongMemEval-S 데이터 + inspect 도구 (2026-05-17)

- **무엇**: `benchmarks/LongMemEval/data/longmemeval_s_cleaned.json`(277MB, 500Q, HF `xiaowu0162/longmemeval-cleaned`). oracle(증거세션만)과 달리 질문당 ~48세션(증거+distractor), ~245 user턴. `has_answer`/`answer_session_ids` 동일 스키마. **idx 정렬이 로컬 oracle 과 다름 — qid 로 매칭**(s idx0=oracle idx286). distractor 는 ShareGPT/UltraChat 재활용 → 세션당 user턴 0~66 불규칙(증거세션은 균질).
- **어디**: `scripts/inspect_longmemeval_s.py` (`--qid`/`--idx`, 증거세션만 ★ 출력, `--with-distractors`, `--out`). gitignored(`benchmarks/LongMemEval/` 전체).
- **왜**: oracle 은 retrieval 난이도 제거판 → retrieval 병목(distractor 에서 증거 찾기) 평가 불가. s 필요.
- **행동 영향**: 향후 evidence_recall@K 평가는 s 기반. oracle 은 segmentation/추론만.
- **한계**: 질문당 ~245턴 × 500 = 무거움 → qtype stratified subset 권장. 로컬 oracle 은 cleaned 이전 구버전 가능(qid 매칭 필수).

---

## 작성 규칙

새 인프라/cross-cutting 설계 항목 추가 시:
1. 위 형식 (무엇/어디/왜/행동영향/한계) 으로 한 섹션 추가.
2. 어느 버전이 영향받는지 명시.
3. 실험에서 이 인프라가 *결과에 영향을 줬으면* 반드시 기록 (예: `--no-thinking` 누락으로 sweep 망친 사례).
