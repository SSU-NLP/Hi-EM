# CLAUDE.md — Hi-EM 프로젝트 규칙

## 중요 의사결정은 항상 codex로 답변 (최상위 규칙)

설계 옵션 선택, 수식 확정, 모듈 구조 결정, 알고리즘 트레이드오프 평가 등 **프로젝트의 향후 방향에 영향을 주는 의사결정 질문**은 Claude가 직접 답하지 않고 반드시 `codex:rescue` 서브에이전트로 위임해 한국어로 답변한다.

판단 기준 (다음 중 하나라도 해당하면 codex 위임):
- "어느 방법이 더 나은가" / "무엇을 선택해야 하나" / "이게 적절한가" 형태의 비교·판단 질문
- 수식·모델 설계 변경에 대한 권고
- mega-topic, segmentation, prior/likelihood 등 핵심 알고리즘 결정
- decision-log에 남길만한 결정

단순 코드 수정·디버그·문서 업데이트·요약은 Claude가 직접 처리해도 된다.

위임 시 사용자가 이전 codex 스레드의 컨텍스트를 이어가는 후속 질문이면 `--resume`을 붙인다.

## SEM 계승 원칙 (최상위 설계 제약)

Hi-EM은 SEM / SEM2 계승 모델이다. **SEM(원논문) / SEM2(`nicktfranklin/SEM2`)에 존재하지 않는 메커니즘은 합리적 근거 없이 도입하지 않는다.**

대상 예시 (SEM에 없으므로 default = 도입 안 함):
- vMF likelihood, multi-prototype topic, per-topic concentration κ 학습
- bounded cosine 외의 새로운 likelihood 함수
- 학습형 dynamics 외의 새로운 prediction error 정의
- SEM이 명시적으로 "쓰지 않는다"고 가정한 임의의 새 component

도입을 검토하려면 다음 셋을 모두 충족해야 한다:
1. **SEM에 왜 없는지** (의도적으로 빠진 건지, 단순 미구현인지) 명시한다.
2. 추가했을 때 얻는 이점이 SEM의 철학(sCRP / Bayes / local MAP / scene dynamics)과 **충돌하지 않음**을 보인다. 충돌하면 어느 쪽을 우선할지 명시한다.
3. `context/06-decision-log.md`에 근거 + 날짜를 append한다.

신규 component 제안이 들어오면 Claude의 default 응답은 **"SEM에 있나?"** — 없으면 위 세 단계의 정당화 부담을 그 제안에 얹은 채로만 검토한다.

이유: SEM 계승이라는 정체성을 잃으면 Hi-EM은 그저 "임의로 조합된 retrieval heuristic"이 된다. 새 메커니즘은 SEM에서 빠진 기능을 복원하거나, SEM 가정의 한계를 명시적으로 인정한 위에서만 추가한다.

## Retrieval 은 importance score 만으로 한다 (최상위 설계 제약, 2026-05-11)

**Hi-EM 의 query-time retrieval 은 오직 SEM importance score 기반이다. Query embedding 과의 cosine matching (RAG-style) 을 retrieval 단에 도입하지 않는다.**

이유:
- SEM 정신상 "지금 working memory 에 무엇이 있어야 하는가" 는 importance score 가 결정. 그 결정이 이미 끝난 다음, query-time 에 cosine 으로 다시 ranking 하면 두 메커니즘 (SEM importance + RAG cosine) 이 *경쟁* 하며 정합성 깨짐.
- importance policy 가 부실해서 cosine 으로 우회하면, 실제 문제 (importance 자체 부정확) 가 가려짐. 근본 해결 미룸.
- SEM 모든 변형 (v3.x 계열) 의 retrieval 은 STM 안 topic 의 turn 들을 *통째로* prefill 하는 atomicity + importance ordering 으로만 해결되어야 함.

구체적 적용:
1. **STM 안 retrieval**: STM 의 모든 turn 또는 importance 상위 K topic 의 turn 을 통째로 prefill. **각 turn 에 대한 query cosine 계산 / ranking 금지.**
2. **LTM fetch (라운드 중간 / promotion miss 시)**: 필요한 topic_id 를 *정확히 일치* 시켜 가져옴. cosine top-N 같은 fallback 금지. "어떤 topic_id 가 필요한가" 자체는 segmentation (현재 turn 의 assigned topic) 또는 importance 기반.
3. **Episode-level retrieval (v3.3.3-4 계열)**: episode 단위 atomicity 는 유지. 단 episode score 계산에서 query cosine 항 제거. score = topic importance + episode 내부 salience (importance EMA 등) + recency 보정. **query 와의 의미 거리 사용 금지.**
4. **Dormant LTM safety net**: cosine top-N 으로 dormant turn 끌어오는 정책 폐기 대상. 대안 — promotion threshold 인하, importance policy 자체 강화, topic_id 기반 explicit fetch.

도입 검토 예외 조건:
- 만약 cosine 을 *재도입* 하려면 (예: query-aware salience 추정 같은 SEM 외 readout), CLAUDE.md SEM 계승 § 의 3-step 정당화 + 본 규칙 명시적 예외 표시 + decision-log 모두 필수. **default 답변은 "안 한다."**

이 규칙은 v3.3.3-4 의 `_episode_rerank_prefill()` 의 cosine-based scoring 을 폐기 대상으로 만들고, `dormant_ltm_top_n` 의 cosine top-N fallback 도 폐기 대상으로 만든다. 전체 retrieval 정책 재설계가 본 규칙 도입의 직접 결과.

decision-log 2026-05-11 entry 참조.

## Step 완료 프로토콜 (최상위 강제 규칙)

### 1단계: 3-angle Self-Audit (check_step_done 실행 **전**)

`check_step_done.py`는 길이·키워드 수준의 피상 검증만 한다. 그 전에 반드시 다음 세 각도에서 **자기질문-자기답변**을 수행한다:

1. **구조 이해** — 결과물(논문/데이터/설계)의 **형태와 구성요소의 역할**을 스스로에게 질문하고 답할 수 있는가?
2. **동작/inference 이해** — 알고리즘·처리 흐름·검색 로직을 끊김 없이 설명할 수 있는가? 수식이라면 각 변수의 역할과 식 간 연결을 정확히 복원할 수 있는가?
3. **설계 방향 이해** — 이 결과물이 Hi-EM의 다음 결정(옵션 선택/수식 확정/모듈 설계)에 어떻게 연결되는가? 무엇이 열려 있는가?

**최소 3 Q&A per angle**. 답하지 못하거나 근거가 약한 항목은 **해당 결과물 파일의 "검증 미해결" 섹션에 명시**한다 (제거 금지, 솔직 기록).

Self-audit 자체는 세션 내에서 진행하고 별도 파일로 저장하지 않아도 된다. 단, **gap으로 식별된 것은 반드시 결과물에 기록**한다.

### 2단계: `check_step_done.py` 실행

```bash
python scripts/check_step_done.py
```

- **exit code 0이 나올 때까지 Step 완료 처리 금지.**
- FAIL이 남아있으면 원인을 수정하고 스크립트를 재실행한다. **통과할 때까지 반복한다.**
- WARN은 허용 가능하지만, 가능하면 해결한다.
- 검증 없이 `[x]` 처리하거나 커밋하지 않는다.
- Step에 대한 검증 로직이 `STEP_CHECKS`에 없으면 추가한 뒤 진행한다.

### 금지

- self-audit 건너뛰고 곧장 `check_step_done.py`만 돌려 통과시키기 금지. 스크립트 통과는 **필요조건이지 충분조건이 아니다.**

## 실험 결과 저장 (최상위 강제 규칙)

**어떤 벤치마크든 (LoCoMo / TIAGE / TopiOCQA / LongMemEval / 기타) 모든 실험 결과는 `outputs/experiments/<name>/REPORT.md` 한 곳에 저장한다.** 다른 위치 (`outputs/reports/`, `outputs/` 루트, 임의 디렉토리) 에 실험 결과 REPORT 를 만들지 않는다.

### REPORT.md 필수 구성

자동 생성된 표만으로는 부족하다. 모든 REPORT.md 는 다음 섹션을 **자세하고 명확하게** 포함한다:

1. **실험 setup** — 목적 / 데이터 (path + n_conv/n_turn/n_question) / 방법 list / HP (실제 사용한 default + override) / seed / metric 정의 / 비교 baseline.
2. **결과 표** — mean ± std (3-run 이상) + 핵심 지표 + best 강조.
3. **해석** — 숫자가 의미하는 것, 왜 이런 패턴이 나오는지, 어디까지가 noise 범위 안인지.
4. **판정** — iteration 내에서 method 별 향상/동일/회귀 분류, 다음 iteration 결정 (재구현 / 다음 단계 / 보류).
5. **한계 / 검증 미해결** — HP 적합성, seed 부재, atomicity, 표본 크기 등 솔직 기록 (제거 금지).

자동 생성된 짧은 표만 남긴 REPORT 는 미완성으로 간주.

### outputs/reports/ 의 역할 (정정)

`outputs/reports/` 는 **실험 결과 저장 금지**. cross-experiment methodology 비교, design rationale doc, 일반 분석 등 **단일 실험에 귀속되지 않는 문서** 만 둔다. (이전 정책에서 "TIAGE/TopiOCQA segmentation 비교"를 `reports/` 에 두던 관행은 본 규칙으로 대체됨.)

### Single-script 실험에도 적용

`scripts/run_*.py` 형태의 단독 script (예: `run_tiage_full_compare.py`) 도 default output 을 `outputs/experiments/<name>/REPORT.md` 로 둔다. `--name` 인자 + 그 폴더 안에 self-contained.

## 산출물 / 실험 디렉토리 사용법 (canonical, 최상위 규칙)

**용어**:
- *experiment* = 이름 붙은 여러 *run* 의 집합 (sweep, ablation, comparison 모두 동일).
- *run* = method × HP 한 조합 (`run_experiment.py` 한 번 호출).

### 1. 새 experiment 실행 = `scripts/experiment.py` 한 entry 만

```
uv run python scripts/experiment.py \
  --name <date>_<descriptor> \
  --benchmark locomo --data <path> \
  --workers 100 --no-thinking \
  --method "method[:label][@k1=v1,k2=v2]" [--method ...] \
  [--limit N --stratify]
```

규칙:
- 새 sweep / ablation / comparison 마다 별도 shell 또는 aggregator 스크립트 작성 **금지**.
- 모든 옵션 (HP override, baseline 추가 등) 은 `--method` spec 으로 표현 (`method[:label][@k=v,k=v]`).
- 같은 `--name` 으로 재실행하면 `<run_dir>/exit_code.txt = 0` 인 run 자동 skip → 죽었다 살려도 안전.
- 실험 끝나면 `outputs/experiments/<name>/REPORT.md` 자동 생성 (acc + qtype 5종 + T1/T2/T3 + T1max/T2max/T1var + n_topics + gen_p50 + wall).
- `--aggregate-only` 로 결과 안 돌리고 표만 재생성 가능.

자세한 spec 문법 → `context/methodology/infrastructure.md` § 7.

### 2. 단일 ad-hoc run = `scripts/run_experiment.py`

디버그 / smoke / 일회성 단발 실행. default `--results-root` = `outputs/runs/`. 자료가치 있으면 `outputs/runs/<date>/<exp_id>/` 에 자동 누적.

### 3. 산출물 디렉토리 구조

```
outputs/                  # 모든 active/historical generated. top-level 1개.
├── experiments/             # experiment.py 산출 — self-contained
│   └── <name>/<run_label>/{run.log, exit_code.txt, stm_topk.json, results/...}
├── runs/                    # run_experiment.py 단독 실행 (날짜 subdir 누적)
├── reports/                 # cross-experiment methodology/design 분석 MD only. 실험 결과 저장 금지 (위 § "실험 결과 저장" 참조).
└── design/                  # 설계 문서 (committed)

archive/                  # *의도적으로 폐기* 한 것만 (시간 흐름 무관)
├── 2026-04-26-baseline/     # Phase 4 reference (RAG 에 패배 → 폐기, 보존)
├── 2026-04-{28,29,30}/      # Phase 4 follow-up (같이 폐기)
└── README.md                # 왜 버렸는지 기록
```

**원칙**:
- "오래됐다" 와 "폐기됐다" 는 **다른 것**. archive/ 는 *폐기 결정* 했을 때만. 단순히 시간 지나서 뒤로 밀린 데이터는 `outputs/runs/<date>/` 에 그대로 둔다.
- 새 dir 만들지 말고 위 4 카테고리 (`experiments/`, `runs/`, `reports/`, `design/`) 중 하나에 넣기. 의미 모호하면 `outputs/runs/_misc/`.
- archive 에 새 항목 추가할 땐 `archive/README.md` 에 *폐기 사유 + 날짜* 한 줄 append. 사유 없는 archive 추가 금지.
- `outputs/experiments/<name>/` 안의 데이터는 self-contained. 한 experiment 의 모든 정보 (config, log, hypothesis, summary, REPORT) 가 이 한 디렉토리 안에서 답이 나와야 함.

### 4. `scripts/legacy/` — 읽기 전용

옛 per-experiment shell (`run_*sweep.sh`) + 옛 aggregator (`aggregate_*.py`) 보관됨. 참고용. **새 스크립트 추가 금지**, 기존 파일도 수정하지 않는다 — 현재 유효한 entry 는 `scripts/experiment.py` 와 `scripts/run_experiment.py` 둘 뿐.

### 5. git 정책

- committed: `outputs/experiments/<name>/REPORT.md`, `outputs/reports/*.md`, `outputs/design/*`, `archive/<name>/README.md`
- gitignored: `outputs/runs/`, `outputs/experiments/*/*/results/`, `outputs/experiments/*/*/run.log`, `outputs/experiments/*/*/stm_topk*.json*`, `archive/*/ltm/`, `archive/*/outputs/*.{jsonl,json,wandb-run-id}`
- `.gitignore` 변경은 cascade 검사 대상.

### 6. wandb logging (필수)

모든 실험은 wandb 로 자동 로깅된다 (project convention 2026-05-08~). `experiment.py` / `run_experiment.py` 가 자동으로 wandb run 을 시작하고, round 별 metric + final summary 를 push 한다.

**1회 setup**: `uv run wandb login` 또는 `.env` 에 `WANDB_API_KEY=...` 추가. 인증 안 돼 있으면 wandb init 가 실패 catch 되어 *no-op* 으로 fallback (실험은 진행되지만 logging 안 됨 — `[wandb] init failed` 경고).

**opt-out**: 특정 run 만 끄고 싶으면 `WANDB_MODE=disabled uv run python scripts/experiment.py ...` 로 prefix.

**확인**: 실행 후 `outputs/experiments/<name>/<label>/run.log` 의 wandb 라인 또는 wandb 웹 UI 에서 `hi-em-phase4` (default `--wandb-project`) 확인.

## `context/methodology/` 가 최우선 관리 대상 (최상위 규칙)

**가장 1순위로 관리한다.** 코드 수정 / 설계 결정 / 인프라 변경 시 가장 먼저 확인·갱신해야 할 디렉토리.

이유: 버전(v1/v2/v3.1.1/v3.2.1/v3.3.1) 별 *알고리즘 수준 정의* 와 cross-cutting 인프라 결정이 모이는 단일 진실 원천. 다른 docs (handoff, plan, decision-log) 는 *현재 진행 상황* 을 기록하지만, methodology/ 는 *설계 자체* 가 무엇인지 기록한다.

기록 범위 (사소해 보여도 빠짐없이):
- 알고리즘 수식 / score 식 / update rule
- topic state 의 모든 필드, hyperparameter default
- SEM 계승 측면 (있는 것 / 없는 것 / 변형한 것)
- 알려진 한계 + 변형 후보 (적용 안 했어도 "고려 중인 변형" 섹션에 누적)
- cross-cutting 인프라 (`EncoderCache`, `HiEMConvCache`, encoder lock, LLM 호환 플래그(`--no-thinking`), retrieval policy, STM atomicity 등)

규칙:
- 버전마다 1 파일 (`vX.md`). 새 버전 추가 시 직전 버전 파일을 템플릿 삼아 같은 구조 유지.
- 버전 간 공유 항목은 `infrastructure.md` 에 추가. 항목 형식: 무엇 / 어디(file:line) / 왜 / 행동 영향(어느 method) / 알려진 한계.
- *알고리즘 의미가 바뀌는* 변경은 새 버전 분리. 단순 성능/캐시 최적화는 같은 파일 안 "고려 중인 변형" 또는 "변경 이력" 섹션 누적.
- 어떤 파일을 수정했든, 그 변경이 알고리즘·infrastructure 측면에서 의미가 있으면 *반드시* 해당 methodology 파일에 한 줄 이상 반영.
- 의미가 모호하면 default = 기록한다. 기록은 cheap, 망각은 expensive.

cascade 검사는 (아래 § 참조) `context/methodology/` 를 가장 먼저 확인 대상으로 둔다.

## 파일 수정 시 최신성 cascade 검사 (필수)

Claude Code가 파일 1개를 수정·생성·삭제할 때마다 **다른 파일에 영향 가능성이 있는지 즉시 검사**한다. 영향받을 가능성이 있는 파일은 **사용자에게 한꺼번에 제시하고 같이 업데이트할지 묻는다**.

확인 대상 (우선순위 순):
- `context/methodology/*` — **최우선** (알고리즘·인프라 변경 시). 위 § 참조.
- `README.md` — 디렉토리 구조, 외부 레포, gitignored 목록, 현재 상태
- `plan.md` — 체크박스, Phase 진행률, 결과 수치
- `handoff.md` — 현재 상태, 다음 할 일, 마지막 업데이트 날짜
- `context/04-benchmarks.md` — 데이터 / 평가 축 변경 시
- `context/03-architecture.md` — 모듈·파일 추가/삭제·이름 변경 시
- `context/06-decision-log.md` — 설계 결정 변경 시 (append-only)
- `context/sem-equations.md` — SEM 식 관련 작업 시
- `report.md` — Phase 결과·미해결 사항 변경 시
- `.gitignore` — 새 파일 패턴 추가/제거 시

검사 방법:
- 변경 키워드(파일명/모듈명/Step 번호/Phase 결과)를 `grep -rn` 으로 다른 docs에서 검색
- 발견된 파일 + 무엇이 stale해 보이는지 사용자에게 보고
- 사용자 응답 (전체 / 일부 / skip) 받은 후 일괄 수정

목적: docs 간 불일치 누적 방지. 한 파일만 고치고 다른 곳 잊으면 다음 세션 / 협업자가 잘못된 정보로 작업.

## 장기 실행 작업 진행 점검 (필수)

10분 초과로 예상되는 작업(실험·학습·평가 등)은 **시작 후 10분이 지나면 반드시 한 번 진행 상황을 확인**한다. background task / `run_in_background=true` / 별도 프로세스 모두 적용.

방법:
- `ScheduleWakeup` (delaySeconds=600 이하) 으로 10분 내 자가 점검 예약
- 체크 시점에 stdout 마지막, exit status, results 디렉토리 (`results/experiments/<exp-id>/checkpoints/latest.json` 또는 `summary.json`) 셋 다 확인
- 진행 정상이면 다음 점검(또 10분 이내) 예약, 정체/오류면 **즉시 사용자에게 보고**

이유: vLLM 멈춤·STM 폭주·OOM·import error 등 silent failure가 발생해도 사용자가 모르고 기다리는 일을 막는다. 한 번 시작하고 던져두지 않는다.

## 환경 분리

- `setup_colab.ipynb`는 항상 `.gitignore` 유지. git에 커밋하지 않는다.
- `setup_colab.ipynb`는 로컬/Colab 공용 setup notebook으로 유지한다.
- 그 외 모든 파일은 Colab 전용 의존 없이 로컬 기준으로 동작해야 한다.
  - `from google.colab import drive`, `drive.mount`, `/content/` 경로를 기본 파일에 넣지 않는다.
  - Colab 전용 코드가 꼭 필요하면 `IS_COLAB` 분기 안에만 작성한다.
- 경로는 하드코딩 금지. `git rev-parse --show-toplevel` 또는 상대경로 사용.

## Notebook 실행 정책

- **모든 실험 notebook(`notebooks/*.ipynb`)은 `setup_colab.ipynb` 선행 실행을 가정한다.**
- 실험 notebook 안에 환경 셋업(repo clone, 벤치마크 clone, 패키지 설치, 데이터 다운로드, 모델 다운로드) 로직을 **중복으로 넣지 않는다.** setup_colab이 단일 책임자.
- 실험 notebook의 첫 셀들은 `setup_colab` 사전 조건이 만족됐는지 **검증만** 하고, 부족하면 명확한 에러 메시지(`'setup_colab.ipynb 먼저 실행'`)로 실패시킨다.
- 이유: 환경 셋업 로직이 여러 notebook에 흩어지면 동기화 부담 + 혼란. setup_colab만 유지·업데이트하면 모든 실험이 따라옴.

### Notebook ↔ Script 분리 원칙 (portability)

- **모든 실험 로직은 `scripts/*.py`에 둔다.** notebook은 그 스크립트를 `subprocess.run(['python', 'scripts/X.py', ...])`로 호출하는 **얇은 wrapper**일 뿐이다.
- `notebooks/` 디렉토리 통째로 삭제해도 프로젝트가 그대로 동작해야 한다 — 로컬 GPU·다른 환경 전환 시 `python scripts/X.py` 직접 실행으로 모든 실험 가능.
- notebook이 추가로 갖는 가치: Colab kernel 연동, IPython 출력 렌더링(`display(Markdown(...))`), 셀 단위 인터랙티브 디버깅. 이 가치 외엔 `.py` 스크립트로 옮긴다.
- `setup_colab.ipynb`만 예외 — 환경 셋업은 본질적으로 노트북 형식이 자연스러워 그대로 둠 + `.gitignore`로 제외.

### Tracking 정책

- `setup_colab.ipynb`: **gitignored** (Colab 전용 환경 셋업, 일회성 도구)
- `notebooks/*.ipynb` 그 외 모두: **git tracked** (연구 기록, 협업자 공유). 단 위 portability 원칙 위반 시 무효 → script로 분리.

## 코딩 스타일

- Python 3.10+ (match statement, union types 활용)
- Type hint 필수 (`from __future__ import annotations`)
- Docstring: Google style
- Line length: 100

## 파일 조직

- 한 모듈 = 한 책임
- 순환 import 금지
- 코어 코드: `src/hi_em/`
- 실험/스크립트: `scripts/`
- 테스트: `tests/`

## 커밋 규칙

- **Claude Code는 `git add`/`git commit`/`git push`를 직접 실행하지 않는다.** 변경이 준비되면 사용자가 복사해서 실행할 수 있는 **명령어만 제시**한다. 커밋 실행 권한은 사용자에게 있다.
- 한 커밋 = 한 논리적 단위
- 제목 50자 이내, 본문은 이유 중심
- prefix: `feat`, `fix`, `docs`, `refactor`, `test`, `exp`

## 문서화

- 새 모듈 추가 시 `context/03-architecture.md` 반영
- 설계 결정 시 `context/06-decision-log.md`에 근거 + 날짜 append
- 실험 시작 전 `templates/experiment-log.md` 복사해서 로그 생성

## 테스트

- pytest
- 필수 테스트: sCRP 계산, topic assignment, centroid 업데이트
- 재현성: 모든 randomness에 seed 고정

## 외부 레포 사용

- `benchmarks/*`: 읽기 전용
- `SEM/` (= SEM2, `nicktfranklin/SEM2` current build): 참조 전용, 코드 복사 금지

## 금지 사항

- 메인 LLM fine-tuning 금지
- TensorFlow/Keras 사용 금지 (PyTorch만)
- SEM2 코드 직접 복사 금지 (참조만 허용)
- 설계 문서 업데이트 없이 구조 변경 금지
- 벤치마크 분석 없이 설계 확정 금지
