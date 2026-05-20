# `methods/` — baseline 방법 정리 (offline 원본 / online 수정본)

각 baseline 의 **원본(offline)** 과 **Hi-EM 수정본(online, prefix-causal)**
진입점을 한 곳에 모은 디렉토리. 현재 범위: **TextTiling, BayesSeg** 둘만
(2026-05-20 사용자 결정; 필요 시 추후 확장).

## 설계 원칙 (A: wrapper, 코드 복사 없음)

- 원본 코드는 `benchmarks/superdialseg/src/super_dialseg/models`
  (Coldog2333/SuperDialseg) 에 있고 **read-only**(CLAUDE.md) → 여기로
  **복사하지 않음**. `offline.py` 는 그 원본 알고리즘을 *호출/재현 동작*
  만 함 (TextTiling=nltk 동일 파라미터, BayesSeg=빌드된 `segment`
  스크립트 그대로).
- `online.py` 는 검증된 단일 진실원천 `scripts/run_*_prefix.py` 를
  그대로 실행하는 얇은 진입점 (로직 중복 없음).
- offline·online **모두 같은 harness**: 데이터 = Def-DTS 번들
  (`tiage/dialseg711/superseg`), metric = `autoseg` segeval Pk/WD + F1,
  `Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD)` → Hi-EM 내 offline↔online
  apples-to-apples 비교.

## 구성

| 경로 | 무엇 |
|---|---|
| `texttiling/offline.py` | 원본 TextTiling, **전체 대화**(미래 포함). nltk w=10,k=6. |
| `texttiling/online.py`  | prefix-causal(U1..Ut) **prefix-recompute** (매턴 nltk fresh 호출, O(t)/turn), AUXILIARY → `scripts/run_texttiling_prefix.py` |
| `texttiling/online_streaming.py` | prefix-causal **streaming** (block-cosine incremental, Welford running threshold, **O(w)/turn**), 3 dataset (tiage/dialseg711/superseg), Def-DTS 는 데이터 로드만, metric 은 segeval 직접, AUXILIARY (별도 method 명: `TextTiling-online-streaming`). 2026-05-20 신규. |
| `bayesseg/offline.py`   | 원본 SuperDialseg BayesSegmenter, 전체 대화, `segment dp.config`(`-num-segs 7`) |
| `bayesseg/online.py`    | persistent JVM·native-K·prefix, AUXILIARY → `scripts/run_bayesseg_prefix.py` |
| `greedyseg/online_delay2.py` | **GreedySeg-online-delay2** (BERT bounded-lookahead, delay=2). 원본 SuperDialseg `GreedySegmenter` 의 score 공식·HP·argmin greedy 선택 그대로 보존. cuda/mps/cpu device-agnostic. 2026-05-21 신규. *codex 검증 = 5행 핵심표 가능*. |

**TextTiling 3종 구분 (혼동 주의)**:
- `offline.py` (NLTK 원본, 전체 대화, global threshold) — *원본 baseline*.
- `online.py` (NLTK 호출만 prefix 로 감쌈, 매턴 fresh recompute) — causal 인터페이스 + 원본 점수 *근사*.
- `online_streaming.py` (자체 구현, incremental block-cosine, running threshold, one-sided depth) — **NLTK 원본 점수 재현하지 않음**. 핵심 비교값 = per-turn latency. codex:rescue 위임 결과 (decision-log 2026-05-20).

## 실행

```
python methods/texttiling/offline.py            # full test (논문 방향 검증)
python methods/texttiling/online.py --target-turns 100
python methods/texttiling/online_streaming.py --target-turns 0     # 3 dataset 전체 (Def-DTS bundle 데이터)
python methods/texttiling/online_streaming.py --datasets tiage --target-turns 100  # 빠른 smoke
python methods/bayesseg/offline.py  --limit 100
python methods/bayesseg/online.py   --target-turns 100
python methods/greedyseg/online_delay2.py --target-turns 0          # 3 dataset 전체, device=auto
python methods/greedyseg/online_delay2.py --datasets tiage --target-turns 100  # smoke
```
모두 `outputs/experiments/<name>/REPORT.md` 산출 (CLAUDE.md 규칙).
non-LLM (quota·비용 0).

## `TextTiling-online-streaming` 실행 가이드 (2026-05-20 신규)

### 1) 의존성 (1회)

```bash
# segeval (Pk/WD metric)
uv add segeval     # 이미 pyproject.toml/uv.lock 에 반영됨

# Def-DTS 번들 데이터 (3 dataset 의 test jsonl). 알고리즘 의존 X, 데이터만.
git clone --depth=1 https://github.com/ElPlaguister/Def-DTS.git benchmarks/Def-DTS
```

`benchmarks/superdialseg`, `bayes-seg`, `OnlineSegServer.java`, `ant build`
**모두 불필요**. Java/JVM 도 불필요.

### 2) 기본 실행 — 3 dataset 전체

```bash
uv run python methods/texttiling/online_streaming.py --target-turns 0
```

산출: `outputs/experiments/2026-05-20_texttiling_streaming/REPORT.md` +
`turns_{tiage,dialseg711,superseg}.jsonl` (sidecar, gitignored). 약 **30 초**.
GPU·LLM 무관, pure-CPU Python.

### 3) 자주 쓰는 옵션

```bash
# 빠른 smoke (1 dataset, 누적 발화 ≥ 100)
uv run python methods/texttiling/online_streaming.py \
    --datasets tiage --target-turns 100

# HP 조정 (default: w=5/k=3/c=0.5/min_gap=3/warmup_gaps=3)
uv run python methods/texttiling/online_streaming.py \
    -w 10 -k 6 --c 1.0 --min-gap 4

# 실험 이름 분리 (HP sweep 시 결과 덮어쓰기 방지)
uv run python methods/texttiling/online_streaming.py \
    --name 2026-05-20_tt_streaming_w10k6 -w 10 -k 6
```

### 4) 결과 해석 요점

- **핵심 비교값 = per-turn latency** (mean/p50/p95/max ms). Pk/WD/F1 = INDICATIVE
  (NLTK 원본 점수 재현 *안 함*). 자세한 사유는 § "정직성 / 한계" 와 REPORT 본문.
- 결과 표 schema: `[dataset, n(dial/turn), Pk, WD, F1, Score, lat/turn(ms),
  pred_bs, gold_bs]`. `Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD)`.
- 알고리즘 결정적 → seed 무관, 같은 HP·같은 데이터면 byte-identical 결과.

### 5) 코드/테스트

```bash
# 모듈 import (Hi-EM 본체 연동 / 다른 runner 와 공유)
from hi_em.baselines import StreamingTextTiling
seg = StreamingTextTiling(w=10, k=6, c=0.5, min_gap=4, warmup_gaps=3)
for utt in dialogue_utts:
    new_boundary_indices = seg.push(utt)   # list[int] (1-based)
final_boundaries = seg.flush()              # 잔여 gap 처리

# 단위 테스트 (11 ea, < 1s)
uv run pytest tests/test_texttiling_streaming.py -v
```

## `GreedySeg-online-delay2` 실행 가이드 (2026-05-21 신규)

### 1) 의존성

```bash
# Def-DTS 번들 데이터 (이미 clone 됐다면 skip)
git clone --depth=1 https://github.com/ElPlaguister/Def-DTS.git benchmarks/Def-DTS
# transformers/torch 는 이미 의존성. BERT (bert-base-uncased ~440MB) 자동 다운.
```

### 2) 기본 실행 — 3 dataset 전체

```bash
uv run python methods/greedyseg/online_delay2.py --target-turns 0
```

device 자동 선택 (cuda → mps → cpu). 산출:
`outputs/experiments/2026-05-21_greedyseg_online_delay2/REPORT.md` +
`turns_{ds}.jsonl` (sidecar, gitignored).

### 3) 자주 쓰는 옵션

```bash
# 빠른 smoke
uv run python methods/greedyseg/online_delay2.py --datasets tiage --target-turns 100

# device 강제
uv run python methods/greedyseg/online_delay2.py --device cpu  # 결정성 가장 안정
uv run python methods/greedyseg/online_delay2.py --device cuda  # GPU 권장

# HP 조정 (원본 default 유지가 정직)
uv run python methods/greedyseg/online_delay2.py \
    --window-size 2 --jump-step 2 --max-seg-round 8 --sim-threshold 0.6
```

### 4) 정직성 핵심

- **이름**: `GreedySeg-online-delay2` (강한 online 명명 OK — codex 2026-05-21 검증
  통과). 원본 score 공식·HP·argmin greedy 그대로 보존, *입력 인터페이스 streaming
  + boundary emit 만 right-context (window=2) 만큼 지연*.
- **5행 핵심표 가능** (본 plan 의 baseline 중 유일). offline 결과와 별도 열/블록 분리.
- TextTiling-streaming (encoder-free, ~0.01ms) 과 같은 latency 표에 *직접 비교 금지*
  — encoder cost (BERT forward) 차원이 다름.
- 결정성: CPU > CUDA > MPS. 논문 reproducibility 시 device·seed·버전·fallback
  REPORT 명기 + 동일 device 반복 측정.

### 5) 코드 사용

```python
from hi_em.baselines import GreedySegOnlineDelay2

seg = GreedySegOnlineDelay2(
    backbone="bert-base-uncased",
    window_size=2, jump_step=2, max_seg_round=8, sim_threshold=0.6,
    max_seq_length=50, device="auto",
)
for utt in dialogue_utts:
    new_boundary_indices = seg.push(utt)   # list[int] (1-based)
final = seg.flush()
print(seg.state())  # {t, cut_index, n_candidates, device, bert_forwards, last_boundary}
```

```bash
uv run pytest tests/test_greedyseg_delay2.py -v   # 6 ea (tiny model, < 10s)
```

## 정직성 / 한계

- **online(prefix-causal) 은 보조(AUXILIARY)** — codex decision-log
  2026-05-20. 미래 미관측·causal 이라 Pk/F1 가 offline 보다 낮고
  **indicative**(BayesSeg-online=native-K 과분할, TextTiling-online=
  짧은 prefix 과소분할). 핵심 5행 비교표(CSM/Def-DTS/Plain/Ours)에
  넣지 않음. 의미 있는 비교값은 **per-turn latency**.
- offline 의 Pk/WD/F1 은 *원 SuperDialseg 논문 보고치와 데이터·공식
  metric 이 달라 정확히 일치하지 않음* (방향·정상동작·offline↔online
  격차 검증용). 논문값: TextTiling tiage .363/superseg .471/dialseg711
  .382 ; BayesSeg .419/.463/.614.
- 원본 알고리즘 코드 무수정(benchmarks read-only 유지). online 의
  OnlineSegServer 는 I/O harness 추가일 뿐 알고리즘 불변.

decision-log: `context/06-decision-log.md` 2026-05-19/20 항목 참조.
