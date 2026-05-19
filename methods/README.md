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
| `texttiling/online.py`  | prefix-causal(U1..Ut), AUXILIARY → `scripts/run_texttiling_prefix.py` |
| `bayesseg/offline.py`   | 원본 SuperDialseg BayesSegmenter, 전체 대화, `segment dp.config`(`-num-segs 7`) |
| `bayesseg/online.py`    | persistent JVM·native-K·prefix, AUXILIARY → `scripts/run_bayesseg_prefix.py` |

## 실행

```
python methods/texttiling/offline.py            # full test (논문 방향 검증)
python methods/texttiling/online.py --target-turns 100
python methods/bayesseg/offline.py  --limit 100
python methods/bayesseg/online.py   --target-turns 100
```
모두 `outputs/experiments/<name>/REPORT.md` 산출 (CLAUDE.md 규칙).
non-LLM (quota·비용 0).

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
