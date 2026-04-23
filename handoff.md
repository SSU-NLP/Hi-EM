# Claude Code 핸드오프

> **이 파일은 매 세션 시작 시 첫 번째로 읽는 파일이다.**
> **매 세션 종료 시 반드시 업데이트된다.**

---

## 세션 시작 프로토콜 (매번 순서대로)

1. 이 파일(`handoff.md`) 전체 읽기
2. `plan.md`에서 현재 Phase와 미완료 체크박스 확인
3. `CLAUDE.md`의 작업 규칙 확인
4. 현재 Phase에 해당하는 `context/*` 문서 읽기
5. 마지막 commit 로그 확인: `git log -5 --oneline`
6. "다음 할 일" 섹션(아래)의 첫 항목부터 시작

---

## 세션 종료 프로토콜 (매번 순서대로)

작업을 멈추기 전에 **반드시** 아래를 수행:

1. 이 파일의 "현재 상태"와 "다음 할 일" 섹션 업데이트 (날짜 `마지막 업데이트` 오늘로 갱신)
2. 설계 결정이 있었다면 `context/06-decision-log.md`에 append
3. 설계 변경이 있었다면 `context/01-hi-em-design.md` 또는 `02-math-model.md` 갱신
4. **`python scripts/check_step_done.py` 실행 → exit code 0 받을 때까지 수정-재실행 반복**
5. `plan.md` 체크박스 갱신 (검증 통과한 항목만 `[x]`)
6. `git add -A && git commit` (커밋 메시지는 CLAUDE.md 규칙 따름)
7. 다음 세션이 바로 이어갈 수 있게 이 파일의 "다음 할 일" 첫 항목을 구체적으로 작성

---

## 현재 상태

**마지막 업데이트**: 2026-04-23
**현재 Phase**: Phase 0 — 자료 분석 및 설계 확정
**진행률**: 1/3 Step 완료

### 완료된 것
- 프로젝트 디렉토리 구조 생성
- 벤치마크 레포 3개 clone (`benchmarks/locomo/`, `benchmarks/topiocqa/`, `benchmarks/LongMemEval/`)
- SEM2 레포 clone (`SEM/` ← `nicktfranklin/SEM2`, 과거 v1에서 교체)
- 설계 문서 초안 작성 (`context/`)
- **Step 0-1 완료**: SEM 논문 전 35페이지 정독(pdftotext 산문 + 수식 10페이지 PNG 직독). `SEM/sem/sem.py` 실제 코드 검증(`_calculate_unnormed_sCRP`, `run()` verbatim pseudocode). `context/00-sem-paper.md` 재작성 — 식 1~24 정확한 정의, Hi-EM 계승/대체/폐기 매핑, 검증 미해결 지점 3건 명시 (11940자).

### 미완료
- 벤치마크 데이터 다운로드 및 분석 (Step 0-2)
- 사건 모델 형태 결정 (Step 0-3)
- Phase 1 구현

---

## 다음 할 일 (세션 시작 시 여기서부터)

### 즉시 시작: Phase 0 Step 2 — 벤치마크 데이터 분석

1. **LoCoMo** (`benchmarks/locomo/data/locomo10.json`) — 레포 내 포함됨
   - JSON 구조 직접 확인 (샘플 1~2개)
   - 세션 수, 턴 수, 쿼리 길이 통계
   - Topic 전환 annotation 있는지 확인
2. **TopiOCQA** — `benchmarks/topiocqa/download_data.py` 실행해 train/dev 다운
   - Wiki document 기반 topic 경계 확인
   - 평균 턴 수, topic shift 빈도
3. **LongMemEval** — HuggingFace에서 `longmemeval_oracle.json` 등 다운
   - 5개 능력(IE/Multi-session/Temporal/Knowledge update/Abstention) 분포
   - Needle-in-haystack 구조 확인
4. 세 벤치마크를 비교: Claude-유사 장기 대화와의 유사성, 어떤 사건 모델 옵션이 유리할지 판단 근거
5. `outputs/benchmark-analysis.md` 작성 — 각 벤치마크 섹션 + 종합 판단 (1500자 이상)
6. `python scripts/check_step_done.py` 통과 확인

---

## 주의: 이전 대화의 편향된 결론을 맹신하지 마라

이전에 "Centroid + Entity set + Multi-signal" 같은 설계가 논의됐지만,
이는 TopiOCQA/TIAGE만 본 편향된 판단이었다.
LoCoMo/LongMemEval까지 직접 보고 다시 판단해라.

---

## Phase별 진입점

현재 Phase에 따라 다른 파일을 집중적으로 봐라.

### Phase 0 (현재): 자료 분석 및 설계 확정
- 주로 읽기: `SEM-paper.pdf`, `benchmarks/*/`, `context/00-sem-paper.md`, `context/04-benchmarks.md`
- 주로 쓰기: `outputs/benchmark-analysis.md`, `context/01-hi-em-design.md`, `context/02-math-model.md`, `context/06-decision-log.md`
- 완료 기준: 사건 모델 형태가 `context/01-hi-em-design.md`에 확정됨

### Phase 1: 코어 모듈 구현
- 주로 읽기: `context/01-hi-em-design.md` (확정된 설계), `context/02-math-model.md` (수식), `context/03-architecture.md`, `SEM/sem/sem.py`
- 주로 쓰기: `src/hi_em/`
- 완료 기준: sCRP + 사건 모델 + online MAP 루프가 `src/hi_em/`에 구현되고 `tests/`에 단위 테스트 통과

### Phase 2 이후: `plan.md` 참조

---

## 벤치마크 데이터 준비 (Phase 0 Step 2에서 필요)

### LoCoMo
```bash
cat benchmarks/locomo/data/locomo10.json | jq '.[0]' | head -200
```

### TopiOCQA
```bash
cd benchmarks/topiocqa
python download_data.py --resource data.retriever.all_history.dev
cd ../..
```

### LongMemEval
```bash
cat benchmarks/LongMemEval/README.md
# HuggingFace에서 받음 - README 안내 따라감
```

---

## 환경 세팅 (아직 안 했으면)

```bash
cd /home/namchailin/Hi-EM

python --version  # 3.10+ 확인
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 환경 검증
python scripts/verify_env.py
```

Colab A100 환경은 `colab/README.md` 참조.

---

## 외부 레포 사용 규칙 (강제)

- `benchmarks/*` — **읽기 전용.** 수정 금지.
- `SEM/` (= SEM2, `nicktfranklin/SEM2` current build) — **참조 전용.** 알고리즘 구조만 참고, 코드 복사 금지. 원본 아카이브는 `ProjectSEM/SEM`.
- `SEM-paper.pdf` — 수정 불가.

---

## 전역 금지 사항 (CLAUDE.md와 중복이지만 재확인)

- TensorFlow/Keras 사용 금지 (PyTorch만)
- 외부 LLM fine-tuning 금지
- SEM2 코드 직접 복사 금지 (참조만 허용)
- 벤치마크 근거 없이 사건 모델 확정 금지
- `handoff.md` 업데이트 없이 세션 종료 금지
- `context/06-decision-log.md`에 기록 없이 설계 결정 금지

---

## 참고 문서 전체 목록

반드시 읽을 것:
- `brief.md` — 프로젝트 한 줄 요약
- `plan.md` — 전체 로드맵 + 체크박스
- `CLAUDE.md` — 코딩 규칙
- `context/00-sem-paper.md` — SEM 계승/폐기 정리
- `context/01-hi-em-design.md` — 확정/미확정 설계
- `context/02-math-model.md` — 수식
- `context/03-architecture.md` — 파일 구조
- `context/04-benchmarks.md` — 벤치마크 정보
- `context/05-open-questions.md` — 열려있는 질문
- `context/06-decision-log.md` — 결정 이력

필요 시:
- `SEM-paper.pdf` — Franklin et al. 2020 원본 논문
- `SEM/` — SEM2 코드 (참조, `nicktfranklin/SEM2`)
- `colab/README.md` — Colab A100 환경