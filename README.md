# Hi-EM — Human-inspired Episodic Memory for LLM Conversations

> Transformer 기반 LLM에 **fine-tuning 없이** 붙는 **실시간 대화 메모리 관리 시스템**.
> 인지과학 Structured Event Memory (Franklin et al. 2020)을 쿼리-토픽 구조로
> 재해석해 **LTM(SSD) 영속 저장**과 **Memory window(STM) 승격**으로
> 긴 대화에서 현재 라운드에 필요한 턴만 prefill prefix로 로드한다.

자세한 목표·제약은 `brief.md`.

---

## 현재 상태 (2026-04-25)

**Phase 2 진입 (옵션 3 reframing 채택), Step 2-1 LTM 저장 포맷 확정. Phase 1-6 Gate FAIL은 boundary F1·ARI 모두 Hi-EM의 진짜 가치 지표가 아니라는 reframing으로 인정 → Phase 4 downstream QA로 가치 검증.**

- **Phase 0 완료**: 자료 분석, 사건 모델 옵션 A 확정 ($P(\mathbf{s}_n \mid e_n=k) = \mathcal{N}(\mu_k, \mathrm{diag}(\sigma_k^2))$)
- **Phase 1-1, 1-2 완료**: `src/hi_em/` 구현 + 18 unit tests passing
- **Phase 1-3, 1-4 (TopiOCQA)**: Hi-EM F1=0.471 vs cosine 0.467 → **Gate PASS (marginal)**
- **Phase 1-5 (TIAGE)**: Hi-EM 두 HP 모두 cosine 0.421에 패배 → **Gate FAIL**
- **Phase 1-6 종합 Gate**: TopiOCQA PASS + TIAGE FAIL → **Phase 2 진입 보류**
- **2026-04-25 TIAGE 108-config sweep 종료**: best F1=0.383 (α=10, λ=3, σ₀²=0.1) < cosine 0.421, 두 Gate 조건 모두 FAIL → **어떤 HP로도 baseline 못 넘음** 결정적 증거 확보.
- **2026-04-25 옵션 5 (clustering quality) 완료**: V-measure/NMI/ARI 측정 — 모든 metric에서 cosine 우위. **Boundary F1 ↔ ARI trade-off** 발견 — persistence HP (α=1) ARI=0.398·0.397 vs freq-shift HP (α=10) ARI=0.187·0.314. **메모리 시스템 관점에선 persistence HP 적합** (cluster 보존성↑) → **Phase 2 LTM/Memory window HP 채택 근거**.
- **2026-04-25 Phase 2 진입 + Step 2-1 LTM 저장 포맷 확정**: per-conversation **JSONL** (turn append-only) + **`<conv_id>.state.json`** (topic 상태 latest snapshot, overwrite). 디렉토리 `data/ltm/` (gitignored). Topic 분할 HP **persistence (α=1, λ=10, σ₀²=0.01)** 채택. 자세한 trade-off: `context/01-hi-em-design.md §9.1`.
- **2026-04-25 Step 2-2 완료**: `src/hi_em/ltm.py` (LTM API 5 methods) + `tests/test_ltm.py` (8 tests). 전체 테스트 회귀 **26/26 PASS**.
- **2026-04-25 Step 2-3 완료**: `src/hi_em/memory_window.py` (`select_memory_window`: cosine top-k topics × recency top-k turns/topic) + `tests/test_memory_window.py` (8 tests). 전체 회귀 **34/34 PASS**. Step 2-4 (importance/merge/adaptive K)는 Phase 4 결과로 튜닝 — 현 baseline 정책으로 Phase 3 진입.
- 종합 회고 + 다음 행동 후보 5종: `report.md`
- 결정 이력 (append-only): `context/06-decision-log.md`
- HP regime split 발견: persistence(α=1, λ=10, σ₀²=0.01) vs frequent-shift(α=10, λ=1, σ₀²=0.1)

---

## 레포 구조 (현 시점)

```
Hi-EM/
├── brief.md                  프로젝트 한 줄 요약, 목표, 제약, 벤치마크 우선순위
├── plan.md                   구현 로드맵 (Phase 0 → 5, 체크박스)
├── handoff.md                세션 진입점 — 지금 무엇을 해야 하는지
├── CLAUDE.md                 Claude Code 작업 규칙 (환경 분리, 커밋, Notebook 정책, Step 완료 프로토콜)
├── README.md                 이 파일
├── report.md                 Phase 0~1-5 종합 회고 + 다음 결정 분기
├── requirements.txt          로컬/git 환경 Python 의존성 (Colab과 분리 관리)
├── .gitignore
│
├── context/                  확립된 설계 문서 (세션 시작 시 읽기)
│   ├── 00-sem-paper.md           SEM 논문 정리 (Hi-EM 관점: 계승/대체/폐기 매핑, SEM2 코드 분석)
│   ├── sem-equations.md          SEM 식 1~24 원본 LaTeX reference + 페이지 + 변수 의미
│   ├── 01-hi-em-design.md        Hi-EM 설계 확정본 + Phase 1/2/4로 위임된 결정
│   ├── 02-math-model.md          수식 확정본 + HP regime split 표
│   ├── 03-architecture.md        모듈 구조, 파일 레이아웃
│   ├── 04-benchmarks.md          벤치마크 메타 (평가 축: 토픽 경계 vs downstream QA)
│   ├── 05-open-questions.md      열려있는 질문들
│   └── 06-decision-log.md        설계 결정 이력 (append-only)
│
├── src/hi_em/                코어 구현 (Phase 1 + Phase 2-2/2-3 완료)
│   ├── embedding.py              bge-base-en-v1.5 wrapper (L2 norm, 768dim)
│   ├── topic.py                  centroid + diag σ² + Welford 온라인 업데이트
│   ├── scrp.py                   sticky_crp_unnormed (SEM 식 1)
│   ├── sem_core.py               HiEMSegmenter.assign() — online MAP 루프
│   ├── ltm.py                    LTM read/write API (per-conv JSONL + state.json, §9.1)
│   └── memory_window.py          select_memory_window — cosine top-k topics × recency top-k turns
│
├── tests/                    pytest (Phase 1-2 + 2-2/2-3 완료, 34 tests passing)
│   ├── test_scrp.py              7 tests
│   ├── test_topic.py             6 tests
│   ├── test_sem_core.py          5 tests
│   ├── test_ltm.py               8 tests
│   └── test_memory_window.py     8 tests
│
├── scripts/                  실행/분석 스크립트
│   ├── check_step_done.py            Step 완료 gate
│   ├── run_topiocqa_segmentation.py  Phase 1-3 main 측정
│   ├── run_topiocqa_sweep.py         HP grid sweep (108 configs)
│   ├── run_topiocqa_variants.py      구조 변형 5종
│   ├── run_topiocqa_multisignal.py   Multi-signal 564 configs
│   ├── run_topiocqa_anchors.py       Anchor 4종
│   ├── run_topiocqa_bigencoder.py    bge-large 비교
│   ├── run_topiocqa_contextualized.py Context window K∈{0..all}
│   ├── run_tiage_segmentation.py     Phase 1-5 main 측정
│   ├── run_tiage_sweep.py            Phase 1-6 TIAGE 108-config grid (TopiOCQA mirror)
│   └── run_clustering_quality.py     Phase 1-6 옵션 5: V-measure/NMI/ARI
│
├── notebooks/                얇은 wrapper — Colab 인터랙티브 실행 편의용 (선택적)
│   └── phase-1-tiage.ipynb       TIAGE 평가 + Phase 1-6 종합 Gate
│   └── setup_colab.ipynb         Colab 환경 셋업 (gitignored)
│
│   * Portability 원칙: notebooks/ 통째로 삭제해도 동작.
│     모든 실험 로직은 scripts/*.py에 있음 → 로컬 환경에선
│     `python scripts/X.py` 직접 실행으로 진행 가능.
│
├── templates/                반복 사용 템플릿
│   ├── module-template.py
│   └── experiment-log.md
│
├── outputs/                  실험 결과
│   ├── benchmark-analysis.md             Phase 0-2 4 벤치마크 분석
│   ├── phase-1-topiocqa.md               Phase 1-3/1-4 결과 + 7-iter 탐색 이력
│   ├── phase-1-topiocqa-sweep.json       (탐색 결과 raw)
│   ├── phase-1-topiocqa-variants.json
│   ├── phase-1-topiocqa-multisignal.json
│   ├── phase-1-topiocqa-anchors.json
│   ├── phase-1-topiocqa-encoder.json
│   ├── phase-1-topiocqa-contextualized.json
│   ├── phase-1-tiage.md                  Phase 1-5 결과
│   ├── phase-1-tiage-sweep.json          Phase 1-6 TIAGE sweep raw (108 configs)
│   ├── phase-1-clustering-quality.md     Phase 1-6 옵션 5 분석
│   └── phase-1-clustering-quality.json   raw
│
└── SEM-paper.pdf             Franklin et al. 2020 원본 논문 (Psychological Review)
```

### gitignored (각자 준비)

```
notebooks/setup_colab.ipynb   Colab 환경 셋업 노트북 (notebooks/ 안의 유일한 gitignored 파일)
sem.txt                       SEM PDF → pdftotext 덤프 (작업 보조)
benchmarks/locomo/            아래 "외부 레포" 참조
benchmarks/topiocqa/
benchmarks/LongMemEval/
benchmarks/tiage/
SEM/                          SEM2 레포 (현재 tracked, 참조 전용)
outputs/*.npy|*.pkl|*.log     대용량 실험 산출물
outputs/tmp/
.claude/                      Claude Code 로컬 settings
.venv/
```

---

## 외부 레포 (clone 필요)

**SEM2 — 코드 알고리즘 참조 (현재는 tracked되어 있음)**
```bash
# 이미 레포에 포함됨. 재설치가 필요하면:
git clone --depth=1 https://github.com/nicktfranklin/SEM2 SEM && rm -rf SEM/.git
```

**벤치마크 — Phase 1+ 평가용 4종**
```bash
mkdir -p benchmarks && cd benchmarks
git clone --depth=1 https://github.com/snap-research/locomo       # Phase 4 QA
git clone --depth=1 https://github.com/McGill-NLP/topiocqa        # Phase 1 topic 경계 (factoid)
git clone --depth=1 https://github.com/xiaowu0162/LongMemEval     # Phase 4 QA (5 abilities)
git clone --depth=1 https://github.com/HuiyuanXie/tiage           # Phase 1 topic 경계 (chit-chat)
cd ..
```

**평가 축 분리** (`context/04-benchmarks.md`):
- **Topic 경계 감지** (Phase 1) — TopiOCQA, TIAGE: turn-transition F1
- **Downstream QA** (Phase 4) — LoCoMo, LongMemEval: QA accuracy. **session 경계를 topic 경계로 쓰면 안 됨.**

**데이터 다운로드** — Phase 0 Step 0-2 실행 이력 참고:
- LoCoMo: `benchmarks/locomo/data/locomo10.json` (레포 내 포함)
- TopiOCQA dev: `cd benchmarks/topiocqa && python download_data.py --resource data.topiocqa_dataset.dev --output_dir .`
- LongMemEval oracle: `wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json -P benchmarks/LongMemEval/data/`

---

## 빠른 시작

```bash
# 0. 로컬 환경 의존성 설치 (Colab 사용자는 setup_colab.ipynb 실행, 아래 무시)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Step 진행 중 언제든 상태 검증
python scripts/check_step_done.py             # 현재 Step 자동 감지
python scripts/check_step_done.py --step 0-3  # 특정 Step

# 2. Step 완료 처리 (CLAUDE.md "Step 완료 프로토콜" 참조)
#    - 3-angle self-audit (구조/동작/설계)
#    - check_step_done.py exit 0 확인
#    - plan.md 체크박스 [x]
#    - handoff.md 현재 상태/다음 할 일 갱신
#    - git add/commit 명령어를 사용자에게 제시 (사용자가 실행)
```

---

## 제약 (CLAUDE.md 전문 참조)

- **No fine-tuning** (외부 LLM 학습 금지)
- **PyTorch only** (TensorFlow/Keras 금지)
- **SEM2 코드 복사 금지** (참조만 허용)
- **설계 변경은 반드시 `06-decision-log.md` append 후**
- **환경 분리**: `setup_colab.ipynb`만 Colab 의존 허용, 그 외 파일은 로컬/git 기준 동작
