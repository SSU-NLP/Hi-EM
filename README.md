# Hi-EM — Human-inspired Episodic Memory for LLM Conversations

> Transformer 기반 LLM에 **fine-tuning 없이** 붙는 **실시간 대화 메모리 관리 시스템**.
> 인지과학 Structured Event Memory (Franklin et al. 2020)을 쿼리-토픽 구조로
> 재해석해 **토픽 단위 STM/LTM 관리**와 **KV cache paging**을 구현한다.

자세한 목표·제약은 `brief.md`.

---

## 현재 상태

**Phase 0 완료** (2026-04-23). 다음은 Phase 1 (코어 모듈 구현) — 사용자 승인 후 진행.

- 사건 모델 확정: **옵션 A (Centroid + diag variance)** — $P(\mathbf{s}_n \mid e_n=k) = \mathcal{N}(\mu_k, \mathrm{diag}(\sigma_k^2))$
- Prior: sticky-CRP, $\alpha=1.0$, $\lambda=10.0$
- Scene embedding: `bge-base-en-v1.5` (L2 normalize, 768dim)
- 세부 결정 이력: `context/06-decision-log.md`
- Phase 진행: `plan.md`, 다음 할 일: `handoff.md`

---

## 레포 구조

```
Hi-EM/
├── brief.md                  프로젝트 한 줄 요약, 목표, 제약, 벤치마크 우선순위
├── plan.md                   구현 로드맵 (Phase 0 → 5, 체크박스)
├── handoff.md                세션 진입점 — 지금 무엇을 해야 하는지
├── CLAUDE.md                 Claude Code 작업 규칙 (환경 분리, 커밋, Step 완료 프로토콜)
├── README.md                 이 파일
├── .gitignore
│
├── context/                  확립된 설계 문서 (세션 시작 시 읽기)
│   ├── 00-sem-paper.md           SEM 논문(Franklin 2020) + SEM2 정리, 식 1~24 검증본
│   ├── 01-hi-em-design.md        Hi-EM 설계 확정본 + Phase 1/2/4로 위임된 결정
│   ├── 02-math-model.md          수식 확정본 + Phase 4 튜닝 대상 하이퍼파라미터
│   ├── 03-architecture.md        모듈 구조, 파일 레이아웃
│   ├── 04-benchmarks.md          벤치마크 메타 정보
│   ├── 05-open-questions.md      열려있는 질문들
│   └── 06-decision-log.md        설계 결정 이력 (append-only)
│
├── templates/                반복 사용 템플릿
│   ├── module-template.py        새 모듈 작성 시 시작점
│   └── experiment-log.md         실험 시작 시 복사해서 사용
│
├── scripts/                  실행/분석 스크립트
│   └── check_step_done.py        Step 완료 gate (plan.md 체크박스 [x] 전 필수)
│
├── outputs/                  실험 결과, 분석 문서
│   └── benchmark-analysis.md     Phase 0 Step 0-2 벤치마크 분석 (LoCoMo / TopiOCQA / LongMemEval)
│
├── SEM-paper.pdf             Franklin et al. 2020 원본 논문 (Psychological Review)
│
└── (Phase 1 이후 추가)
    ├── src/hi_em/                코어 구현
    └── tests/                    pytest 테스트
```

### gitignored (각자 준비)

```
setup_colab.ipynb             Colab/로컬 공용 환경 세팅 노트북 (레포에 없음)
sem.txt                       SEM PDF → pdftotext 덤프 (작업 보조)
benchmarks/locomo/            LoCoMo 레포 (아래 "외부 레포" 참조)
benchmarks/topiocqa/
benchmarks/LongMemEval/
SEM/                          SEM2 레포 (주: 현재 tracked이나 참조 전용)
outputs/*.npy|*.pkl|*.log     대용량 실험 산출물
.venv/
```

---

## 외부 레포 (clone 필요)

**SEM2 — 코드 알고리즘 참조 (현재는 tracked되어 있음)**
```bash
# 이미 레포에 포함됨. 재설치가 필요하면:
git clone --depth=1 https://github.com/nicktfranklin/SEM2 SEM && rm -rf SEM/.git
```

**벤치마크 — Phase 0 이후 실험용**
```bash
mkdir -p benchmarks && cd benchmarks
git clone --depth=1 https://github.com/snap-research/locomo
git clone --depth=1 https://github.com/McGill-NLP/topiocqa
git clone --depth=1 https://github.com/xiaowu0162/LongMemEval
cd ..
```

**데이터 다운로드** — Phase 0 Step 0-2 실행 이력 참고:
- LoCoMo: `benchmarks/locomo/data/locomo10.json` (레포 내 포함)
- TopiOCQA dev: `cd benchmarks/topiocqa && python download_data.py --resource data.topiocqa_dataset.dev --output_dir .`
- LongMemEval oracle: `wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json -P benchmarks/LongMemEval/data/`

---

## 빠른 시작 (Claude Code 세션)

```bash
# 1. Step 진행 중 언제든 상태 검증
python scripts/check_step_done.py             # 현재 Step 자동 감지
python scripts/check_step_done.py --step 0-3  # 특정 Step

# 2. Step 완료 처리 (CLAUDE.md "Step 완료 프로토콜" 참조)
#    - 3-angle self-audit (구조/동작/설계)
#    - check_step_done.py exit 0 확인
#    - plan.md 체크박스 [x]
#    - handoff.md 현재 상태/다음 할 일 갱신
#    - git commit
```

---

## 제약 (CLAUDE.md 전문 참조)

- **No fine-tuning** (외부 LLM 학습 금지)
- **PyTorch only** (TensorFlow/Keras 금지)
- **SEM2 코드 복사 금지** (참조만 허용)
- **설계 변경은 반드시 `06-decision-log.md` append 후**
- **환경 분리**: `setup_colab.ipynb`만 Colab 의존 허용, 그 외 파일은 로컬/git 기준 동작
