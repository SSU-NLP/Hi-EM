/home/namchailin/Hi-EM/
│
├── brief.md                  # 프로젝트 한 줄 요약, 목표, 제약, 벤치마크 우선순위
├── plan.md                   # 구현 로드맵 (Phase 0 → 5, 체크박스)
├── handoff.md                # Claude Code 진입점 — 지금 무엇을 해야 하는지
├── CLAUDE.md                 # 코딩 스타일, 커밋 규칙, 문서화/테스트 규칙
├── requirements.txt          # Python 의존성 고정
├── .gitignore
│
├── context/                  # 확립된 설계 문서 (읽기 필수)
│   ├── 00-sem-paper.md       #   SEM 논문(Franklin 2020) + SEM2 정리, Hi-EM이 계승/폐기한 것
│   ├── 01-hi-em-design.md    #   Hi-EM 설계 결정사항 (확정/미확정 분리)
│   ├── 02-math-model.md      #   수학 모형 (확정 수식 + 미확정 섹션)
│   ├── 03-architecture.md    #   모듈 구조, 파일 레이아웃, 의존 그래프
│   ├── 04-benchmarks.md      #   벤치마크 정보 + 데이터 분석 체크리스트
│   ├── 05-open-questions.md  #   아직 확정 안 된 질문들
│   └── 06-decision-log.md    #   설계 결정 이력 (날짜 + 근거 append-only)
│
├── colab/                    # Colab A100 환경 세팅
│   ├── setup_colab.ipynb     #   세션 시작 시 실행하는 노트북
│   ├── setup_colab.sh        #   한 줄 세팅 셸 스크립트
│   └── README.md             #   Colab 사용 가이드 (Drive 동기화, VSCode 연결)
│
├── templates/                # 반복 사용 템플릿
│   ├── module-template.py    #   새 모듈 작성 시 시작점
│   └── experiment-log.md     #   실험 시작 시 복사해서 사용
│
├── src/                      # 구현 코드 (Claude Code가 채움)
│   └── hi_em/
│       └── .gitkeep          #   사건 모델 확정 후 세부 구조 결정
│
├── scripts/                  # 실행/전처리/분석 스크립트
│   ├── download_benchmarks.sh #  벤치마크 데이터 일괄 다운로드
│   ├── verify_env.py         #   환경 검증 (Python/GPU/패키지/데이터)
│   └── .gitkeep              #   분석/실행 스크립트는 Phase 진행하며 추가
│
├── tests/                    # pytest 테스트
│   └── .gitkeep
│
├── outputs/                  # 실험 결과, 로그, 생성된 파일
│   └── .gitkeep              #   benchmark-analysis.md, phase-N-results.md 등
│
├── benchmarks/               # [읽기 전용] 외부 벤치마크 레포
│   ├── locomo/               #   데이터: data/locomo10.json (레포 내 포함)
│   ├── topiocqa/             #   데이터: python download_data.py 필요
│   └── LongMemEval/          #   데이터: HuggingFace에서 받음
│
├── SEM/                      # [참조 전용] SEM2 (nicktfranklin/SEM2) — 코드 복사 금지
│
└── SEM-paper.pdf             # Franklin et al. 2020 원본 논문