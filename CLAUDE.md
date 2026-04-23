# CLAUDE.md — Hi-EM 프로젝트 규칙

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

## 환경 분리

- `setup_colab.ipynb`는 항상 `.gitignore` 유지. git에 커밋하지 않는다.
- `setup_colab.ipynb`는 로컬/Colab 공용 setup notebook으로 유지한다.
- 그 외 모든 파일은 Colab 전용 의존 없이 로컬 기준으로 동작해야 한다.
  - `from google.colab import drive`, `drive.mount`, `/content/` 경로를 기본 파일에 넣지 않는다.
  - Colab 전용 코드가 꼭 필요하면 `IS_COLAB` 분기 안에만 작성한다.
- 경로는 하드코딩 금지. `git rev-parse --show-toplevel` 또는 상대경로 사용.

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
