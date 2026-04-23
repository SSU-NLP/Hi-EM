# CLAUDE.md — Hi-EM 프로젝트 규칙

## Step 완료 프로토콜 (최상위 강제 규칙)

Step을 `plan.md`에서 `[x]`로 표시하기 **직전에 반드시** 실행:

```bash
python scripts/check_step_done.py
```

- **exit code 0이 나올 때까지 Step 완료 처리 금지.**
- FAIL이 남아있으면 원인을 수정하고 스크립트를 재실행한다. **통과할 때까지 반복한다.**
- WARN은 허용 가능하지만, 가능하면 해결한다.
- 검증 없이 `[x]` 처리하거나 커밋하지 않는다.
- Step에 대한 검증 로직이 `STEP_CHECKS`에 없으면 추가한 뒤 진행한다.

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
