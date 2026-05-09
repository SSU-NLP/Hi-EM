# Hi-EM Methodology — version index

각 버전의 *알고리즘 수준* 차이를 한 파일씩 정리. 사소한 변경(예: prediction caching, hyperparameter default, prior decay 등)도 해당 버전 파일의 "고려 중인 변형" 또는 "변경 이력" 섹션에 누적한다.

## 버전 계보

```
v1   (Gaussian likelihood, flat history)
 │
 └→ v2   (= v1 + STM round-clear + sessioned preload)
       │
       └→ v3.1.1   (likelihood: Gaussian → bounded cosine, centroid-only topic)
              │
              └→ v3.2.1   (sticky-CRP count: raw → sub-linear (C_k+1)^β)
                     │
                     └→ v3.3.1   (likelihood term: cos(s, μ_k) → cos(s, ŝ_k), per-topic RNN)
                            │
                            └→ v3.3.2   (+ SEM2 surprise → hard boundary on PE spike)
```

## 한 줄 요약

| 버전 | segmenter class | 핵심 차이 |
|---|---|---|
| [v1](v1.md) | `HiEMSegmenter` | SEM2 본진 — Gaussian likelihood + raw sticky-CRP |
| [v2](v2.md) | `HiEMSegmenter` | v1 segmenter 그대로 + STM round-clear / session-aware preload |
| [v3.1.1](v3.1.1.md) | `HiEMSegmenterV3` | Gaussian → bounded cosine `τ·cos(s, μ_k)` |
| [v3.2.1](v3.2.1.md) | `HiEMSegmenterV32` | sticky-CRP count 식에 sub-linear `(C_k+1)^β` |
| [v3.3.1](v3.3.1.md) | `HiEMSegmenterV331` | μ_k 대신 per-topic GRU 예측 ŝ_k로 cos 점수 |
| [v3.3.2](v3.3.2.md) | `HiEMSegmenterV332` | + `max_k cos < 1−pe_threshold` 면 새 topic 강제 (SEM2 surprise) |

## Cross-cutting infrastructure

버전 간 공유되는 인프라 / 설계 결정 (cache, locking, LLM 어댑터 호환 플래그 등) 은 별도:

- [infrastructure.md](infrastructure.md) — `EncoderCache`, `HiEMConvCache`, encoder lock, `--no-thinking`, retrieval policy, STM atomicity ...

## 표기 규약

- `s` : 새로 들어온 turn의 L2-normalized embedding (768-D)
- `μ_k` : topic k의 centroid
- `ŝ_k` : topic k의 *예측된 다음 embedding* (v3.3.1+에서만 의미 있음)
- `C_k` : topic k의 누적 assignment 수
- `e_{n-1}` : 직전 turn의 topic id
- `α` (alpha) : sCRP 새 cluster prior weight
- `λ` (lmda) : sCRP stickiness
- `τ` (tau) : cosine likelihood temperature (v3.x)
- `β` (beta) : sub-linear count exponent (v3.2+)
- `σ²₀` (sigma0_sq) : Gaussian cold-start 분산 (v1/v2 only)

## 작성 규칙

새 버전 추가 시:
1. `vX.md` 작성 — 이 README 의 템플릿(=직전 버전 파일) 구조 그대로 따라간다.
2. README 의 계보 그림과 한 줄 요약 표에 entry 추가.
3. `context/06-decision-log.md` 에 채택/폐기 사유 + 날짜 append.

기존 버전에 마이크로 변경 적용 시:
1. 해당 `vX.md` 의 "변경 이력" 섹션에 추가 (날짜 + 한 줄).
2. 알고리즘 의미가 바뀌면 새 버전 파일 (`vX.Y` 또는 `vX.Y.Z`) 분리. 단순 성능/캐시 최적화는 같은 파일 안에 누적.
