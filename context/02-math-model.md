# Hi-EM 수학 모형

---

## 확정 수식

### 쿼리 임베딩
$$\mathbf{s}_n = \text{normalize}(\text{encoder}(\text{query}_n))$$

### 생성 모형 (일반형)
$$P(\mathbf{s}_{1:N}, e_{1:N}) = \prod_{n=1}^{N} P(e_n \mid e_{1:n-1}, \mathbf{s}_{n-1}) \cdot P(\mathbf{s}_n \mid e_n, \cdot)$$

$P(\mathbf{s}_n \mid e_n, \cdot)$의 구체 형태는 **사건 모델 설계 후 확정**.

### sCRP Prior (SEM2와 동일 수식)
$$\Pr(e_n = k \mid e_{1:n-1}) \propto \begin{cases} C_k + \lambda \mathbb{I}[e_{n-1}=k] & k \leq K \\ \alpha & k = K+1 \end{cases}$$

### Topic 배정 (MAP)
$$\hat{e}_n = \arg\max_k\, \log \Pr(e_n = k \mid e_{1:n-1}) + \log P(\mathbf{s}_n \mid e_n = k, \cdot)$$

### Centroid / Variance 업데이트 (Welford, 어떤 사건 모델이든 공통)
n_e += 1
delta = s_n - mu_e
mu_e += delta / n_e
M2_e += delta * (s_n - mu_e)
sigma2_e = M2_e / n_e
---

## 미확정 수식 (Phase 0 완료 후 채워넣을 것)

### Topic 동역학 $P(\mathbf{s}_n \mid e_n, \cdot)$
- 옵션 A~F 중 결정 (`context/01-hi-em-design.md` 참조)
- 결정되면 이 섹션을 채워넣을 것

### Prediction Error / Boundary Score
사건 모델에 따라:
- 단순: $\|\mathbf{s}_n - \boldsymbol{\mu}_e\|^2$
- Multi-signal: $\sum_i w_i \cdot \text{sig}_i$
- 기타

### Topic Importance
$$I(e) = f(n_e, m_e, \Delta t_e, \text{neighbors})$$
구체 함수형 미정.

---

## 확정 하이퍼파라미터
| 파라미터 | 값 | 역할 |
|---|---|---|
| $\alpha$ | 1.0 | sCRP concentration |
| $\lambda$ | 10.0 | sCRP stickiness |
| $\sigma_0^2$ | 0.01 | variance prior |

## 미확정 하이퍼파라미터
- Boundary score 가중치
- Topic merge threshold
- Importance decay rate
- STM 용량

모두 **벤치마크 실험 후 경험적으로 결정.**