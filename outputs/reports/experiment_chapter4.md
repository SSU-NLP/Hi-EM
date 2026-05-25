# 4. Experiments

본 장은 (1) **DTS 벤치마크** 에서의 segmentation 품질 비교 (§4.1–§4.4) 와 (2)
**Long-MT-Bench+ (MTB+)** 다운스트림 QA 응답 품질 비교 (§4.5) 의 두 평가를
다룬다. 모든 실험은 **CPU 환경** 에서 수행 — 일부 임베딩 모델 (특히
`sentence-transformers/multi-qa-mpnet-base-dot-v1` int8 ONNX) 이 GPU API 가
없어 환경 일관성 (NO GPU) 으로 통일했다. **fine-tuned 모델 (CSM, RoBERTa) 의
결과는 시간 제약상 seed 3회 평균을 산출하지 못하고 random 1회 학습 결과로
보고**하며, 향후 작업에서 seed 평균으로 보완할 예정이다.

---

## 4.1 Datasets

### 4.1.1 DTS 벤치마크 (§4.2–§4.4 에서 사용)

세 개의 공개 dialogue topic segmentation 벤치마크를 **Def-DTS bundle**
(`benchmarks/Def-DTS/`) 형태로 통일해 평가에 사용한다.

| Benchmark | 전체 dialogue | Train (calibration) | Test | 도메인 | 주석 출처 |
|---|---:|---:|---:|---|---|
| **TIAGE** | 400 | 300 (75%) | 100 | open-domain crowdsourced topic shift | Xie et al., NAACL'21 |
| **Dialseg711** | 711 | 498 (70%) | 213 | Wikipedia 기반 인공 합성 | Xu et al., EMNLP'21 |
| **SuperDialseg** | 1722 | 400 (23%) | 1322 | document-grounded dialogue (DC-1) | Coldog2333 et al., EMNLP'23 |

- **Train split 의 역할**: 본 논문은 train 을 *모델 파라미터 학습* 이 아니라
  **δ\* calibration source** (§4.4 와 Fig. H) 또는 supervised 베이스라인
  (RoBERTa) 의 학습용으로만 사용. unsupervised 베이스라인 (TextTiling,
  GraphSeg, GreedySeg, CSM) 은 train 을 사용하지 않는다.
- **SuperDialseg 의 23% calibration ratio**: SuperDialseg 데이터셋은 1722
  dialogue 로 가장 큰데, calibration cap = **400 개** 를 적용해 TIAGE /
  Dialseg711 와 N≤300 비교 가능성을 유지했다 (나머지 1322 dialogue 는 test
  split). Figure H 의 convergence 실험은 N≤300 까지 다루므로 이 cap 이
  결과에 영향을 주지 않는다.

### 4.1.2 Long-MT-Bench+ (MTB+, §4.5 다운스트림 평가)

- **데이터셋**: `benchmarks/SeCom/experiment/data/mtbp/mtbp.jsonl`
- **n_conv = 11 dialogue**, 각 dialogue 는 평균 5 session, 총 65 turn
- **n_qa = 288 questions** (dialogue 당 평균 26 QA)
- 각 QA 는 dialogue 내부의 정보를 reference 해 답해야 하는 multi-session
  long-context QA 형태. SeCom 원논문 (Pan et al., 2024) 의 평가 설정 그대로.

---

## 4.2 Baselines

본 절에서는 비교 대상 baseline 들을 **(원본 offline 알고리즘 요약 + Hi-EM 의
online 변형 / 수정 사항)** 의 짝으로 정리한다. 모든 online 변형 코드는
`methods/<baseline>/online/` 에 있다.

### 4.2.1 Unsupervised baselines

#### **TextTiling** *(Hearst, 1997)*
- **Offline 원본**: 토큰화 + 불용어 제거 + bag-of-words → 슬라이딩 윈도우 간
  cosine block-similarity → depth score 계산 → global threshold (μ−σ) 적용.
  NLTK `nltk.tokenize.TextTilingTokenizer` 기본 파라미터 (w=10, k=6) 사용.
- **Hi-EM 변형 (online, prefix-causal)**: `methods/texttiling/online/prefix.py`
  는 prefix-recompute (매 turn 마다 NLTK 를 `u_{1..t}` 에 fresh 호출, O(t)/turn).
  `methods/texttiling/online/streaming.py` 는 자체 구현된 incremental
  block-cosine + Welford running threshold (one-sided depth, O(w)/turn) —
  본 논문 메인 비교에서 사용된 **TextTiling-Style-Seg** 가 이쪽이다.

#### **GraphSeg** *(Glavaš et al., 2016)*
- **Offline 원본**: 발화 → POS 태깅 + 불용어/GloVe 필터 + Information
  Content lookup → IC×GloVe 가중 cosine matrix → Hungarian assignment →
  유사도 graph + Bron-Kerbosch maximal clique → sequential merge.
- **Hi-EM 변형 (online, windowed)**: `methods/graphseg/online_window.py` —
  원본 알고리즘을 **window=d 안에서만** 적용 (메모리/지연 한정). 강한
  "online" 명명은 codex review 결과 회피하고 `GraphSeg-window-d` 로 명명.
  본 논문 표에서는 단순히 **GraphSeg-Style-Seg** 로 표기.

#### **GreedySeg** *(Xu et al., 2021)*
- **Offline 원본**: BERT-base 의 발화 임베딩 → cosine 거리 → argmin greedy
  로 가장 dissimilar 한 인접 발화 사이를 boundary 로 선언. 슬라이딩 윈도우
  내 다른 발화와의 평균 cosine 비교.
- **Hi-EM 변형 (online, delay=2)**: `methods/greedyseg/online/delay2.py` —
  **delay=2 bounded lookahead** (turn t 의 boundary 는 t+2 도착 시 결정).
  원본의 score 공식·HP·argmin greedy 선택을 그대로 보존하되, 미래 context
  를 2-turn 으로 제한해 streaming 호환. device-agnostic (cuda/mps/cpu).

#### **CSM** *(Xing & Carenini, 2021)*
- **Offline 원본**: lxing532 의 **CoherenceNet** (BERT NSP 위에 MLP head)
  로 인접 발화쌍의 coherence score → TextTiling-style depth + threshold 로
  boundary 결정. SuperDialseg `CSMSegmenter` 의 표준 구현.
- **Hi-EM 변형 (online, delay=2)**: `methods/CSM/online/delay2.py` (wrap
  `src/hi_em/baselines/csm_online.py`) — streaming CoherenceNet + depth,
  **delay=2 right context**. 핵심 수정:
  1. score 함수에 **`sigmoid(logits[0, 0])`** 적용 (원본 raw logit → 안정화).
  2. **alpha=1.0** (원본 default 0.0 → paper cut_rate 일치하도록 변경).
  3. **off-by-one boundary index 수정** (`owner_t = gi + 2`, push 와 flush 모두).
  ckpt = `outputs/runs/_misc/cpt_277000.pth` (lxing532 fine-tuned weights),
  backbone = `bert-base-uncased`.

### 4.2.2 Supervised baseline

#### **RoBERTa** *(Coldog2333 et al., EMNLP'23, Table 3)*
- **Offline 원본**: `RobertaForTokenClassification` 위에 발화 끝 첫 `</s>`
  토큰의 binary label (boundary / non-boundary) 학습. 슬라이딩 윈도우
  |T|=20 으로 학습 + 추론, 경계 결정 시 미래 20 발화 포함 logit 평균.
- **Hi-EM 변형 (online, strict causal)**: `methods/RoBERTa/online/segment.py`
  — offline 학습된 체크포인트를 그대로 재사용, **추론만** 변경. 경계
  (t-1, t) 를 turn t 도착 시점의 causal window `u_{..t}` 하나로 *1회* 결정.
  미래 발화 0개, 재수정 없음, O(1)/turn. 학습은 안 함.
- **학습 제약**: 시간 관계상 **random seed 1회 학습** 으로 보고. 본 표에
  std 가 없는 이유.

### 4.2.3 LLM-based baselines (SeCom segmenter)

SeCom (Pan et al., 2024) 의 **LLM 기반 segmenter** 를 그대로 사용. 한 session
의 모든 발화를 prompt 로 주고 LLM 이 segment boundary 를 출력하도록 함
(non-streaming, full-session-at-once). 우리는 단지 segment LLM 만 swap.

| LLM | params | Crts slug |
|---|---:|---|
| GPT-4o-mini | ~8B (est.) | `openai/gpt-4o-mini` |
| GPT-5 | (closed) | `openai/gpt-5` (reasoning_effort=minimal) |
| Qwen3.5-122B-A10B | 122B (MoE, active 10B) | `qwen/qwen3.5-122b-a10b` |
| Qwen3.5-27B | 27B | `qwen/qwen3.5-27b` |
| Qwen3.5-4B | 4B | `qwen/qwen3.5-4b` |
| Qwen3.5-2B | 2B | `qwen/qwen3.5-2b` |
| Llama3.2-3B | 3B | `meta/llama-3.2-3b-instruct` |
| Mistral3-3B | 3B | `mistralai/ministral-3b` |

모든 LLM 은 **hybrid-thinking 비활성화** 상태로 호출 (Qwen3.5 → `reasoning_effort=none`,
GPT-5 → `reasoning_effort=minimal`) — apples-to-apples 비교를 위해 reasoning
chain 사용 안 함.

### 4.2.4 Our method: **Hi-OnTop** (Hi-DoTS 의 reduced form)

본 논문의 메인 모델. **메인 비교표 (§4.4, §4.5)** 에서 6 variant 보고:
- 인코더 2종: **MPNet** (`sentence-transformers/multi-qa-mpnet-base-dot-v1`,
  768-d, fp32) / **MiniLM-int8** (`all-MiniLM-L6-v2`, 384-d, ONNX `quint8_avx2`).
- δ\* percentile 3종: **p60 / p70 / p80** (label-free 추정값).
- δ_eff 식 (§3.3): context window m=2, decay ρ=0.7, blend a=0.5.

**Figure H (calibration N convergence ablation)** 만 인코더 3종 사용 —
위 2종에 더해 **MiniLM-fp32** (`all-MiniLM-L6-v2`, 384-d, non-quantized) 를
중간 ground truth 로 포함. 메인 표에는 보고하지 않음.

---

## 4.3 Metrics

### 4.3.1 DTS metrics (§4.4)

세 가지 표준 segmentation metric 의 **mean** + **composite Score**:

- **Pk (Beeferman et al., 1999)** ↓ — segment boundary disagreement
  probability. 낮을수록 좋음.
- **WinDiff (WD)** ↓ — Pk 의 변형, boundary count 도 함께 본다. 낮을수록 좋음.
- **F1** ↑ — boundary set 의 micro-F1 (pred boundary set vs gold).
- **Score** ↑ — *composite metric*:

  $$\text{Score} = 0.5 \cdot \text{F1} + 0.25 \cdot (1 - \text{Pk}) + 0.25 \cdot (1 - \text{WD})$$

  (Methods/README.md 정의 동일.) Pk/WD/F1 의 균형을 단일 숫자로 보기 위한 도구.

평가 라이브러리는 모두 `segeval` (autoseg 의 wrapper 사용), Def-DTS bundle 데이터.
**예외: RoBERTa** — 학습 데이터가 SuperDialseg train 만 사용하므로 metric 도
원논문 5.3 의 *sliding window = avg seg length / 2* 방식 official Pk/WD 사용
(autoseg segeval 미사용). Score 공식은 동일.

### 4.3.2 Downstream QA metrics (§4.5)

LLM-based QA 응답에 대해 6 metric 보고:

- **GPT4Score** ↑ — `openai/gpt-4o` judge 가 1–10 점 부여 → ×10 정규화.
  본 논문의 *primary* metric. 출력 풍부, 의미·완전성 평가.
- **BLEU** ↑ (sacreBLEU)
- **Rouge-1 / Rouge-2 / Rouge-L** ↑
- **BERTScore F1** ↑ (`bert_score` 라이브러리 default `roberta-large`, CPU)

Context length 두 종:
- **# Turns** — retrieved 후 chat LLM 에 입력되는 turn 수
- **# Tokens** — 같은 입력의 tokenizer 토큰 수

Latency 두 종 (ms/turn):
- **Pre. (Preprocess)** — 결정 *이전* 의 표현 추출 (encoder forward + 어휘 연산)
- **Seg. (Segmentation decision)** — preprocess 출력 이후의 결정 로직만.
  LLM segmenter 의 경우 API 1회 monolithic latency.

---

## 4.4 Experimental Setup

### 4.4.1 환경 (DTS + downstream 공통)

- **CPU only**: AMD Ryzen 9 7950X · 16 logical cores · 16 GB RAM · WSL2
  Linux 5.15 (no GPU). 일부 임베딩 모델 (MiniLM-int8 ONNX) 에 GPU 백엔드
  API 가 없는 점 + 환경 일관성을 위해 모든 latency 측정 / segmentation 을
  CPU 로 통일.
- **batch_size = 1** (per-turn streaming 시뮬레이션). 단, SeCom 의
  segmenter 가 session 단위로 LLM 1회 호출하는 부분은 batch=1 이 무의미
  (한 session 의 모든 발화가 한 prompt 안에 들어감).
- **idle CPU 측정**: latency 는 다른 background job 이 없는 상태에서 측정 —
  pipeline (LLMLingua-2 compress 등) 과 동시 측정 시 CPU contention 으로
  noise 가 끼는 것을 별도 확인했다.

### 4.4.2 δ\* calibration (Hi-OnTop)

DTS 벤치마크 평가용 calibration 은 *각 벤치의 train split* 에서 δ_eff 를
모아 percentile 값을 δ\* 로 채택 — calibration 데이터셋·split 분량:

| Benchmark | Train (calibration) | 전체 train pool | 비고 |
|---|---:|---:|---|
| TIAGE | 300 | 300 (cap=전체) | dataset 자체가 400 dialogue |
| Dialseg711 | 498 | 498 (cap=전체) | dataset 711 dialogue |
| SuperDialseg | 400 | 400 (cap=400) | dataset 1722 dialogue, 400 으로 cap |

- **Layer 1 — percentile rank 선택** (p60/p70/p80): segmentation 벤치
  (TIAGE/Dialseg711/SuperDialseg) 의 F1·Score 로 선택. 다운스트림
  (Long-MT-Bench+) 의 GPT4Score 로 선택하지 **않음** — in-sample
  selection bias 회피.
- **Layer 2 — δ\* 절대값**: deploy 도메인의 *unlabeled* δ_eff 분포의
  해당 percentile (label-free, leakage 없음). **DTS 평가용** 은 각 벤치
  (TIAGE/Dialseg711/SuperDialseg) train split 에서 percentile 채택 — 벤치별로
  값 다름 (dts_result.md 의 δ\* calibration 표 참조). **다운스트림 (MTB+)
  평가용** 은 MTB+ 자체의 unlabeled δ_eff pool 에서 percentile 채택:
  MPNet δ\*=0.4799, MiniLM-int8 δ\*=0.7049 (인코더별 분포 영역이 달라 절대값
  다름; Observation 1 §3.3 참조).
- **p60 / p80 ablation**: percentile 민감도 보고용 (메인 모델은 p70).

### 4.4.3 Random seed

- Hi-OnTop 의 calibration 자체는 deterministic (전체 train pool 에서
  percentile 계산) → seed 무관.
- CSM, RoBERTa fine-tuning **= seed 1회 random run** (시간 제약, §4
  도입부 명시).
- LLM segmenter 호출은 stochastic (temperature 1.0 기본) 이지만 1회 실행 보고.

---

## 4.5 Downstream Task — Long-MT-Bench+ Application

### 4.5.1 Experimental setup (downstream)

#### 데이터셋
- **MTB+** (Long-MT-Bench+) — 11 conv, 288 QA.
- 평가 split 은 SeCom 원논문 그대로 사용 (cross-validation 없음, 본 데이터셋이
  evaluation-only 로 설계됨).

#### Baselines
DTS 평가용 6 unsup + 1 sup + Hi-OnTop 6 variant 에 더해, SeCom 의 원래
**LLM 기반 분절 (SeCom segmenter)** 들을 함께 비교:

- GPT-4o-mini-Seg, **GPT-5-Seg**, Qwen3.5-{2B/4B/27B/**122B-A10B**}-Seg,
  Llama3.2-3B-Seg, Mistral3-3B-Seg

추가로 두 가지 *no-DTS* upper/lower bound 도 보고:
- **Zero History** — chat LLM 에 dialogue history 0 제공 (lower bound).
- **Full History** — 전체 dialogue (~65 turn / 22.7K token) 그대로 전달
  (upper bound, retrieval/segmentation 없이).

#### Metrics
§4.3.2 참조 — primary = **GPT4Score** (gpt-4o judge), secondary = BLEU /
Rouge-{1,2,L} / BERTScore F1. Context length (# Turns, # Tokens) 와 segmentation
latency 도 함께.

#### Setup (pipeline)
SeCom 표준 pipeline 의 *segmentation 단계만* swap, 나머지는 원논문 그대로:

| 단계 | 도구 | HP |
|---|---|---|
| ① Segmentation | (swap 대상) | 각 method 자체 default |
| ② Compression | LLMLingua-2 (local HF, ~2 GB CPU) | `compress_rate = 0.75` |
| ③ Retrieval | dense bi-encoder (multi-qa-mpnet) | `topk = 1` |
| ④ Chat (응답 생성) | `openai/gpt-4o-mini` | workers=8, temperature=0.0, max_tokens=512 |
| ⑤ Eval (judge) | `openai/gpt-4o` | workers=8, JUDGE_PROMPT 동일 |

- **온라인의 범위 제한**: 본 논문은 *segmentation* 만 online (causal,
  prefix-only) 으로 한정. retrieval / compression / chat 은 SeCom 원본 그대로
  offline (segment 결정 이후의 후속 단계). 다운스트림 표의 Pre./Seg. latency
  는 segmentation 단계만의 ms/turn 이다.
- **Hi-OnTop p 선택**: percentile p ∈ {60, 70, 80} 은 **DTS 벤치마크
  (§4.4) 의 F1/Score** 만 보고 선택함 — MTB+ 의 GPT4Score 를 보지 않음.
  즉 본 표의 p70 best 결과는 **out-of-distribution selection** 의 산물.
  이 selection protocol 은 §4.4.2 Layer 1 과 동일.

실제 수치 및 해석은 **§5 (Results)** 참조 — Hi-OnTop 6 variant + algorithmic
unsup 4종 + supervised RoBERTa + LLM 8종 + history bound 2종 의 11×17 비교표.
DTS 결과는 `outputs/reports/dts_result.md`, 다운스트림은
`outputs/reports/downstream_task.md` 에 저장되어 있다.
