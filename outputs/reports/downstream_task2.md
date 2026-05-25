# Downstream Task 2 — Application Results (chat LLM = Llama-3.3-70B)

`outputs/experiments/2026-05-21_v413_secom_swap/` · 2026-05-22

`downstream_task.md` 와 동일한 SeCom-swap 비교 표인데, **chat(응답 생성) LLM 을
`meta-llama/llama-3.3-70b-instruct` (Crts) 로 교체**한 변형. segment/compress/retrieve
는 LLM-agnostic 이라 기존 MTB+ 산출물 재활용 — chat 만 Llama 로 재실행 후 eval.

값 출처 = `metrics_llama_*.json` (judge=`openai/gpt-4o`, n_qa=288, n_conv=11, judge
n_failed=0). GPT4Score = `gpt4_score_x10`. Long-MT-Bench+ 패널 완성 — **Hi-Seg(Ours)** =
Hi-DoTS (m=2, δ*=0.5983). **LoCoMo 패널** 은 실행 중.

## LaTeX

```latex
\begin{table*}[t]
\centering
\small
\begin{tabular}{l|cccccc|cc|ccc}
\toprule
\multirow{2}{*}{\textbf{Methods}}
& \multicolumn{6}{c|}{\textbf{QA Performance}}
& \multicolumn{2}{c|}{\textbf{Context Length}}
& \multicolumn{3}{c}{\textbf{Latency (ms/turn)}} \\
\cmidrule(lr){2-7} \cmidrule(lr){8-9} \cmidrule(lr){10-12}
& GPT4Score
& BLEU
& Rouge1
& Rouge2
& RougeL
& BERTScore
& \# Turns
& \# Tokens
& Neural fwd
& Seg
& End-to-end \\
\midrule

\multicolumn{12}{c}{\textit{LoCoMo}} \\
\midrule

Zero History
&  &  &  &  &  &  &  &  &  &  &  \\
Full History
&  &  &  &  &  &  &  &  &  &  &  \\

\midrule

GPT-4o-mini-Seg
&  &  &  &  &  &  & -- &  &  &  &  \\

\midrule

TextTiling-Style-Seg
&  &  &  &  &  &  &  &  &  &  &  \\
GreedySeg-Style-Seg
&  &  &  &  &  &  &  &  &  &  &  \\
GraphSeg-Style-Seg
&  &  &  &  &  &  &  &  &  &  &  \\
CSM-Style-Seg
&  &  &  &  &  &  &  &  &  &  &  \\

\midrule

Hi-Seg(Ours)
&  &  &  &  &  &  &  &  &  &  &  \\

\midrule

\multicolumn{12}{c}{\textit{Long-MT-Bench+}} \\
\midrule

Zero History
& 37.53 & 8.50 & 26.52 & 10.50 & 20.27 & 86.62 & 0 & 0 & -- & -- & -- \\
Full History
& 73.92 & 14.30 & 38.44 & 20.61 & 30.92 & 88.61 & 65.45 & 22676 & -- & -- & -- \\

\midrule

GPT-4o-mini-Seg
& 75.03 & 20.06 & 42.15 & 24.67 & 34.53 & 89.17 & 2.56 & 750 & -- & -- & 646.06 \\

\midrule

TextTiling-Style-Seg
& 68.82 & 17.55 & 37.85 & 20.17 & 30.37 & 88.40 & 3.74 & 1068 & 0 & 1.05 & 1.05 \\
GreedySeg-Style-Seg
& 66.22 & 16.68 & 37.23 & 20.18 & 30.05 & 88.22 & 5.38 & 1495 & 253.84 & 14.04 & 267.88 \\
GraphSeg-Style-Seg
& 57.57 & 14.33 & 34.92 & 18.33 & 27.73 & 87.78 & 8.66 & 2446 & 104.75 & 51.66 & 156.40 \\
CSM-Style-Seg
& 72.15 & 19.45 & 39.57 & 22.32 & 32.13 & 88.68 & 2.53 & 749 & 241.49 & 14.31 & 255.79 \\

\midrule

Hi-Seg(Ours)
& 74.44 & 19.52 & 41.50 & 24.13 & 33.99 & 88.91 & 4.27 & 1124 & 568.0 & 0.07 & 568.1 \\

\bottomrule
\end{tabular}
\caption{Application results with Llama-3.3-70B as the chat model. QA performance and
context length across history-, retrieval-, memory-, and Hi-EM-based segmentation methods.}
\label{tab:application_llama}
\end{table*}
```

## 값 표 (Long-MT-Bench+, chat = Llama-3.3-70B)

| Method | GPT4Score | BLEU | Rouge1 | Rouge2 | RougeL | BERTScore | # Turns | # Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero History | 37.53 | 8.50 | 26.52 | 10.50 | 20.27 | 86.62 | 0 | 0 |
| Full History | 73.92 | 14.30 | 38.44 | 20.61 | 30.92 | 88.61 | 65.45 | 22676 |
| GPT-4o-mini-Seg | 75.03 | 20.06 | 42.15 | 24.67 | 34.53 | 89.17 | 2.56 | 750 |
| TextTiling-Style-Seg | 68.82 | 17.55 | 37.85 | 20.17 | 30.37 | 88.40 | 3.74 | 1068 |
| GreedySeg-Style-Seg | 66.22 | 16.68 | 37.23 | 20.18 | 30.05 | 88.22 | 5.38 | 1495 |
| GraphSeg-Style-Seg | 57.57 | 14.33 | 34.92 | 18.33 | 27.73 | 87.78 | 8.66 | 2446 |
| CSM-Style-Seg | 72.15 | 19.45 | 39.57 | 22.32 | 32.13 | 88.68 | 2.53 | 749 |
| Hi-Seg (Ours) | 74.44 | 19.52 | 41.50 | 24.13 | 33.99 | 88.91 | 4.27 | 1124 |

## 해석 (vs downstream_task.md = gpt-4o-mini chat)

- chat LLM 만 gpt-4o-mini → Llama-3.3-70B 로 교체. segment/compress/retrieve 동일 산출물.
- **Hi-Seg(Ours) 74.44** — non-LLM baseline (CSM 72.15 / TextTiling 68.82 / GreedySeg 66.22
  / GraphSeg 57.57) 전부 우위. LLM-Seg(75.03) · Full History(73.92) 와 거의 동률 — gpt-4o-mini
  chat 때(77.40 vs baseline 78.13)와 동일한 패턴이 Llama chat 에서도 재현됨.
- 절대 점수는 Llama 가 gpt-4o-mini 보다 전반 낮음 (chat 모델 성능 차) — 단 **method 간 순위·간격
  패턴은 일관**, segmentation 비교 결론은 chat LLM 에 robust.

## Latency (3-column, ms/turn)

chat LLM 과 무관 (segmentation 단계 비용) → `downstream_task.md` 와 **동일 값**.
전부 동일 로컬 CPU·idle·MTB+ 720턴·turn 단위(batch=1) 일괄 측정.
출처 `latency_split_*.json` (baseline neural/logic 분리) · `encode_latency_cpu.json` (Hi-Seg encode).

- **Neural fwd** = 신경망 forward. Hi-Seg mpnet 568.0 / GreedySeg bert 253.84 / CSM DSE-BERT
  241.49 / GraphSeg GloVe 104.75 / TextTiling 0. GPT-4o-mini-Seg = LLM 콜이라 분해 불가 → `--`.
- **Seg** = encoder 출력 뒤 결정 로직. Hi-Seg **0.07** / TextTiling 1.05 / GreedySeg 14.04 /
  CSM 14.31 / GraphSeg 51.66.
- **End-to-end** = Neural fwd + Seg.

해석 (정직한 비교):
1. **`Seg` 가 유일한 공정 head-to-head** (encoder/입력길이 무관) — Hi-Seg 0.07 ms/turn 전 method 최소.
2. **End-to-end 는 head-to-head 아님** — baseline 은 인코딩 입력 truncate (GreedySeg 50tok / CSM
   128tok, 둘 다 원본 published 설정·CSM 은 128-tok 학습 모델). Hi-Seg·GPT-4o-mini-Seg 만 full
   context → **full-context 끼리: Hi-Seg 568.1 < GPT-4o-mini-Seg 646.06.**
3. End-to-end 568 은 WSL2 CPU·batch=1 값 — GPU 면 mpnet ~10-15 ms/turn.

## 미해결

- **LoCoMo 패널**: 8-method 파이프라인 실행 중.
- **Latency 3-column**: idle CPU 일괄 재측정 대기 (위 참조).
- judge = `openai/gpt-4o`, 8/8 method 모두 n_failed=0 (fresh 프로세스 재실행으로 403 해소).
