"""Per-turn orchestration of embedding + segmentation + LTM + memory window + LLM.

A single ``HiEM`` instance handles one ``conv_id``.

Two memory regimes:

* ``use_stm=False`` (default; Phase 2 baseline) — every turn calls the
  stateless :func:`select_memory_window` over LTM (cosine top-k topics ×
  recency top-k turns).
* ``use_stm=True`` (Phase 2-Full) — STM is the working buffer + cache:

    - **Every turn lands in STM** (per spec: "단기 메모리에 모든 대화 원문
      저장"). New topic? Create a new STM entry seeded with this turn pair.
      Existing topic in STM? Append in-sync. Topic atomicity preserved.
    - LTM also gets a sync write every turn (dual-write; current = target
      per phase-2-full-design.md §0.1 "LTM 쓰기 시점: 매 턴 sync").
    - Within a round: STM membership is fixed except (a) cache miss for the
      current topic triggers a one-time promotion from LTM, (b) the just-
      finished user/assistant pair is appended to its topic in STM.
    - At round boundary (every ``round_size`` user turns = ``2*round_size``
      jsonl rows): :class:`RoundProcessor` recomputes normalized importance,
      promotes topics ≥ threshold (refreshing with full LTM contents), and
      evicts to capacity. Runs on a daemon thread by default.

Resuming an existing conversation is *not* supported here: the segmenter
is rebuilt fresh each session (centroids in ``<conv_id>.state.json`` are
written but never loaded back). Phase 4 evaluation runs whole conversations
end-to-end in one process.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hi_em.embedding import QueryEncoder
from hi_em.llm import OpenAIChatLLM
from hi_em.ltm import LTM
from hi_em.memory_window import MemoryWindow, select_memory_window
from hi_em.round_processor import RoundProcessor
from hi_em.sem_core import HiEMSegmenter
from hi_em.sem_core_optimize import HiEMSegmenterV3


class HiEM:
    """Drives one conversation: each ``handle_turn`` is a full pipeline pass."""

    def __init__(
        self,
        conv_id: str,
        encoder: QueryEncoder,
        llm: OpenAIChatLLM,
        model: str,
        ltm_root: Path | str,
        alpha: float = 1.0,
        lmda: float = 10.0,
        sigma0_sq: float = 0.01,
        k_topics: int = 3,
        k_turns_per_topic: int = 5,
        system_prompt: str | None = None,
        response_filter: Callable[[str], str] | None = None,
        # ---- Phase 2-Full STM ---------------------------------------
        use_stm: bool = False,
        round_size: int = 10,
        stm_max_topics: int = 10,
        stm_max_turns: int = 200,
        promotion_threshold: float = 0.5,
        importance_alpha: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
        importance_version: str = "v1",
        importance_v2_extra: dict | None = None,
        lambda_r: float = 0.5,
        lambda_freq: float = 0.5,
        min_floor: float = 0.1,
        round_async: bool = True,
        round_clear_stm: bool = False,
        # ---- v3.x (Bounded Cosine variants) -------------------------
        version: str = "v3.3.9",  # 2026-05-18 current BEST (TIAGE DTS)
        tau: float = 50.0,
        cos_threshold: float = 0.7,
        beta: float = 0.5,  # v3.2/v3.3: sub-linear sCRP count exponent
        rnn_hidden_dim: int = 32,
        rnn_lr: float = 1e-3,
        rnn_train_steps: int = 1,
        rnn_max_context: int = 8,
        rnn_min_history: int = 2,
        pe_threshold: float = 1.0,  # v3.3.2: surprise-driven boundary
        # ---- v3.3.3 only (f0 / SEM2 restart branch) -----------------
        restart_pe_threshold: float = 0.5,
        restart_margin: float = 0.0,
        f0_tau: float | None = None,
        f0_min_starts: int = 2,
        # ---- v3.3.4 only (per-topic PE variance calibration) --------
        pe_var_decay: float = 0.95,
        pe_var_min_samples: int = 5,
        pe_var_sigma0_sq: float = 0.04,
        pe_var_min_sq: float = 1e-4,
        pe_var_max_sq: float = 0.25,
        var_likelihood_weight: float = 1.0,
        hard_pe_fallback: bool = False,
        # ---- v3.3.5 only (SEM2 f_is_trained cold-start gating) ------
        min_transitions_for_pe: int = 1,
        # ---- v3.3.6 only (SEM2-faithful event dynamics) -------------
        rnn_n_epochs: int = 10,
        rnn_ready_min_transitions: int = 3,
        rnn_max_history: int = 64,
        # ---- v3.3.7 only (SEM2 map_variance posterior σ²) -----------
        pe_var_df0: float = 1.0,
        pe_var_window: int = 256,
        # ---- v3.3.8 only (SEM2-calibrated fresh baseline) -----------
        pe_prior: float = 1.0,
        # ---- v3.3.6+ reproducibility -------------------------------
        seed: int = 0,
        # ---- v3.3.3-2 only (prototype f0 + posterior odds) ----------
        f0_proto_max: int = 4,
        restart_p_threshold: float = 0.35,
        restart_pe_min: float = 0.0,
        # ---- v3.3.3-3 only (restart hysteresis + prototype softening)
        restart_prob_margin: float = 0.15,
        episode_min_span: int = 4,
        f0_proto_weight: float = 0.25,
        # ---- v3.3.4-2 only (σ² shrinkage + robust scale) ------------
        pe_var_shrink_c: float = 8.0,
        pe_var_robust: bool = False,
        # ---- v3.3.3-4 only (retrieval atomicity + dormant LTM) ------
        retrieval_mode: str = "stm_all_turns",  # or "episode_rerank"
        episode_top_k: int = 4,
        dormant_ltm_top_n: int = 0,             # 0 → off
        rerank_query_weight: float = 1.0,
        rerank_topic_weight: float = 0.10,
        rerank_episode_weight: float = 0.35,
        rerank_pe_penalty: float = 0.05,
        rerank_recency_weight: float = 0.03,
        # -------------------------------------------------------------
        **llm_kwargs: Any,
    ) -> None:
        self.conv_id = conv_id
        self._encoder = encoder
        self._llm = llm
        self._model = model
        self._ltm = LTM(ltm_root)
        self._version = version
        if version == "v3.1.1":
            self._segmenter = HiEMSegmenterV3(
                dim=encoder.dim, alpha=alpha, lmda=lmda,
                tau=tau, cos_threshold=cos_threshold,
            )
        elif version == "v3.2.1":
            from hi_em.sem_core_optimize_pe import HiEMSegmenterV32
            self._segmenter = HiEMSegmenterV32(
                dim=encoder.dim, alpha=alpha, lmda=lmda,
                tau=tau, cos_threshold=cos_threshold, beta=beta,
            )
        elif version == "v3.3.1":
            from hi_em.sem_core_v331_rnn import HiEMSegmenterV331
            self._segmenter = HiEMSegmenterV331(
                dim=encoder.dim, alpha=alpha, lmda=lmda,
                tau=tau, cos_threshold=cos_threshold, beta=beta,
                rnn_hidden_dim=rnn_hidden_dim,
                rnn_lr=rnn_lr,
                rnn_train_steps=rnn_train_steps,
                rnn_max_context=rnn_max_context,
                rnn_min_history=rnn_min_history,
            )
        elif version == "v3.3.2":
            from hi_em.sem_core_v332_rnn_pe import HiEMSegmenterV332
            self._segmenter = HiEMSegmenterV332(
                dim=encoder.dim, alpha=alpha, lmda=lmda,
                tau=tau, cos_threshold=cos_threshold, beta=beta,
                pe_threshold=pe_threshold,
                rnn_hidden_dim=rnn_hidden_dim,
                rnn_lr=rnn_lr,
                rnn_train_steps=rnn_train_steps,
                rnn_max_context=rnn_max_context,
                rnn_min_history=rnn_min_history,
            )
        elif version == "v3.3.3":
            from hi_em.sem_core_v333_rnn_f0 import HiEMSegmenterV333
            self._segmenter = HiEMSegmenterV333(
                dim=encoder.dim, alpha=alpha, lmda=lmda,
                tau=tau, cos_threshold=cos_threshold, beta=beta,
                pe_threshold=pe_threshold,
                rnn_hidden_dim=rnn_hidden_dim,
                rnn_lr=rnn_lr,
                rnn_train_steps=rnn_train_steps,
                rnn_max_context=rnn_max_context,
                rnn_min_history=rnn_min_history,
                restart_pe_threshold=restart_pe_threshold,
                restart_margin=restart_margin,
                f0_tau=f0_tau,
                f0_min_starts=f0_min_starts,
            )
        elif version == "v3.3.4":
            from hi_em.sem_core_v334_rnn_var import HiEMSegmenterV334
            self._segmenter = HiEMSegmenterV334(
                dim=encoder.dim, alpha=alpha, lmda=lmda,
                tau=tau, cos_threshold=cos_threshold, beta=beta,
                pe_threshold=pe_threshold,
                rnn_hidden_dim=rnn_hidden_dim,
                rnn_lr=rnn_lr,
                rnn_train_steps=rnn_train_steps,
                rnn_max_context=rnn_max_context,
                rnn_min_history=rnn_min_history,
                pe_var_decay=pe_var_decay,
                pe_var_min_samples=pe_var_min_samples,
                pe_var_sigma0_sq=pe_var_sigma0_sq,
                pe_var_min_sq=pe_var_min_sq,
                pe_var_max_sq=pe_var_max_sq,
                var_likelihood_weight=var_likelihood_weight,
                hard_pe_fallback=hard_pe_fallback,
            )
        elif version == "v3.3.3-2":
            from hi_em.sem_core_v333_2 import HiEMSegmenterV333_2
            self._segmenter = HiEMSegmenterV333_2(
                dim=encoder.dim, alpha=alpha, lmda=lmda,
                tau=tau, cos_threshold=cos_threshold, beta=beta,
                pe_threshold=pe_threshold,
                rnn_hidden_dim=rnn_hidden_dim,
                rnn_lr=rnn_lr,
                rnn_train_steps=rnn_train_steps,
                rnn_max_context=rnn_max_context,
                rnn_min_history=rnn_min_history,
                f0_tau=f0_tau,
                f0_min_starts=f0_min_starts,
                f0_proto_max=f0_proto_max,
                restart_p_threshold=restart_p_threshold,
                restart_pe_min=restart_pe_min,
            )
        elif version == "v3.3.3-3":
            from hi_em.sem_core_v333_3 import HiEMSegmenterV333_3
            self._segmenter = HiEMSegmenterV333_3(
                dim=encoder.dim, alpha=alpha, lmda=lmda,
                tau=tau, cos_threshold=cos_threshold, beta=beta,
                pe_threshold=pe_threshold,
                rnn_hidden_dim=rnn_hidden_dim,
                rnn_lr=rnn_lr,
                rnn_train_steps=rnn_train_steps,
                rnn_max_context=rnn_max_context,
                rnn_min_history=rnn_min_history,
                f0_tau=f0_tau,
                f0_min_starts=f0_min_starts,
                f0_proto_max=f0_proto_max,
                f0_proto_weight=f0_proto_weight,
                restart_p_threshold=restart_p_threshold,
                restart_prob_margin=restart_prob_margin,
                episode_min_span=episode_min_span,
                restart_pe_min=restart_pe_min,
            )
        elif version == "v3.3.3-4":
            from hi_em.sem_core_v333_4 import HiEMSegmenterV333_4
            self._segmenter = HiEMSegmenterV333_4(
                dim=encoder.dim, alpha=alpha, lmda=lmda,
                tau=tau, cos_threshold=cos_threshold, beta=beta,
                pe_threshold=pe_threshold,
                rnn_hidden_dim=rnn_hidden_dim,
                rnn_lr=rnn_lr,
                rnn_train_steps=rnn_train_steps,
                rnn_max_context=rnn_max_context,
                rnn_min_history=rnn_min_history,
                f0_tau=f0_tau,
                f0_min_starts=f0_min_starts,
                f0_proto_max=f0_proto_max,
                f0_proto_weight=f0_proto_weight,
                restart_p_threshold=restart_p_threshold,
                restart_prob_margin=restart_prob_margin,
                episode_min_span=episode_min_span,
                restart_pe_min=restart_pe_min,
            )
            # v3.3.3-4 default (2026-05-11 재설계): importance-only retrieval
            # + importance v2 (PE + boundary + persistence) policy.
            if retrieval_mode == "stm_all_turns":
                retrieval_mode = "importance_only"
            if promotion_threshold == 0.5:
                promotion_threshold = 0.3
            # importance v2: codex 2026-05-11 권장 default weights (7-tuple)
            # (w1 count, w2 freq, w3 recency, w4 nbr, w5 PE, w6 boundary, w7 span)
            if importance_version == "v1" and importance_alpha == (1.0, 1.0, 1.0, 1.0):
                importance_version = "v2"
                importance_alpha = (0.70, 0.60, 0.45, 0.35, 0.90, 0.70, 0.25)
        elif version == "v3.3.4-2":
            from hi_em.sem_core_v334_2 import HiEMSegmenterV334_2
            self._segmenter = HiEMSegmenterV334_2(
                dim=encoder.dim, alpha=alpha, lmda=lmda,
                tau=tau, cos_threshold=cos_threshold, beta=beta,
                pe_threshold=pe_threshold,
                rnn_hidden_dim=rnn_hidden_dim,
                rnn_lr=rnn_lr,
                rnn_train_steps=rnn_train_steps,
                rnn_max_context=rnn_max_context,
                rnn_min_history=rnn_min_history,
                pe_var_decay=pe_var_decay,
                pe_var_min_samples=pe_var_min_samples,
                pe_var_sigma0_sq=pe_var_sigma0_sq,
                pe_var_min_sq=pe_var_min_sq,
                pe_var_max_sq=pe_var_max_sq,
                var_likelihood_weight=var_likelihood_weight,
                hard_pe_fallback=hard_pe_fallback,
                pe_var_shrink_c=pe_var_shrink_c,
                pe_var_robust=pe_var_robust,
            )
        elif version == "v3.3.5":
            from hi_em.sem_core_v335 import HiEMSegmenterV335
            self._segmenter = HiEMSegmenterV335(
                dim=encoder.dim, alpha=alpha, lmda=lmda,
                tau=tau, cos_threshold=cos_threshold, beta=beta,
                pe_threshold=pe_threshold,
                rnn_hidden_dim=rnn_hidden_dim,
                rnn_lr=rnn_lr,
                rnn_train_steps=rnn_train_steps,
                rnn_max_context=rnn_max_context,
                rnn_min_history=rnn_min_history,
                pe_var_decay=pe_var_decay,
                pe_var_min_samples=pe_var_min_samples,
                pe_var_sigma0_sq=pe_var_sigma0_sq,
                pe_var_min_sq=pe_var_min_sq,
                pe_var_max_sq=pe_var_max_sq,
                var_likelihood_weight=var_likelihood_weight,
                hard_pe_fallback=hard_pe_fallback,
                min_transitions_for_pe=min_transitions_for_pe,
                restart_pe_threshold=restart_pe_threshold,
                restart_margin=restart_margin,
                f0_min_starts=f0_min_starts,
            )
        elif version == "v3.3.6":
            from hi_em.sem_core_v336 import HiEMSegmenterV336
            self._segmenter = HiEMSegmenterV336(
                dim=encoder.dim, alpha=alpha, lmda=lmda,
                tau=tau, cos_threshold=cos_threshold, beta=beta,
                pe_threshold=pe_threshold,
                rnn_hidden_dim=rnn_hidden_dim,
                rnn_lr=rnn_lr,
                pe_var_decay=pe_var_decay,
                pe_var_min_samples=pe_var_min_samples,
                pe_var_sigma0_sq=pe_var_sigma0_sq,
                pe_var_min_sq=pe_var_min_sq,
                pe_var_max_sq=pe_var_max_sq,
                var_likelihood_weight=var_likelihood_weight,
                hard_pe_fallback=hard_pe_fallback,
                min_transitions_for_pe=min_transitions_for_pe,
                restart_pe_threshold=restart_pe_threshold,
                restart_margin=restart_margin,
                f0_min_starts=f0_min_starts,
                rnn_n_epochs=rnn_n_epochs,
                rnn_ready_min_transitions=rnn_ready_min_transitions,
                rnn_max_history=rnn_max_history,
                seed=seed,
            )
        elif version == "v3.3.7":
            from hi_em.sem_core_v337 import HiEMSegmenterV337
            self._segmenter = HiEMSegmenterV337(
                dim=encoder.dim, alpha=alpha, lmda=lmda,
                tau=tau, cos_threshold=cos_threshold, beta=beta,
                pe_threshold=pe_threshold,
                rnn_hidden_dim=rnn_hidden_dim,
                rnn_lr=rnn_lr,
                pe_var_sigma0_sq=pe_var_sigma0_sq,
                pe_var_df0=pe_var_df0,
                pe_var_min_sq=pe_var_min_sq,
                pe_var_max_sq=pe_var_max_sq,
                pe_var_window=pe_var_window,
                var_likelihood_weight=var_likelihood_weight,
                hard_pe_fallback=hard_pe_fallback,
                min_transitions_for_pe=min_transitions_for_pe,
                restart_pe_threshold=restart_pe_threshold,
                restart_margin=restart_margin,
                f0_min_starts=f0_min_starts,
                rnn_n_epochs=rnn_n_epochs,
                rnn_ready_min_transitions=rnn_ready_min_transitions,
                rnn_max_history=rnn_max_history,
                seed=seed,
            )
        elif version == "v3.3.8":
            from hi_em.sem_core_v338 import HiEMSegmenterV338
            self._segmenter = HiEMSegmenterV338(
                dim=encoder.dim, alpha=alpha, lmda=lmda,
                tau=tau, cos_threshold=cos_threshold, beta=beta,
                pe_threshold=pe_threshold,
                rnn_hidden_dim=rnn_hidden_dim,
                rnn_lr=rnn_lr,
                pe_var_sigma0_sq=pe_var_sigma0_sq,
                pe_var_df0=pe_var_df0,
                pe_var_min_sq=pe_var_min_sq,
                pe_var_max_sq=pe_var_max_sq,
                pe_var_window=pe_var_window,
                pe_prior=pe_prior,
                var_likelihood_weight=var_likelihood_weight,
                hard_pe_fallback=hard_pe_fallback,
                min_transitions_for_pe=min_transitions_for_pe,
                restart_pe_threshold=restart_pe_threshold,
                restart_margin=restart_margin,
                f0_min_starts=f0_min_starts,
                rnn_n_epochs=rnn_n_epochs,
                rnn_ready_min_transitions=rnn_ready_min_transitions,
                rnn_max_history=rnn_max_history,
                seed=seed,
            )
        elif version == "v3.3.9":
            # 2026-05-18 current BEST (TIAGE test, target WD/F1/Pk + ARI
            # guard): F1 0.437 / WD 0.605 / Pk 0.415 / ARI 0.408 — best
            # of 13 methods, non-degenerate. prev-cos restored as SEM2
            # identity-dynamics PE w/ prior-corrected baseline. v3.3.9
            # __init__ defaults (eta_prev=1.0, delta_star=0.5557,
            # sigma_delta_c=0.0625) ARE the best config — pass through
            # only the shared HP; do not override the v3.3.9-specific
            # calibration defaults.
            from hi_em.sem_core_v339 import HiEMSegmenterV339
            self._segmenter = HiEMSegmenterV339(
                dim=encoder.dim, alpha=alpha, lmda=lmda,
                tau=tau, cos_threshold=cos_threshold, beta=beta,
                pe_threshold=pe_threshold,
                rnn_hidden_dim=rnn_hidden_dim,
                rnn_lr=rnn_lr,
                pe_var_sigma0_sq=pe_var_sigma0_sq,
                pe_var_df0=pe_var_df0,
                pe_var_min_sq=pe_var_min_sq,
                pe_var_max_sq=pe_var_max_sq,
                pe_var_window=pe_var_window,
                pe_prior=pe_prior,
                var_likelihood_weight=var_likelihood_weight,
                hard_pe_fallback=hard_pe_fallback,
                min_transitions_for_pe=min_transitions_for_pe,
                restart_pe_threshold=restart_pe_threshold,
                restart_margin=restart_margin,
                f0_min_starts=f0_min_starts,
                rnn_n_epochs=rnn_n_epochs,
                rnn_ready_min_transitions=rnn_ready_min_transitions,
                rnn_max_history=rnn_max_history,
                seed=seed,
            )
        elif version == "v2":
            self._segmenter = HiEMSegmenter(
                dim=encoder.dim, alpha=alpha, lmda=lmda, sigma0_sq=sigma0_sq
            )
        else:
            raise ValueError(
                f"unknown HiEM version: {version!r} "
                "(expected 'v2', 'v3.1.1', 'v3.2.1', 'v3.3.1', 'v3.3.2', "
                "'v3.3.3', 'v3.3.4', 'v3.3.5', 'v3.3.6', 'v3.3.7', "
                "'v3.3.8', 'v3.3.9', 'v3.3.3-2', 'v3.3.4-2', 'v3.3.3-3', "
                "or 'v3.3.3-4')"
            )
        self._k_topics = k_topics
        self._k_turns_per_topic = k_turns_per_topic
        self._system_prompt = system_prompt
        self._response_filter = response_filter
        self._llm_kwargs = llm_kwargs
        self._next_turn_id = 0

        # Retrieval policy (v3.3.3-4 episode rerank + dormant LTM safety).
        self._retrieval_mode = retrieval_mode
        self._episode_top_k = episode_top_k
        self._dormant_ltm_top_n = dormant_ltm_top_n
        self._rerank_w = {
            "q": rerank_query_weight,
            "topic": rerank_topic_weight,
            "episode": rerank_episode_weight,
            "pe": rerank_pe_penalty,
            "recency": rerank_recency_weight,
        }

        # Phase 2-Full STM wiring
        self._use_stm = use_stm
        self._round_size = round_size
        self._round_async = round_async
        if use_stm:
            self._stm: MemoryWindow | None = MemoryWindow(
                max_topics=stm_max_topics, max_turns=stm_max_turns
            )
            self._round_processor: RoundProcessor | None = RoundProcessor(
                conv_id=conv_id,
                ltm=self._ltm,
                stm=self._stm,
                threshold=promotion_threshold,
                alpha=importance_alpha,
                lambda_r=lambda_r,
                lambda_freq=lambda_freq,
                min_floor=min_floor,
                clear_stm_each_round=round_clear_stm,
                importance_version=importance_version,
                importance_v2_extra=importance_v2_extra or {},
                salience_provider=self._segmenter,
            )
        else:
            self._stm = None
            self._round_processor = None

    # ------------------------------------------------------------------
    # Public properties (debug / tests)
    # ------------------------------------------------------------------

    def _importance_prefill(self) -> list[dict[str, Any]]:
        """v3.3.3-4 retrieval — importance-only (CLAUDE.md 최상위 규칙, 2026-05-11).

        STM membership 자체가 importance gate (round-end promotion + eviction
        에서 이미 결정됨). Query time 에는 그대로 dump. query-aware ranking 없음.

        반환: STM 의 모든 turn 을 turn_id 오름차순으로. (topic, episode) atomicity
        는 STM atomicity (topic 단위 promote / evict) 가 자연 보장.
        """
        if self._stm is None:
            return []
        turns = list(self._stm.all_turns())
        turns.sort(key=lambda t: int(t.get("turn_id", 0)))
        return turns

    def _assign_with_episode(self, q: np.ndarray) -> tuple[int, bool, int]:
        """Wrapper around ``segmenter.assign`` that always returns
        ``(topic_id, is_boundary, episode_id)``.

        Older segmenters return only ``(topic_id, is_boundary)``; we default
        ``episode_id=0`` so downstream code can uniformly handle the field.
        Only v3.3.3-4 emits a real per-turn episode id.
        """
        out = self._segmenter.assign(q)
        if isinstance(out, tuple) and len(out) == 3:
            return int(out[0]), bool(out[1]), int(out[2])
        return int(out[0]), bool(out[1]), 0

    @property
    def stm(self) -> MemoryWindow | None:
        return self._stm

    @property
    def round_processor(self) -> RoundProcessor | None:
        return self._round_processor

    # ------------------------------------------------------------------
    # Main per-turn entry
    # ------------------------------------------------------------------

    def handle_turn(
        self, user_text: str, return_debug: bool = False
    ) -> str | tuple[str, dict[str, Any]]:
        """Process one user turn and return the assistant's response.

        With ``return_debug=True``, returns ``(response, debug)`` where
        ``debug`` includes ``topic_id``, ``is_boundary``, ``prefill_turns``,
        ``messages``, and—when ``use_stm`` is on—``stm_hit`` (bool) and
        ``round_triggered`` (bool).
        """
        # 1. embed
        q = np.asarray(self._encoder.encode([user_text])[0])

        # 2. segment (mutates segmenter state)
        topic_id, is_boundary, episode_id = self._assign_with_episode(q)

        # 3. snapshot topic state
        self._ltm.update_state(self.conv_id, self._snapshot_state())

        # 4. memory window — STM-first if enabled, else stateless baseline
        stm_hit: bool | None = None
        if self._stm is not None:
            stm_hit = self._stm.has(topic_id)
            if not stm_hit:
                ltm_turns = self._ltm.load_turns(self.conv_id, topic_id=topic_id)
                if ltm_turns:
                    self._stm.promote(topic_id, ltm_turns)
            if self._retrieval_mode in ("episode_rerank", "importance_only"):
                prefill = self._importance_prefill()
            else:
                prefill = self._stm.all_turns()
        else:
            prefill = select_memory_window(
                q, self._ltm, self.conv_id, self._k_topics, self._k_turns_per_topic
            )

        # 5. build messages
        messages: list[dict[str, Any]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend({"role": t["role"], "content": t["text"]} for t in prefill)
        messages.append({"role": "user", "content": user_text})

        # 6. LLM call (raw to caller, filtered to LTM)
        response = self._llm.chat(messages, model=self._model, **self._llm_kwargs)
        stored_response = (
            self._response_filter(response) if self._response_filter else response
        )

        # 7. persist user + assistant turns (LTM dual-write + STM working buffer)
        user_turn = self._make_turn(
            self._next_turn_id, "user", user_text, q.tolist(), topic_id,
            is_boundary, episode_id=episode_id,
        )
        self._ltm.append_turn(self.conv_id, user_turn)
        self._next_turn_id += 1
        assistant_turn = self._make_turn(
            self._next_turn_id, "assistant", stored_response, None, topic_id, False,
            episode_id=episode_id,
        )
        self._ltm.append_turn(self.conv_id, assistant_turn)
        self._next_turn_id += 1
        # Per spec: every turn lands in STM. add_turn_or_promote either
        # appends to an existing cached topic (atomicity preserved — same
        # full topic + 1 turn) or seeds a new STM entry with [turn] (new
        # topic's full history at this moment is just this turn — still
        # atomic). RoundProcessor reconciles all topics with importance at
        # round boundary.
        if self._stm is not None:
            self._stm.add_turn_or_promote(topic_id, user_turn)
            self._stm.add_turn_or_promote(topic_id, assistant_turn)

        # 8. round trigger (Phase 2-Full only)
        round_triggered = self._maybe_trigger_round()

        if return_debug:
            debug: dict[str, Any] = {
                "topic_id": topic_id,
                "is_boundary": is_boundary,
                "prefill_turns": prefill,
                "messages": messages,
            }
            if self._use_stm:
                debug["stm_hit"] = stm_hit
                debug["round_triggered"] = round_triggered
            return response, debug
        return response

    # ------------------------------------------------------------------
    # Read-only evaluation query
    # ------------------------------------------------------------------

    def eval_query(
        self, user_text: str, return_debug: bool = False
    ) -> str | tuple[str, dict[str, Any]]:
        """Read-only query: select memory window + generate response, with
        **no mutation** of segmenter state, STM, or LTM.

        Use this for benchmark evaluation when many test questions share
        a single frozen post-conversation memory state — e.g., LoCoMo,
        where each conversation has ~200 questions all asked against the
        same conversation history. Calling :meth:`handle_turn` instead
        would (a) embed every test question into the conversation history
        as if it were a real user turn, (b) cause STM miss-promotes that
        permanently change the cached topic set, and (c) require an LTM
        rebuild between questions to keep them independent — exactly the
        200× rebuild overhead this method exists to avoid.

        Semantically equivalent to a snapshot/restore around handle_turn,
        but cheaper because there's nothing to revert.

        Behaviour:

        * Segmenter assigns the question's topic via
          :meth:`HiEMSegmenter.predict_topic` (no centroid update, no
          ``prev_k`` change, no count bump).
        * STM membership unchanged. If the predicted topic is in STM:
          prefill = ``stm.all_turns()``. If not: prefill = chronological
          merge of current STM with the LTM turns of that topic — same
          *content* a miss-promote would have produced, but the STM dict
          is not modified.
        * LTM is read but not appended to.
        """
        q = np.asarray(self._encoder.encode([user_text])[0])

        if self._stm is not None:
            topic_id = self._segmenter.predict_topic(q)
            stm_hit = self._stm.has(topic_id)
            if self._retrieval_mode in ("episode_rerank", "importance_only"):
                prefill = self._importance_prefill()
            else:
                stm_turns = self._stm.all_turns()
                if stm_hit:
                    prefill = stm_turns
                else:
                    ltm_turns = self._ltm.load_turns(self.conv_id, topic_id=topic_id)
                    if ltm_turns:
                        merged = {t["turn_id"]: t for t in stm_turns}
                        for t in ltm_turns:
                            merged.setdefault(t["turn_id"], t)
                        prefill = sorted(merged.values(), key=lambda t: t["turn_id"])
                    else:
                        prefill = stm_turns
        else:
            topic_id = -1
            stm_hit = None
            prefill = select_memory_window(
                q, self._ltm, self.conv_id, self._k_topics, self._k_turns_per_topic
            )

        messages: list[dict[str, Any]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend({"role": t["role"], "content": t["text"]} for t in prefill)
        messages.append({"role": "user", "content": user_text})

        response = self._llm.chat(messages, model=self._model, **self._llm_kwargs)

        if return_debug:
            debug: dict[str, Any] = {
                "topic_id": topic_id,
                "prefill_turns": prefill,
                "messages": messages,
            }
            if self._use_stm:
                debug["stm_hit"] = stm_hit
            return response, debug
        return response

    # ------------------------------------------------------------------
    # History preload (benchmarks)
    # ------------------------------------------------------------------

    def flush_stm(self) -> None:
        """Drop all STM contents while leaving LTM, segmenter, and turn-id
        counter intact. Used at haystack-session boundaries by ``hi-em-full-v2``
        to simulate "user closes the app, opens it later" — working memory is
        gone but long-term memory and topic identity persist.
        """
        if self._stm is not None:
            self._stm.clear()

    def preload_history_sessioned(
        self,
        sessions: list[list[dict[str, Any]]],
        flush_stm_each: bool = True,
    ) -> None:
        """Preload nested haystack-sessions; flush STM at each session boundary.

        Differs from :meth:`preload_history` only in that:

        * Sessions are processed one at a time. Round processor is invoked
          ``ceil(session_pairs / round_size)`` times **per session** so the
          mention log / neighbor weights / importance scores reflect each
          session's contribution before STM is flushed.
        * After each session (incl. the last when ``flush_stm_each``):
          :meth:`flush_stm` clears the working buffer. LTM persists. The
          subsequent ``handle_turn`` will repopulate STM via the existing
          cache-miss path (LTM → STM promote on topic re-hit).

        Without ``use_stm``, this falls through to flat :meth:`preload_history`
        since there is no STM to flush.
        """
        if not self._use_stm:
            flat = [t for sess in sessions for t in sess]
            self.preload_history(flat)
            return

        for sess in sessions:
            if not sess:
                continue
            self._ingest_session_into_ltm(sess)
            n_pairs = sum(1 for t in sess if t["role"] == "user")
            n_rounds = max(1, (n_pairs + self._round_size - 1) // self._round_size)
            for _ in range(n_rounds):
                self._round_processor.process()  # type: ignore[union-attr]
            if flush_stm_each:
                self.flush_stm()
        # Spec: at session boundary, STM is flushed and then "LTM에서 importance
        # 순으로 다시 STM으로 이동시켜서 시작" (spec step 4). For mid-conversation
        # boundaries this happens at the next session's first round (clear+promote
        # under round_clear_stm=True). For the final boundary there's no next
        # round, so we run one extra round_processor.process() to repopulate STM
        # by importance — leaving the post-preload state as
        # "LTM persists, STM = high-importance topic set" rather than empty.
        # Skipped if no flush happened or STM already populated by the last in-
        # session round (which would mean flush_stm_each was False).
        if (
            flush_stm_each
            and self._stm is not None
            and not self._stm.current_topics()
        ):
            self._round_processor.process()  # type: ignore[union-attr]
        self._ltm.update_state(self.conv_id, self._snapshot_state())

    def _ingest_session_into_ltm(self, turns: list[dict[str, Any]]) -> None:
        """Mirrors :meth:`preload_history` body sans round trigger.

        Encodes user turns, segments, appends user/assistant rows to LTM,
        bumps ``_next_turn_id``. State snapshot is **not** written here; the
        caller writes one snapshot per sessioned-preload call.
        """
        user_indices = [i for i, t in enumerate(turns) if t["role"] == "user"]
        if user_indices:
            user_texts = [turns[i]["content"] for i in user_indices]
            user_embs = np.asarray(self._encoder.encode(user_texts))
        else:
            user_embs = np.empty((0, self._encoder.dim))
        emb_by_turn_idx = {idx: user_embs[k] for k, idx in enumerate(user_indices)}

        last_topic_id = 0
        last_episode_id = 0
        for i, t in enumerate(turns):
            role = t["role"]
            text = t["content"]
            ts = t.get("ts", datetime.now(timezone.utc).isoformat())
            if role == "user":
                q = emb_by_turn_idx[i]
                topic_id, is_boundary, episode_id = self._assign_with_episode(q)
                last_topic_id = topic_id
                last_episode_id = episode_id
                self._ltm.append_turn(
                    self.conv_id,
                    {
                        "turn_id": self._next_turn_id,
                        "ts": ts,
                        "role": "user",
                        "text": text,
                        "embedding": q.tolist(),
                        "topic_id": topic_id,
                        "episode_id": episode_id,
                        "is_boundary": is_boundary,
                        "dia_id": t.get("dia_id"),
                    },
                )
            else:
                self._ltm.append_turn(
                    self.conv_id,
                    {
                        "turn_id": self._next_turn_id,
                        "ts": ts,
                        "role": role,
                        "text": text,
                        "embedding": None,
                        "topic_id": last_topic_id,
                        "episode_id": last_episode_id,
                        "is_boundary": False,
                        "dia_id": t.get("dia_id"),
                    },
                )
            self._next_turn_id += 1

    def preload_history(self, turns: list[dict[str, Any]]) -> None:
        """Inject pre-existing user/assistant turns into LTM without LLM calls.

        With STM enabled, runs ``ceil(loaded_pairs / round_size)`` synchronous
        :class:`RoundProcessor` rounds at the end so the mention log,
        neighbor weights, and STM accurately reflect the preloaded history
        (bug 10 fix: previously fixed at 1 round regardless of size).
        """
        user_indices = [i for i, t in enumerate(turns) if t["role"] == "user"]
        if user_indices:
            user_texts = [turns[i]["content"] for i in user_indices]
            user_embs = np.asarray(self._encoder.encode(user_texts))
        else:
            user_embs = np.empty((0, self._encoder.dim))
        emb_by_turn_idx = {idx: user_embs[k] for k, idx in enumerate(user_indices)}

        last_topic_id = 0
        last_episode_id = 0
        for i, t in enumerate(turns):
            role = t["role"]
            text = t["content"]
            ts = t.get("ts", datetime.now(timezone.utc).isoformat())
            if role == "user":
                q = emb_by_turn_idx[i]
                topic_id, is_boundary, episode_id = self._assign_with_episode(q)
                last_topic_id = topic_id
                last_episode_id = episode_id
                self._ltm.append_turn(
                    self.conv_id,
                    {
                        "turn_id": self._next_turn_id,
                        "ts": ts,
                        "role": "user",
                        "text": text,
                        "embedding": q.tolist(),
                        "topic_id": topic_id,
                        "episode_id": episode_id,
                        "is_boundary": is_boundary,
                        "dia_id": t.get("dia_id"),
                    },
                )
            else:  # assistant — no embedding, inherit prev user's topic
                self._ltm.append_turn(
                    self.conv_id,
                    {
                        "turn_id": self._next_turn_id,
                        "ts": ts,
                        "role": role,
                        "text": text,
                        "embedding": None,
                        "topic_id": last_topic_id,
                        "is_boundary": False,
                        "dia_id": t.get("dia_id"),
                    },
                )
            self._next_turn_id += 1
        self._ltm.update_state(self.conv_id, self._snapshot_state())

        if self._round_processor is not None:
            # ceil-divide loaded_pairs by round_size; min 1 round.
            n_pairs = self._next_turn_id // 2
            n_rounds = max(1, (n_pairs + self._round_size - 1) // self._round_size)
            for _ in range(n_rounds):
                self._round_processor.process()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_trigger_round(self) -> bool:
        """Trigger ``RoundProcessor`` exactly once when ``self._next_turn_id``
        crosses the next ``2*round_size`` boundary.

        Returns True iff a round was dispatched (sync or async).
        """
        if self._round_processor is None:
            return False
        period = 2 * self._round_size
        if self._next_turn_id == 0 or self._next_turn_id % period != 0:
            return False
        if self._round_async:
            self._round_processor.process_async()
        else:
            self._round_processor.process()
        return True

    def wait_for_round(self, timeout: float | None = None) -> None:
        """Block until any in-flight async round finishes (Phase 2-Full only)."""
        if self._round_processor is not None:
            self._round_processor.wait(timeout)

    def _make_turn(
        self,
        turn_id: int,
        role: str,
        text: str,
        embedding: list[float] | None,
        topic_id: int,
        is_boundary: bool,
        episode_id: int = 0,
    ) -> dict[str, Any]:
        return {
            "turn_id": turn_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "text": text,
            "embedding": embedding,
            "topic_id": topic_id,
            "episode_id": episode_id,
            "is_boundary": is_boundary,
        }

    def _snapshot_state(self) -> dict[str, Any]:
        topics_out = []
        for t in self._segmenter.topics:
            row: dict[str, Any] = {
                "topic_id": t.topic_id,
                "centroid": t.mu.tolist(),
                "count": t.n,
            }
            # v2 Gaussian topic carries diagonal variance; v3 cosine topic
            # has no variance state. Snapshot what's available.
            variance_fn = getattr(t, "variance", None)
            if callable(variance_fn):
                row["variance"] = variance_fn().tolist()
            topics_out.append(row)
        return {
            "conv_id": self.conv_id,
            "n_turns": self._next_turn_id,
            "topics": topics_out,
        }
