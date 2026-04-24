"""Sticky Chinese Restaurant Process prior (SEM Eq 1).

Hi-EM uses SEM2's sticky-CRP verbatim with hyperparameters reversed
(alpha=1.0, lambda=10.0) for conversation topic persistence. See
``context/01-hi-em-design.md`` §2 and ``context/00-sem-paper.md`` §2.
"""

from __future__ import annotations

import numpy as np


def sticky_crp_unnormed(
    counts: np.ndarray,
    prev_k: int | None,
    alpha: float,
    lmda: float,
) -> np.ndarray:
    """Unnormalized sticky-CRP prior (SEM Eq 1).

    .. math::
        \\Pr(e_n = k \\mid e_{1:n-1}) \\propto
            \\begin{cases}
                C_k + \\lambda\\,\\mathbb{I}[e_{n-1}=k] & k \\leq K \\\\
                \\alpha & k = K+1
            \\end{cases}

    Args:
        counts: Prior assignment counts per cluster, shape ``(max_k,)``.
            The new-cluster slot is the first index whose count is 0.
        prev_k: Most recent assignment, or ``None`` on the first step.
        alpha: Concentration parameter (new-cluster probability).
        lmda: Stickiness parameter (bonus for staying in previous cluster).

    Returns:
        Unnormalized prior as ``float64`` array, same shape as ``counts``.
        Not normalized — the caller takes :math:`\\log` directly.
    """
    prior = counts.astype(np.float64, copy=True)
    n_visited = int(np.count_nonzero(counts))
    if n_visited < counts.shape[0]:
        prior[n_visited] += alpha
    if prev_k is not None:
        prior[prev_k] += lmda
    return prior
