"""bge-base-en-v1.5 wrapper — L2-normalized query embeddings.

Scene vector ``s_n = normalize(encoder(query_n))`` per
``context/01-hi-em-design.md`` §1 and ``02-math-model.md`` §쿼리 임베딩.

``sentence_transformers`` and ``torch`` are imported lazily inside
:class:`QueryEncoder.__init__` so that this module can be imported
(for type-checking, tests, etc.) without triggering model download or
GPU initialization.

Thread safety: PyTorch MPS context is process-singleton; concurrent
``encode()`` from multiple threads crashes the process. The encoder
serializes its forward pass via an internal lock, so callers (Phase 4
ThreadPool, RAG, Hi-EM preload/handle_turn) are safe out of the box.
"""

from __future__ import annotations

import threading

import numpy as np

BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_DIM = 768


class QueryEncoder:
    """L2-normalized bge embeddings for Hi-EM scene vectors.

    Args:
        device: ``"cuda"`` / ``"mps"`` / ``"cpu"`` or ``None`` (auto-detect).
            Auto-detect priority: cuda → mps (Apple Silicon) → cpu.
            The same ``bge-base-en-v1.5`` weights load on every backend; only
            inference dispatch changes, so embeddings stay numerically
            consistent across environments (≈1e-5 jitter from different
            kernels).
        model_name: ``sentence-transformers`` model id.
    """

    def __init__(
        self,
        device: str | None = None,
        model_name: str = BGE_MODEL_NAME,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            ):
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = BGE_DIM
        self._lock = threading.Lock()

    def encode(self, text: str | list[str]) -> np.ndarray:
        """Encode one or many strings to L2-normalized vectors.

        Thread-safe (serialized via an internal lock — see module docstring).

        Args:
            text: A single string or a list of strings.

        Returns:
            If ``text`` is ``str`` → shape ``(dim,)``.
            If ``text`` is ``list[str]`` → shape ``(n, dim)``.
        """
        with self._lock:
            if isinstance(text, str):
                out = self._model.encode([text], normalize_embeddings=True)
                return np.asarray(out[0])
            return np.asarray(self._model.encode(text, normalize_embeddings=True))
