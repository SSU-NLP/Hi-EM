"""bge-base-en-v1.5 wrapper — L2-normalized query embeddings.

Scene vector ``s_n = normalize(encoder(query_n))`` per
``context/01-hi-em-design.md`` §1 and ``02-math-model.md`` §쿼리 임베딩.

``sentence_transformers`` and ``torch`` are imported lazily inside
:class:`QueryEncoder.__init__` so that this module can be imported
(for type-checking, tests, etc.) without triggering model download or
GPU initialization.
"""

from __future__ import annotations

import numpy as np

BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_DIM = 768


class QueryEncoder:
    """L2-normalized bge embeddings for Hi-EM scene vectors.

    Args:
        device: ``"cuda"`` / ``"cpu"`` or ``None`` (auto-detect).
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
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = BGE_DIM

    def encode(self, text: str | list[str]) -> np.ndarray:
        """Encode one or many strings to L2-normalized vectors.

        Args:
            text: A single string or a list of strings.

        Returns:
            If ``text`` is ``str`` → shape ``(dim,)``.
            If ``text`` is ``list[str]`` → shape ``(n, dim)``.
        """
        if isinstance(text, str):
            out = self._model.encode([text], normalize_embeddings=True)
            return np.asarray(out[0])
        return np.asarray(self._model.encode(text, normalize_embeddings=True))
