"""Standalone CPU-only multilingual E5 encoder used by the project."""

from pathlib import Path
from typing import Iterable

import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_FILE = "onnx/model_O4.onnx"
DIMENSION = 384


class Encoder:
    """Encode multilingual passages and queries as normalized 384-D vectors."""

    def __init__(
        self,
        cache_dir: str | Path = ".model_cache",
        threads: int | None = None,
        offline: bool = False,
    ) -> None:
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)

        supported = {item["model"] for item in TextEmbedding.list_supported_models()}
        if MODEL_NAME not in supported:
            TextEmbedding.add_custom_model(
                model=MODEL_NAME,
                pooling=PoolingType.MEAN,
                normalization=True,
                sources=ModelSource(hf=MODEL_NAME),
                dim=DIMENSION,
                model_file=MODEL_FILE,
            )

        options = {
            "model_name": MODEL_NAME,
            "cache_dir": str(cache),
            "local_files_only": offline,
        }
        if threads is not None:
            options["threads"] = threads
        self.model = TextEmbedding(**options)

        tokenizer_file = hf_hub_download(
            repo_id=MODEL_NAME,
            filename="tokenizer.json",
            cache_dir=str(cache / "huggingface"),
            local_files_only=offline,
        )
        self.tokenizer = Tokenizer.from_file(tokenizer_file)

    def count_tokens(self, text: str) -> int:
        """Count input tokens with the encoder's exact tokenizer."""

        return len(self.tokenizer.encode(text).ids)

    def _encode(self, texts: Iterable[str], batch_size: int) -> np.ndarray:
        """Run ONNX inference and return contiguous unit-length float32 vectors."""

        values = list(texts)
        if not values:
            return np.empty((0, DIMENSION), dtype=np.float32)
        vectors = np.asarray(
            list(self.model.embed(values, batch_size=batch_size)), dtype=np.float32
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors /= np.maximum(norms, np.finfo(np.float32).eps)
        return np.ascontiguousarray(vectors)

    def encode_passages(
        self, passages: Iterable[str], batch_size: int = 16
    ) -> np.ndarray:
        """Encode document fragments with E5's required passage prefix."""

        return self._encode(
            (f"passage: {passage.strip()}" for passage in passages), batch_size
        )

    def encode_queries(
        self, queries: Iterable[str], batch_size: int = 16
    ) -> np.ndarray:
        """Encode search questions with E5's required query prefix."""

        return self._encode(
            (f"query: {query.strip()}" for query in queries), batch_size
        )
