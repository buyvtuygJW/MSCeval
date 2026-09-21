"""
Second embedder: EmbeddingGemma-300m ONNX (onnx-community/embeddinggemma-300m-ONNX).

Same call shape as the app's EmbeddingsService so run.py treats both alike:
async generate_embeddings, get_token_counter, _max_length. Differences the
model card mandates: retrieval prompts prepended per input kind, 2048-token
window, mean-pool + L2 over a Gemma3TextModel.
"""

from __future__ import annotations

import asyncio
from typing import Callable, List

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

_DTYPE_FILES = {
    "fp32": "model.onnx",
    "fp16": "model_fp16.onnx",
    "quantized": "model_quantized.onnx",  # int8
    "q4": "model_q4.onnx",
    "q4f16": "model_q4f16.onnx",
}


class GemmaEmbeddingsService:
    REPO = "onnx-community/embeddinggemma-300m-ONNX"
    QUERY_PREFIX = "task: search result | query: "      # model card, Prompt Instructions
    DOCUMENT_PREFIX = "title: none | text: "

    def __init__(self, dtype: str = "quantized", max_length: int = 2048, batch_size: int = 8):
        onnx_file = _DTYPE_FILES[dtype]
        local = snapshot_download(
            self.REPO,
            allow_patterns=[
                "config.json", "tokenizer*", "special_tokens_map.json",
                "added_tokens.json", f"onnx/{onnx_file}*",  # * pulls the .onnx_data companion
            ],
        )
        self._tokenizer = AutoTokenizer.from_pretrained(local)
        self._session = ort.InferenceSession(f"{local}/onnx/{onnx_file}")
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._max_length = max_length
        self._batch_size = batch_size

    async def get_token_counter(self) -> Callable[[str], int]:
        tok = self._tokenizer
        return lambda text: len(tok(text, add_special_tokens=False)["input_ids"])

    async def generate_embeddings(self, texts: List[str], kind: str = "document") -> List[List[float]]:
        prefix = self.QUERY_PREFIX if kind == "query" else self.DOCUMENT_PREFIX
        texts = [prefix + (t or "") for t in texts]
        out: List[List[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start:start + self._batch_size]
            out.extend(await asyncio.to_thread(self._embed_batch, batch))
        return out

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        enc = self._tokenizer(
            batch, padding=True, truncation=True,
            max_length=self._max_length, return_tensors="np",
        )
        ids = enc["input_ids"].astype(np.int64)
        mask = enc["attention_mask"].astype(np.int64)
        feeds = {}
        if "input_ids" in self._input_names:
            feeds["input_ids"] = ids
        if "attention_mask" in self._input_names:
            feeds["attention_mask"] = mask
        if "position_ids" in self._input_names:
            feeds["position_ids"] = np.tile(np.arange(ids.shape[1], dtype=np.int64), (ids.shape[0], 1))
        hidden = self._session.run(None, feeds)[0]
        if hidden.ndim == 3:  # last_hidden_state -> masked mean pool
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        else:  # export already pools to (batch, dim)
            pooled = hidden
        pooled = pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
        return pooled.astype(np.float32).tolist()
