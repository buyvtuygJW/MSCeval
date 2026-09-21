"""
Embeddings service - generates vector embeddings for text chunks.
"""
import asyncio
import numpy as np
import os
from typing import Any, Callable, Dict, List, Optional
# Bake-off copy: stdlib logger instead of backend.shared.utils.logging.
# Only this block differs from the app file; everything below is verbatim.
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chunk-bakeoff.embeddings")


class EmbeddingsService:
    """Service for generating text embeddings."""
    
    # Non-torch ONNX backend:
    # We use the Xenova ONNX-converted MiniLM model to avoid torch/optimum export at runtime.
    # This prevents large installs and avoids OOM crashes during export.
    MODEL_REPO = os.getenv("EMBEDDINGS_MODEL_REPO", "Xenova/all-MiniLM-L6-v2")
    ONNX_FILENAME = os.getenv("EMBEDDINGS_ONNX_FILENAME", "onnx/model_int8.onnx")
    
    def __init__(self):
        """
        ONNX Runtime embeddings (no torch):
        - Tokenize with HuggingFace `AutoTokenizer`
        - Run an ONNX encoder with `onnxruntime`
        - Mean pooling over token embeddings with attention mask
        - L2-normalization (matches common SBERT usage)
        """
        self._max_length: int = int(os.getenv("EMBEDDINGS_MAX_LENGTH", "256"))
        self._device: Optional[str] = os.getenv("EMBEDDINGS_DEVICE")  # reserved for future

        # If a bad/expired HF token is present in the environment, downloads of *public*
        # models can still fail with 401. Default to NOT using tokens unless explicitly enabled.
        self._hf_use_token = (os.getenv("HF_USE_TOKEN", "false").lower() in ("1", "true", "yes"))
        self._hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

        # Lazy-loaded heavy objects (so rag-service can start even if model download/export is slow)
        self.tokenizer = None
        self.session = None
        self._input_names: Optional[set[str]] = None
        self._load_lock = asyncio.Lock()

        logger.info(
            f"Loading ONNX embedding model: {self.MODEL_REPO} ({self.ONNX_FILENAME}, max_length={self._max_length})"
        )

    async def _ensure_loaded(self) -> None:
        """Load tokenizer + ONNX model once (lazy)."""
        if self.tokenizer is not None and self.session is not None:
            return

        async with self._load_lock:
            if self.tokenizer is not None and self.session is not None:
                return

            # Import lazily so module import is fast and failures are isolated to first use.
            from transformers import AutoTokenizer  # type: ignore
            import onnxruntime as ort  # type: ignore
            from huggingface_hub import hf_hub_download  # type: ignore

            token_arg = self._hf_token if self._hf_use_token else False

            logger.info("Downloading/loading tokenizer and ONNX weights (first run may take a while)...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_REPO, token=token_arg)

            onnx_path = hf_hub_download(
                repo_id=self.MODEL_REPO,
                filename=self.ONNX_FILENAME,
                token=token_arg,
            )

            # CPU-only runtime (no torch, no CUDA).
            providers = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(onnx_path, providers=providers)
            self._input_names = {inp.name for inp in self.session.get_inputs()}

            logger.info(f"ONNX embedding model ready (inputs={sorted(self._input_names)})")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector as list of floats
        """
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else []

    async def count_tokens(self, text: str) -> int:
        """
        Token count under the embedder's own tokenizer (special tokens included,
        NO truncation) - feeds chunks.metadata token_count / token_overflow.
        """
        await self._ensure_loaded()
        encoded = self.tokenizer(text, add_special_tokens=True, truncation=False)  # type: ignore[misc]
        return len(encoded["input_ids"])

    async def get_token_counter(
        self,
        *,
        add_special_tokens: bool = False,
        empty_is_zero: bool = True,
    ) -> Callable[[str], int]:
        """
        Synchronous token counter bound to the loaded tokenizer.

        `chunking/token_budget.py` calls a counter once per sentence inside a
        tight packing loop, where awaiting per call would serialise the whole
        build. Loading happens here, once, and the returned closure is pure CPU.

        Args:
            add_special_tokens: False by default, because the chunk builder
                reserves the [CLS]/[SEP] seats itself and sums per-unit counts.
            empty_is_zero: return 0 for empty text instead of counting the bare
                special tokens.

        Returns:
            A callable counting tokens in one string, never truncating.
        """
        await self._ensure_loaded()
        tokenizer = self.tokenizer

        def _count(text: str) -> int:
            if empty_is_zero and not text:
                return 0
            encoded = tokenizer(  # type: ignore[misc]
                text, add_special_tokens=add_special_tokens, truncation=False
            )
            return len(encoded["input_ids"])

        return _count
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch).
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        await self._ensure_loaded()

        # Tokenize to numpy tensors (onnxruntime-friendly)
        inputs = self.tokenizer(  # type: ignore[misc]
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="np",
        )

        # Build ONNX feed dict (some models don't use token_type_ids)
        feeds: Dict[str, Any] = {}
        input_names = self._input_names or set()
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            if key in inputs and key in input_names:
                feeds[key] = inputs[key]

        # Run ONNX
        # Output 0 is usually last_hidden_state (batch, seq, hidden)
        outputs = self.session.run(None, feeds)  # type: ignore[misc]
        token_embeddings = outputs[0]
        attention_mask = inputs.get("attention_mask")  # (batch, seq)

        if attention_mask is None:
            # Fallback: simple mean over seq dimension
            pooled = token_embeddings.mean(axis=1)
        else:
            mask = attention_mask[..., None].astype(token_embeddings.dtype)  # (batch, seq, 1)
            summed = (token_embeddings * mask).sum(axis=1)  # (batch, hidden)
            counts = np.clip(mask.sum(axis=1), 1e-9, None)  # (batch, 1)
            pooled = summed / counts

        # L2 normalize
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.clip(norms, 1e-9, None)

        pooled = pooled.astype(np.float32, copy=False)
        return pooled.tolist()
