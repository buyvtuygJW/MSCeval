"""
Configuration for veridic-eval.

Everything is env-overridable so the eval can run against the same Postgres the
app writes to, and against the local Ollama, without code changes.

The app schema has no `condition` column and we make no schema change, so a cell
is a slice of the logs. A cell says which slice in one of two ways, checked in
this order: `conversation_ids` (pinned uuids), then a created_at window.

A cell run more than once is a list of such slices, one `Condition` per repeat,
sharing a name and differing by `run_id`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


DEFAULT_RUN_ID = "r1"


@dataclass(frozen=True)
class Condition:
    """One slice of the logs: a cell, or one repeat of a cell."""
    name: str                       # a cells.py key: control, chunk_opt, qdora, combined, prebaseline
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    conversation_ids: List[str] = field(default_factory=list)
    # which repeat of the cell this window is. One sitting per run: the same
    # questions asked again, in a fresh window, scored as its own answer set and
    # averaged into the cell's per-question row.
    run_id: str = DEFAULT_RUN_ID
    note: Optional[str] = None

    def label(self) -> str:
        return self.name

    def run_label(self, *, always: bool = False, sep: str = "#") -> str:
        """``control`` alone, or ``control#r2`` once the cell carries repeats.

        Args:
            always: suffix the first run too, for a cell that has more than one.
            sep: separator between the cell name and the run id.
        """
        if not always and self.run_id == DEFAULT_RUN_ID:
            return self.name
        return f"{self.name}{sep}{self.run_id}"


@dataclass
class Settings:
    # ---- data source (the app's Postgres; read-only use) ----------------
    postgres_url: str = _env(
        "VERIDIC_EVAL_POSTGRES_URL",
        # The app's Postgres runs in Docker (container veridic-postgres-1) and
        # compose publishes 5432 on host port 6432, so the host-side default is
        # 6432. From inside the compose network use the service name instead:
        # POSTGRES_URL=postgresql://postgres:admin@postgres:5432/veridic_db
        _env("POSTGRES_URL", "postgresql://postgres:admin@localhost:6432/veridic_db"),
    )

    # ---- local LLM (Ollama) for any optional LLM-judge backend ----------
    # LettuceDetect (primary faithfulness) is a local HF model, NOT the LLM.
    # These only matter if you opt into the RAGAS-LLM or Phoenix backends.
    ollama_url: str = _env("OLLAMA_URL", "http://localhost:11434")
    ollama_judge_model: str = _env("VERIDIC_EVAL_JUDGE_MODEL", "qwen2.5:7b")

    @property
    def ollama_openai_base(self) -> str:
        """OpenAI-compatible base URL Ollama exposes (for Phoenix/RAGAS judges)."""
        return self.ollama_url.rstrip("/") + "/v1"

    # ---- retrieval eval -------------------------------------------------
    # Logs-only: only the served top_n evidence is persisted, so k is capped.
    ir_k: int = int(_env("VERIDIC_EVAL_IR_K", "5"))

    # ---- faithfulness ---------------------------------------------------
    # backend: "lettucedetect" (default, spec-pinned) | "sirg" | "ragas" | "phoenix" | "none"
    faithfulness_backend: str = _env("VERIDIC_EVAL_FAITHFULNESS", "lettucedetect")
    lettucedetect_model: str = _env(
        "VERIDIC_EVAL_LETTUCE_MODEL", "KRLabsOrg/lettucedect-base-modernbert-en-v1"
    )
    lettucedetect_language: str = _env("VERIDIC_EVAL_LETTUCE_LANG", "en")
    # Only spans with confidence >= threshold count as ungrounded (noise guard).
    lettucedetect_threshold: float = float(_env("VERIDIC_EVAL_LETTUCE_THRESHOLD", "0.5"))

    # ---- SIRG (opt-in white-box detector; arXiv 2601.03052; no official code,
    # local reimplementation in veridic_eval/sirg.py) -----------------------
    # Needs: pip install -e ".[sirg]", local HF generator weights, and a
    # TRAINED discriminator state (python -m veridic_eval.sirg train ...).
    sirg_generator_model: str = _env(
        "VERIDIC_EVAL_SIRG_GENERATOR", "Qwen/Qwen2.5-7B-Instruct"  # HF twin of qwen2.5:7b
    )
    sirg_discriminator_init: str = _env("VERIDIC_EVAL_SIRG_DISC_INIT", "roberta-base")
    # AlignScore-base .ckpt path to match the paper's init ("" = plain roberta-base)
    sirg_alignscore_ckpt: str = _env("VERIDIC_EVAL_SIRG_ALIGNSCORE_CKPT", "")
    sirg_state_path: str = _env("VERIDIC_EVAL_SIRG_STATE", "./out/sirg_state.pt")
    sirg_threshold: float = float(_env("VERIDIC_EVAL_SIRG_THRESHOLD", "0.5"))
    sirg_device: str = _env("VERIDIC_EVAL_SIRG_DEVICE", "auto")
    # "" = module default; MUST mirror the app's real serving prompt template
    sirg_prompt_template: str = _env("VERIDIC_EVAL_SIRG_PROMPT_TEMPLATE", "")

    # ---- abstention -----------------------------------------------------
    # RGB "Rej" marker (Chen et al., AAAI 2024), verbatim by default.
    abstention_marker: str = _env(
        "VERIDIC_EVAL_ABSTAIN_MARKER",
        "I can not answer the question because of the insufficient information in documents.",
    )
    # substring the RGB deterministic scorer keys on (evalue.py behaviour)
    abstention_substring: str = _env(
        "VERIDIC_EVAL_ABSTAIN_SUBSTRING", "insufficient information"
    )
    # promptfoo CLI: "npx promptfoo@latest" or an absolute path; None => pure-python fallback
    promptfoo_cmd: Optional[str] = _env("VERIDIC_EVAL_PROMPTFOO_CMD", "npx -y promptfoo@latest") or None

    # ---- statistics -----------------------------------------------------
    bootstrap_resamples: int = int(_env("VERIDIC_EVAL_BOOTSTRAP_N", "9999"))
    confidence_level: float = float(_env("VERIDIC_EVAL_CONF", "0.95"))
    random_state: int = int(_env("VERIDIC_EVAL_SEED", "42"))
    enable_deepsig: bool = _env("VERIDIC_EVAL_DEEPSIG", "0") in {"1", "true", "yes"}

    # ---- io -------------------------------------------------------------
    benchmark_path: str = _env("VERIDIC_EVAL_BENCHMARK", "./benchmark.yaml")
    output_dir: str = _env("VERIDIC_EVAL_OUT", "./out")

    def as_pricing_basis(self) -> Dict:
        return {
            "ir_k": self.ir_k,
            "faithfulness_backend": self.faithfulness_backend,
            "abstention_marker": self.abstention_marker,
            "seed": self.random_state,
        }


settings = Settings()
