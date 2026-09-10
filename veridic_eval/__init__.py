"""
veridic-eval - logs-only, offline post-hoc evaluation for the VERIDIC RAG app.

Reads the app's Postgres (message_evidence, chunks, documents, completion_logs,
rag_logs, messages) WITHOUT any app-side schema or logging change, and computes:

  * retrieval IR metrics (ranx): recall@5, mrr@5, ndcg@5, hit_rate@5  (k <= top_n)
  * generation faithfulness (LettuceDetect span-level; LangMet-RAGAS / Phoenix optional)
  * abstention (promptfoo `echo` provider: anchored regex + is-refusal net; RGB Rej)
  * cost + CFCA (reuses the app cost model / LangMet cost)
  * bootstrap BCa confidence intervals + paired cell deltas (scipy; deepsig.aso optional)

Design constraints (see methodnow-rescoped.md Part B):
  * logs-only: metrics are capped at k <= top_n (default 5) - the pre-rerank pool
    is not persisted, so no index replay and NO mass LLM/corpus re-scanning happens.
  * fully offline: the local LLM is Ollama; ALL third-party telemetry is disabled.
"""
# MUST be first: kills all third-party telemetry before anything else imports.
from . import telemetry_off as telemetry_off  # noqa: F401  (import for side effect)

__all__ = ["telemetry_off"]
__version__ = "0.1.0"
