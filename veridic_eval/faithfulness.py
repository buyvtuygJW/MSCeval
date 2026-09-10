"""
Faithfulness scoring - does the answer stay true to its retrieved evidence?

Primary backend (spec-pinned): **LettuceDetect** - a peer-reviewed
black-box span-level hallucination detector (arXiv:2502.17125). It is a local HF
model (NOT the LLM), runs on outputs that already exist, and marks which spans
are ungrounded - the granularity citation auditing needs.

Optional backends (opt-in, LLM-judge, pointed at local **Ollama**):
  * "ragas"   - RAGAS faithfulness via the local Ollama model
  * "phoenix" - arize-phoenix HallucinationEvaluator via local Ollama
Both conflict slightly with the methodology's deterministic stance, so they are
never the default; they exist because the task asked to reuse Phoenix/LangMet.

Whatever backend runs, **LangMet** (`compute_raga_metrics`) additionally
aggregates one `RagaEvaluationEvent` per query into the LLM-eval summary the
report renders: faithfulness from the backend above, context precision/recall
from the served evidence against the gold chunk ids, context relevancy from the
app's own logged rerank/retrieval scores, and answer correctness/similarity
against the benchmark's `gold_answer` (see `raga_fields`).

All telemetry for HF / Phoenix / RAGAS is already disabled at import time by
``veridic_eval.telemetry_off``.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from .config import settings
from .extract import QueryRecord, render_served_context

_detector = None  # cached LettuceDetect model


def _scorable(rec: QueryRecord) -> bool:
    """Faithfulness is defined only for answerable, answered, evidence-backed items."""
    return bool(rec.linked and rec.answerable and rec.has_evidence and rec.answer_text.strip())


# ---------------------------------------------------------------- LettuceDetect
#: The window lettucedetect builds with when no caller names one
#: (``TransformerDetector.__init__``). The shipped ModernBERT checkpoint takes 8192, so the
#: library's own default throws away half the context before scoring starts.
LETTUCE_DEFAULT_WINDOW = 4096

#: Transformers writes int(1e30) into ``model_max_length`` when a checkpoint
#: declares no limit. Anything at or above this is a sentinel, not a ceiling.
_TOKENIZER_NO_LIMIT = 10 ** 7

#: The transformers flag behind the "Token indices sequence length is longer
#: than the specified maximum" line (``_eventual_warn_about_too_long_sequence``).
_LENGTH_WARNING_KEY = "sequence-length-is-longer-than-the-specified-maximum"


def mute_token_length_warning(
    detector,
    *,
    enable: bool = True,
    key: str = _LENGTH_WARNING_KEY,
    tokenizer=None,
) -> bool:
    """Silence a warning a token count raises, which no batch fed to the model earns.

    ``_eventual_warn_about_too_long_sequence`` warns on any encode call that omits
    ``max_length``, and lettucedetect measures the whole prompt that way before
    it decides how to split it (``_group_passages_into_chunks``). The batch that
    reaches the model is a different call carrying ``max_length`` and
    ``truncation="only_first"`` (``prepare_tokenized_input``), so the
    "will result in indexing errors" claim never describes a sequence this pass
    runs. Real truncation is counted per query by `lettuce_window_read`.

    Args:
      detector: ``HallucinationDetector`` or a bare transformer detector.
      enable: True mutes the line; False re-arms it for one more emission.
      key: the transformers flag name, should the library rename it.
      tokenizer: mute this tokenizer instead of the detector's.

    Returns True when the flag was reachable. Transformers emits the message
    once per tokenizer instance, so one call covers a whole run.
    """
    tok = tokenizer
    if tok is None:
        tok = getattr(getattr(detector, "detector", detector), "tokenizer", None)
    flags = getattr(tok, "deprecation_warnings", None)
    if not isinstance(flags, dict):
        return False
    flags[key] = bool(enable)
    return True


def detector_context_ceiling(
    detector,
    *,
    hard_cap: Optional[int] = None,
    floor: int = LETTUCE_DEFAULT_WINDOW,
) -> Dict:
    """The longest prompt this checkpoint can read, measured off the loaded model.

    Takes the smaller of the two ceilings the checkpoint carries: the encoder's
    position budget (``config.max_position_embeddings``) and the tokenizer's
    declared limit (``model_max_length``, the number behind the "Token indices
    sequence length is longer than the specified maximum" warning). Neither is
    a knob the caller types, so a cell can never be scored at a window the
    weights do not support.

    Args:
      detector: ``HallucinationDetector`` or a bare transformer detector.
      hard_cap: ceiling the caller will not cross whatever the model claims,
        for when the full window does not fit in VRAM. None takes the model's.
      floor: returned when the model exposes no readable ceiling.

    Returns ``{"ceiling", "position_budget", "tokenizer_limit", "source"}``,
    where ``source`` is ``"model"``, ``"hard_cap"`` or ``"floor"``.
    """
    inner = getattr(detector, "detector", detector)
    config = getattr(getattr(inner, "model", None), "config", None)
    pos = getattr(config, "max_position_embeddings", None)
    tok_limit = getattr(getattr(inner, "tokenizer", None), "model_max_length", None)
    read = [
        int(v)
        for v in (pos, tok_limit)
        if isinstance(v, int) and not isinstance(v, bool) and 0 < v < _TOKENIZER_NO_LIMIT
    ]
    ceiling, source = (min(read), "model") if read else (int(floor), "floor")
    if hard_cap is not None and int(hard_cap) < ceiling:
        ceiling, source = int(hard_cap), "hard_cap"
    return {
        "ceiling": ceiling,
        "position_budget": pos,
        "tokenizer_limit": tok_limit,
        "source": source,
    }


def build_detector(
    *,
    model_path: Optional[str] = None,
    lang: Optional[str] = None,
    method: str = "transformer",
    window: Optional[int] = None,
    hard_cap: Optional[int] = None,
    mute_length_warning: bool = True,
    detector_kwargs: Optional[Dict[str, Any]] = None,
):
    """A detector whose prompt window is the checkpoint's ceiling, not the 4096 default.

    lettucedetect constructs at ``max_length=4096`` and then truncates longer
    prompts one-sidedly, cutting the context and keeping the answer
    (``prepare_tokenized_input``). A cell whose served context runs long is
    therefore judged against evidence the model never read, while a cell that
    fits keeps all of its own, so the faithfulness delta between them is partly
    a difference in how much context each side lost. Reading the window off the
    checkpoint lifts it to 8192 on ``lettucedect-base-modernbert-en-v1``, which
    clears every prompt in control, chunk_opt, qdora, combined and prebaseline2
    (4989 tokens at the longest, against 9 of 9 truncated at 4096 in three of
    them). It does not clear prebaseline, whose 14356, 9973, 8578 and 8570
    token prompts still lose their tail, so 4 of 9 answers there stay scored on
    a cut context and prebaseline is the one cell whose faithfulness delta
    still carries a truncation term.

    ``max_length`` is read at predict time only (``_group_passages_into_chunks``,
    ``_predict_single``, ``_build_spans_from_tokens``, ``predict_prompt``), so the
    window applied here governs every later call.

    Args:
      model_path / lang: None reads ``settings.lettucedetect_model`` and
        ``settings.lettucedetect_language``.
      method: lettucedetect backend; only ``"transformer"`` carries a window.
      window: exact prompt window in tokens, skipping the measurement. None
        measures the checkpoint.
      hard_cap: upper bound on the measured ceiling. Ignored when ``window`` is
        given.
      mute_length_warning: drops the transformers line that lettucedetect's own
        prompt count raises; see `mute_token_length_warning`. False keeps it.
      detector_kwargs: extra constructor arguments (``device``,
        ``taxonomy_head``, tokenizer kwargs).

    Returns the detector. The applied window sits on
    ``detector.detector.max_length`` and is reported per query by
    `lettuce_window_read`.
    """
    from lettucedetect.models.inference import HallucinationDetector  # lazy

    det = HallucinationDetector(
        method=method,
        model_path=model_path or settings.lettucedetect_model,
        lang=lang or settings.lettucedetect_language,
        **(detector_kwargs or {}),
    )
    inner = getattr(det, "detector", det)
    if hasattr(inner, "max_length"):
        inner.max_length = (
            int(window)
            if window
            else detector_context_ceiling(det, hard_cap=hard_cap)["ceiling"]
        )
    if mute_length_warning:
        mute_token_length_warning(det)
    return det


def _get_detector():
    global _detector
    if _detector is None:
        _detector = build_detector()
    return _detector


#: Every span the transformer head emits already won an argmax over two classes,
#: so its confidence cannot sit below 0.5 (``_predict_single`` sets it to the
#: softmax probability of class 1). A threshold at or under this floor filters
#: nothing, whatever `Settings.lettucedetect_threshold` says about noise.
LETTUCE_ARGMAX_FLOOR = 0.5


def lettuce_window_read(
    detector,
    context: Sequence[str],
    question: str,
    answer: str,
    *,
    window: Optional[int] = None,
) -> Dict:
    """How many windows the detector will run for this answer, and which overflow.

    Groups and formats with lettucedetect's own budgeting rather than an
    estimate: ``_group_passages_into_chunks`` then ``PromptUtils.format_context``
    (``predict``), measured with the same ``tokenizer(prompt, answer)``
    length the library warns on (``predict_prompt``). Truncation inside
    ``predict()`` is silent and one-sided: ``truncation="only_first"`` cuts the
    context and keeps the answer (``prepare_tokenized_input``), so an overflow
    means the answer was scored against evidence the model never read.

    Args:
      detector: ``HallucinationDetector`` or a bare detector.
      context: the passage list exactly as ``predict()`` will receive it.
      question / answer: the same strings handed to ``predict()``.
      window: prompt window in tokens; None reads the detector's ``max_length``.

    Returns ``{"window", "n_windows", "prompt_tokens", "truncated"}``, where
    ``prompt_tokens`` is the largest window's token count and ``truncated``
    counts the windows over budget. Values are None when the detector exposes no
    tokenizer, and a failed read adds ``read_error`` instead of raising.
    """
    inner = getattr(detector, "detector", detector)
    tok = getattr(inner, "tokenizer", None)
    win = int(window or getattr(inner, "max_length", 0) or 0) or None
    blank = {"window": win, "n_windows": None, "prompt_tokens": None, "truncated": None}
    if tok is None or win is None:
        return blank
    try:
        from lettucedetect.detectors.prompt_utils import PromptUtils

        lang = getattr(inner, "lang", settings.lettucedetect_language)
        groups = [list(context)]
        if hasattr(inner, "_group_passages_into_chunks"):
            groups = inner._group_passages_into_chunks(list(context), question, answer)
        totals = [
            len(tok(PromptUtils.format_context(g, question, lang), answer,
                    add_special_tokens=True, verbose=False)["input_ids"])
            for g in groups
        ]
    except Exception as exc:  # a diagnostic read never stops the scoring pass
        return {**blank, "read_error": f"{type(exc).__name__}: {exc}"}
    return {
        "window": win,
        "n_windows": len(groups),
        "prompt_tokens": max(totals) if totals else 0,
        "truncated": sum(1 for t in totals if t > win),
    }


def score_lettucedetect(
    rec: QueryRecord,
    *,
    detector=None,
    threshold: Optional[float] = None,
    passages: str = "per_chunk",
    passage_sep: str = "\n\n",
    faithful_max_spans: int = 0,
    faithful_max_char_frac: Optional[float] = None,
    window: Optional[int] = None,
    with_spans: bool = True,
    with_window_read: bool = True,
    context_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict:
    """One answer's span-level faithfulness, beside what the detector actually saw.

    Args:
      detector: live ``HallucinationDetector``; None reuses the cached one.
      threshold: span confidence floor; None reads
        ``settings.lettucedetect_threshold``. Nothing under
        ``LETTUCE_ARGMAX_FLOOR`` can drop a span.
      passages: ``"per_chunk"`` hands one passage per served chunk, which is
        what ships. lettucedetect budgets at the passage level
        (``_group_passages_into_chunks``): separate chunks are packed into as many windows as
        the context needs and spans are aggregated with ``max()``, so a token
        stays flagged unless EVERY window supports it (``_predict_chunked``),
        and no answer is scored against a cut context. ``"joined"`` hands the
        whole context over as ONE passage, which the library cannot split, so
        everything past the window is dropped instead. A single chunk longer
        than the window is cut either way; `lettuce_window_read` counts it.
      passage_sep: what joins the chunk blocks when ``passages="joined"``;
        matches ``QueryRecord.context_text``.
      faithful_max_spans: confident spans an answer may carry and still count
        faithful. 0 is the shipped all-or-nothing rule.
      faithful_max_char_frac: tolerance on the flagged answer fraction, ANDed
        with the span count. None leaves the count as the only rule.
      window: passed to `lettuce_window_read`.
      with_spans: keep every flagged span (start, end, confidence, text).
      with_window_read: measure the prompts the detector will build. False skips
        that tokenizer pass and reports no truncation state.
      context_kwargs: forwarded to `render_served_context`, which frames the
        served chunks with the document header and page line the app put in
        front of the model (``context_assembler._build_context_string``). The judge reads the
        context the answer was written against, so a cited document name and
        page are supported text and not an ungrounded span.

    Returns the scored keys (``faithfulness``, ``faithful``,
    ``ungrounded_spans``, ``halluc_char_frac``) beside the observed state that
    decides whether they mean anything: ``n_served``, ``context_chars`` (the
    framed context the detector read), ``chunk_chars`` (the bare chunk text
    inside it), ``answer_chars``, ``spans``, and `lettuce_window_read`'s window
    block.
    """
    if passages not in ("joined", "per_chunk"):
        raise ValueError("passages must be 'joined' or 'per_chunk'")
    det = detector if detector is not None else _get_detector()
    tau = settings.lettucedetect_threshold if threshold is None else threshold
    chunks = [e.chunk_text for e in rec.served_evidence if e.chunk_text]
    context = render_served_context(
        rec.served_evidence,
        passages=passages,
        passage_sep=passage_sep,
        **(context_kwargs or {}),
    )

    spans = det.predict(
        context=context,
        question=rec.question,
        answer=rec.answer_text,
        output_format="spans",
    ) or []
    strong = [s for s in spans if float(s.get("confidence", 1.0)) >= tau]
    answer_len = max(len(rec.answer_text), 1)
    flagged = sum(max(0, int(s.get("end", 0)) - int(s.get("start", 0))) for s in strong)
    frac = flagged / answer_len
    faithful = len(strong) <= faithful_max_spans
    if faithful_max_char_frac is not None:
        faithful = faithful and frac <= faithful_max_char_frac

    out = {
        "faithfulness": round(max(0.0, 1.0 - frac), 4),
        "faithful": int(faithful),
        "ungrounded_spans": len(strong),
        "halluc_char_frac": round(frac, 4),
        "n_served": len(chunks),
        "context_chars": sum(len(c) for c in context),
        "chunk_chars": sum(len(c) for c in chunks),
        "answer_chars": len(rec.answer_text),
    }
    if with_spans:
        out["spans"] = [
            {
                "start": int(s.get("start", 0)),
                "end": int(s.get("end", 0)),
                "confidence": round(float(s.get("confidence", 1.0)), 4),
                "text": s.get("text") or rec.answer_text[int(s.get("start", 0)):int(s.get("end", 0))],
            }
            for s in strong
        ]
    if with_window_read:
        out.update(
            lettuce_window_read(det, context, rec.question, rec.answer_text, window=window)
        )
    return out


# ------------------------------------------------------------------------ SIRG
_sirg = None  # cached SIRGDetector (white-box candidate; arXiv 2601.03052)


def _get_sirg():
    global _sirg
    if _sirg is None:
        from .sirg import SIRGDetector  # lazy; needs pip install -e ".[sirg]"

        kwargs = {}
        if settings.sirg_prompt_template:
            kwargs["prompt_template"] = settings.sirg_prompt_template
        _sirg = SIRGDetector(
            settings.sirg_generator_model,
            discriminator_init=settings.sirg_discriminator_init,
            alignscore_ckpt=settings.sirg_alignscore_ckpt or None,
            state_path=settings.sirg_state_path,
            threshold=settings.sirg_threshold,
            device=settings.sirg_device,
            **kwargs,
        )
    return _sirg


def _score_sirg(rec: QueryRecord) -> Dict:
    """Same contract as `score_lettucedetect`; spans are flagged answer sentences."""
    spans = _get_sirg().predict(
        question=rec.question, context=rec.context_text, answer=rec.answer_text
    )["spans"]  # predict() already applies the calibrated threshold
    answer_len = max(len(rec.answer_text), 1)
    flagged = sum(max(0, int(s["end"]) - int(s["start"])) for s in spans)
    frac = flagged / answer_len
    return {
        "faithfulness": round(max(0.0, 1.0 - frac), 4),
        "faithful": int(len(spans) == 0),
        "ungrounded_spans": len(spans),
        "halluc_char_frac": round(frac, 4),
    }


# ------------------------------------------------------------------- RAGAS/LLM
def _score_ragas(records: List[QueryRecord]) -> Dict[str, Dict]:
    from langchain_openai import ChatOpenAI  # lazy
    from ragas import evaluate as ragas_evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.metrics import faithfulness as ragas_faithfulness

    llm = ChatOpenAI(
        model=settings.ollama_judge_model,
        base_url=settings.ollama_openai_base,
        api_key="ollama",
        temperature=0.0,
    )
    samples = [
        {
            "user_input": r.question,
            "response": r.answer_text,
            "retrieved_contexts": render_served_context(
                r.served_evidence, passages="per_chunk"
            ),
        }
        for r in records
    ]
    ds = EvaluationDataset.from_list(samples)
    result = ragas_evaluate(ds, metrics=[ragas_faithfulness], llm=llm)
    df = result.to_pandas()
    out: Dict[str, Dict] = {}
    for r, (_, row) in zip(records, df.iterrows()):
        val = float(row.get("faithfulness", 0.0) or 0.0)
        out[r.query_id] = {"faithfulness": round(val, 4), "faithful": int(val >= 0.5),
                           "ungrounded_spans": None}
    return out


def _score_phoenix(records: List[QueryRecord]) -> Dict[str, Dict]:
    import pandas as pd  # lazy
    from phoenix.evals import HallucinationEvaluator, OpenAIModel, llm_classify

    model = OpenAIModel(
        model=settings.ollama_judge_model,
        base_url=settings.ollama_openai_base,
        api_key="ollama",
        temperature=0.0,
    )
    evaluator = HallucinationEvaluator(model)
    df = pd.DataFrame(
        {
            "input": [r.question for r in records],
            "output": [r.answer_text for r in records],
            "reference": [r.context_text for r in records],
        }
    )
    res = llm_classify(dataframe=df, model=model, template=evaluator.template,
                       rails=list(evaluator.default_concurrency and evaluator.rails or ["factual", "hallucinated"]))
    out: Dict[str, Dict] = {}
    for r, (_, row) in zip(records, res.iterrows()):
        faithful = int(str(row.get("label", "")).lower() == "factual")
        out[r.query_id] = {"faithfulness": float(faithful), "faithful": faithful,
                           "ungrounded_spans": None}
    return out


# --------------------------------------------------------------------- driver
def score_faithfulness(records: List[QueryRecord], backend: Optional[str] = None) -> Dict:
    """
    Returns::
        {
          "backend": "lettucedetect",
          "n_scored": 10,
          "aggregate": {"faithfulness": .., "faithful_rate": ..,
                        "total_ungrounded_spans": .., "n_answers_truncated": ..},
          "per_query": {qid: score_lettucedetect(rec)},   # spans + window read
          "langmet_summary": {...},   # LangMet compute_raga_metrics, all 7 fields
        }
    """
    backend = (backend or settings.faithfulness_backend).lower()
    scorable = [r for r in records if _scorable(r)]

    if backend == "none" or not scorable:
        # The context and gold-answer fields do not need a hallucination backend,
        # so the RAGA summary still carries real numbers here.
        return {"backend": backend, "n_scored": 0, "aggregate": {}, "per_query": {},
                "langmet_summary": langmet_raga_summary(records, {})}

    per_query: Dict[str, Dict] = {}
    if backend == "lettucedetect":
        for rec in scorable:
            per_query[rec.query_id] = score_lettucedetect(rec)
    elif backend == "sirg":
        for rec in scorable:
            per_query[rec.query_id] = _score_sirg(rec)
    elif backend == "ragas":
        per_query = _score_ragas(scorable)
    elif backend == "phoenix":
        per_query = _score_phoenix(scorable)
    else:
        raise ValueError(f"Unknown faithfulness backend: {backend!r}")

    scores = [v["faithfulness"] for v in per_query.values()]
    faithful_flags = [v["faithful"] for v in per_query.values()]
    spans = [v["ungrounded_spans"] for v in per_query.values() if v.get("ungrounded_spans") is not None]
    # An answer whose evidence overflowed the detector window was scored against
    # context the model never read, so this count qualifies the two rates above it.
    truncated = [v["truncated"] for v in per_query.values() if v.get("truncated") is not None]
    aggregate = {
        "faithfulness": round(sum(scores) / len(scores), 4) if scores else None,
        "faithful_rate": round(sum(faithful_flags) / len(faithful_flags), 4) if faithful_flags else None,
        "total_ungrounded_spans": sum(spans) if spans else None,
        "n_answers_truncated": sum(1 for t in truncated if t) if truncated else None,
    }

    return {
        "backend": backend,
        "n_scored": len(per_query),
        "aggregate": aggregate,
        "per_query": per_query,
        "langmet_summary": langmet_raga_summary(records, per_query),
    }


# --------------------------------------------------- LangMet RAGA field mapping
_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: LangMet's `RagaEvaluationEvent` metric fields, in its own order.
RAGA_FIELDS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "context_relevancy",
    "answer_correctness",
    "answer_similarity",
)

#: `ServedEvidence` attributes tried per chunk for `context_relevancy`. Both are
#: the app's own numbers: `message_evidence.rerank_score` is the reranker's
#: query-chunk relevance, `retrieval_score` the embedder's, and the same pair is
#: logged as `rag_logs.rerank_scores` / `rag_logs.retrieval_scores`.
RELEVANCE_FIELDS = ("rerank_score", "retrieval_score")


def _tokens(text: Optional[str]) -> List[str]:
    return _TOKEN_RE.findall((text or "").casefold())


def raga_fields(
    rec: QueryRecord,
    *,
    relevance_fields: Sequence[str] = RELEVANCE_FIELDS,
    k: Optional[int] = None,
    rank_field: str = "served_rank",
    ndigits: int = 4,
) -> Dict[str, Optional[float]]:
    """
    The six non-faithfulness RAGA fields for one query, from what the app already
    logged plus the benchmark gold. A field is None when its source is absent;
    LangMet drops None from the averages and counts it in `evaluation_counts`,
    so a missing source stays visible instead of scoring as a zero.

      context_precision   served-order average precision over `judged_chunk_ids`
      context_recall      share of `judged_chunk_ids` the served evidence covers
      context_relevancy   mean logged relevance of the served chunks
      answer_correctness  token-F1 of `answer_text` against `gold_answer`
      answer_similarity   token cosine of `answer_text` against `gold_answer`
      answer_relevancy    always None: no answer-vs-question score is logged and
                          no judge runs by default

    Args:
        relevance_fields: `ServedEvidence` attributes tried in order per chunk;
            the first non-None one is that chunk's relevance.
        k: cap on served evidence considered (None = every served chunk, which
            is already the logged top_n).
        rank_field: `ServedEvidence` attribute holding served order; the same
            one `retrieval_eval` ranks by, so both modules score one ordering.
        ndigits: rounding of every returned value.
    """
    ordered = sorted(rec.served_evidence, key=lambda e: getattr(e, rank_field))
    served = ordered[:k] if k else ordered
    judged = set(rec.judged_chunk_ids)

    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    if judged:
        hits = 0
        ap = 0.0
        for rank, e in enumerate(served, start=1):
            if e.chunk_id in judged:
                hits += 1
                ap += hits / rank
        context_precision = round(ap / hits, ndigits) if hits else 0.0
        covered = len({e.chunk_id for e in served} & judged)
        context_recall = round(covered / len(judged), ndigits)

    relevances: List[float] = []
    for e in served:
        for name in relevance_fields:
            value = getattr(e, name, None)
            if value is not None:
                relevances.append(float(value))
                break
    context_relevancy = (
        round(sum(relevances) / len(relevances), ndigits) if relevances else None
    )

    answer_correctness: Optional[float] = None
    answer_similarity: Optional[float] = None
    if (rec.gold_answer or "").strip():
        pred, gold = _tokens(rec.answer_text), _tokens(rec.gold_answer)
        overlap = sum((Counter(pred) & Counter(gold)).values())
        if pred and gold and overlap:
            precision, recall = overlap / len(pred), overlap / len(gold)
            answer_correctness = round(
                2 * precision * recall / (precision + recall), ndigits
            )
            shared = len(set(pred) & set(gold))
            answer_similarity = round(
                shared / math.sqrt(len(set(pred)) * len(set(gold))), ndigits
            )
        else:
            answer_correctness = answer_similarity = 0.0

    return {
        "answer_relevancy": None,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "context_relevancy": context_relevancy,
        "answer_correctness": answer_correctness,
        "answer_similarity": answer_similarity,
    }


def langmet_raga_summary(
    records: Sequence[QueryRecord],
    per_query: Dict[str, Dict],
    *,
    linked_only: bool = True,
    relevance_fields: Sequence[str] = RELEVANCE_FIELDS,
    k: Optional[int] = None,
    rank_field: str = "served_rank",
    now: Optional[datetime] = None,
) -> Dict:
    """
    LangMet `compute_raga_metrics` over one `RagaEvaluationEvent` per query:
    faithfulness from the backend that just ran, the other six from
    `raga_fields`. Returns LangMet's dict (period / overview / scores /
    evaluation_counts), or ``{"error": ...}`` so a failure prints in the report
    instead of vanishing into an empty block.

    Args:
        linked_only: False also emits events for queries this cell never
            answered, whose fields are all None.
        relevance_fields / k / rank_field: passed through to `raga_fields`.
        now: `created_at` stamped on every event; defaults to call time.
    """
    try:
        from langmet.analytics import compute_raga_metrics
        from langmet.models import RagaEvaluationEvent
    except Exception as exc:
        return {"error": f"langmet unavailable: {type(exc).__name__}: {exc}"}

    stamp = now or datetime.now(timezone.utc)
    scored = [r for r in records if r.linked or not linked_only]
    try:
        events = [
            RagaEvaluationEvent(
                query_id=rec.query_id,
                faithfulness=(per_query.get(rec.query_id) or {}).get("faithfulness"),
                created_at=stamp,
                **raga_fields(
                    rec,
                    relevance_fields=relevance_fields,
                    k=k,
                    rank_field=rank_field,
                ),
            )
            for rec in scored
        ]
        return compute_raga_metrics(events)
    except Exception as exc:
        return {"error": f"compute_raga_metrics failed: {type(exc).__name__}: {exc}"}
