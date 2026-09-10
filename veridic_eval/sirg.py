"""
SIRG - Semantic-level Internal Reasoning Graph faithfulness detector.
(Hu, Li, Zhong, Qi and Zou, 2026, arXiv:2601.03052)

SIRG has NO official code release (checked 2026-07-28), so this module is a
from-paper reimplementation: pipeline from the published method, training and
tooling settings from its Appendix D (Implementation Details):

  1. LRP attribution (`lxt`, paper uses LXT 2.0): the logged answer is
     teacher-forced through the OPEN-WEIGHT LLM; each generated token's
     relevance to every prompt token is collected (one backward per token).
  2. Semantic lift: substantive words (nouns, noun phrases, named entities,
     negations; spaCy plus Stanza in the paper, Stanza optional here) become
     graph nodes and token
     relevance is pooled per node. Per-sentence attribution rows over those
     nodes form the internal reasoning graph.
  3. Discriminator: a small PLM (default "roberta-base"; the paper initialises
     from the AlignScore RoBERTa 124M checkpoint, i.e. AlignScore-base.ckpt,
     downloaded by hand from https://huggingface.co/yzha/AlignScore (code at
     github.com/yuh-zha/AlignScore, MIT), pass alignscore_ckpt=...) is TRAINED
     (Adam, lr 1e-5, batch 16, up to 100 epochs, best-val-F1 kept) on a
     labelled benchmark (e.g. RAGTruth) to classify each
     (top-attributed evidence, answer sentence) pair.
  4. Threshold: the paper sweeps and reports alpha as a tunable and gives no
     calibration procedure, so calibrate_threshold() here sets it from held-out
     data such that a target share of correct samples passes.

White-box constraint: step 1 needs generator weights and gradients, so this
backend re-runs answers through HF transformers locally; the Ollama HTTP API
cannot serve it. LRP runs one backward per answer token: slow by design (the
paper flags the same complexity limit).

Install:  pip install -e ".[sirg]"   then   python -m spacy download en_core_web_sm
Train:    python -m veridic_eval.sirg train --data ragtruth.jsonl --out ./out/sirg_state.pt
Eval use: VERIDIC_EVAL_FAITHFULNESS=sirg  (requires the trained state file)

Every paper-underspecified choice is an explicit argument, never a hidden
constant. The discriminator MUST be trained (fit()) or loaded before scoring.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# NOTE: set prompt_template to the app's REAL serving template (see
# backend/services/chat-service/services/chat_handler.py) or attributions are
# computed against a prompt the generator never saw.
DEFAULT_PROMPT_TEMPLATE = (
    "Answer the question using only the provided documents.\n\n"
    "Documents:\n{context}\n\nQuestion: {question}\nAnswer:"
)

__all__ = [
    "SemanticUnit",
    "SentencePair",
    "extract_semantic_units",
    "pool_to_units",
    "build_sentence_pairs",
    "calibrate_threshold",
    "SIRGAttributor",
    "SIRGDiscriminator",
    "SIRGDetector",
    "DEFAULT_PROMPT_TEMPLATE",
]


# ------------------------------------------------------------- semantic units
@dataclass(frozen=True)
class SemanticUnit:
    text: str
    start: int  # char offset into its source string
    end: int
    kind: str   # "noun" | "ent" | "np" | "neg"


_NLP_CACHE: Dict[str, object] = {}
_STANZA = None


def _get_nlp(spacy_model: str = "en_core_web_sm"):
    nlp = _NLP_CACHE.get(spacy_model)
    if nlp is None:
        import spacy  # lazy

        nlp = spacy.load(spacy_model)
        _NLP_CACHE[spacy_model] = nlp
    return nlp


def _stanza_units(text: str, lang: str) -> List[SemanticUnit]:
    global _STANZA
    if _STANZA is None:
        import stanza  # lazy

        _STANZA = stanza.Pipeline(lang, processors="tokenize,ner", verbose=False)
    return [
        SemanticUnit(e.text, e.start_char, e.end_char, "ent")
        for e in _STANZA(text).ents
    ]


def _drop_contained(units: List[SemanticUnit]) -> List[SemanticUnit]:
    """Keep outermost spans; drop units fully inside an already kept unit."""
    kept: List[SemanticUnit] = []
    for u in sorted(set(units), key=lambda x: (x.start, -(x.end - x.start))):
        if not any(k.start <= u.start and u.end <= k.end for k in kept):
            kept.append(u)
    return kept


def extract_semantic_units(
    text: str,
    *,
    spacy_model: str = "en_core_web_sm",
    include_nouns: bool = True,
    include_entities: bool = True,
    include_noun_phrases: bool = True,
    include_negations: bool = True,
    use_stanza: bool = False,          # Appendix D lists Spacy AND Stanza
    stanza_lang: str = "en",
    dedupe_contained: bool = True,     # paper does not specify overlap handling
) -> List[SemanticUnit]:
    """Substantive words per the paper: nouns, NPs, named entities, negations."""
    doc = _get_nlp(spacy_model)(text)
    units: List[SemanticUnit] = []
    if include_noun_phrases:
        units += [SemanticUnit(c.text, c.start_char, c.end_char, "np") for c in doc.noun_chunks]
    if include_entities:
        units += [SemanticUnit(e.text, e.start_char, e.end_char, "ent") for e in doc.ents]
    if include_nouns:
        units += [
            SemanticUnit(t.text, t.idx, t.idx + len(t.text), "noun")
            for t in doc
            if t.pos_ in ("NOUN", "PROPN", "NUM")
        ]
    if include_negations:
        units += [
            SemanticUnit(t.text, t.idx, t.idx + len(t.text), "neg")
            for t in doc
            if t.dep_ == "neg"
        ]
    if use_stanza:
        units += _stanza_units(text, stanza_lang)
    if dedupe_contained:
        units = _drop_contained(units)
    return sorted(set(units), key=lambda x: (x.start, x.end))


# ----------------------------------------------------------- LRP attribution
class SIRGAttributor:
    """
    Token-level LRP relevance of each teacher-forced answer token to every
    prompt token, via `lxt` (AttnLRP) on a local HF causal LM.
    """

    def __init__(
        self,
        generator_model: str,
        *,
        device: str = "auto",           # "auto" | "cuda" | "cpu"
        dtype: str = "auto",            # "auto" (bf16 on cuda, f32 on cpu) | "bfloat16" | "float32"
        max_input_tokens: int = 4096,   # prompt truncation guard
        positions_per_pass: int = 16,   # answer tokens per forward pass (memory knob)
        trust_remote_code: bool = False,
    ):
        import importlib

        import torch  # lazy
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        try:
            from lxt.efficient import monkey_patch  # lxt >= 2.0 (paper uses LXT 2.0)
        except ImportError as exc:
            raise RuntimeError(
                'SIRG needs the `lxt` package: pip install -e ".[sirg]"'
            ) from exc

        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if dtype == "auto":
            dtype = "bfloat16" if self.device == "cuda" else "float32"
        self.dtype = getattr(torch, dtype)
        self.max_input_tokens = max_input_tokens
        self.positions_per_pass = positions_per_pass

        self.tokenizer = AutoTokenizer.from_pretrained(
            generator_model, trust_remote_code=trust_remote_code
        )
        # Patch the modeling module with LRP rules BEFORE instantiation when we
        # can resolve it; otherwise patch after (class-level, still effective).
        cfg = AutoConfig.from_pretrained(generator_model, trust_remote_code=trust_remote_code)
        model_cls = None
        try:
            from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

            model_cls = MODEL_FOR_CAUSAL_LM_MAPPING[type(cfg)]
            monkey_patch(importlib.import_module(model_cls.__module__), verbose=False)
        except Exception:
            model_cls = None
        self.model = (model_cls or AutoModelForCausalLM).from_pretrained(
            generator_model,
            torch_dtype=self.dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation="eager",  # LRP needs the eager attention graph
        ).to(self.device).eval()
        if model_cls is None:
            monkey_patch(importlib.import_module(type(self.model).__module__), verbose=False)

    def attribute(
        self,
        prompt: str,
        answer: str,
        *,
        normalize: str = "abs_sum",  # "abs_sum" per-row | "none"
    ) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns (R, prompt_offsets, answer_offsets):
          R[i, j]      relevance of answer token i to prompt token j
          *_offsets[k] (char_start, char_end) of token k in its own string
        """
        torch = self.torch
        tok = self.tokenizer
        p = tok(
            prompt,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        a = tok(answer, return_offsets_mapping=True, add_special_tokens=False)
        ids = list(p["input_ids"]) + list(a["input_ids"])
        n_p, n_a = len(p["input_ids"]), len(a["input_ids"])
        if n_a == 0 or n_p == 0:
            return np.zeros((n_a, n_p)), list(p["offset_mapping"]), list(a["offset_mapping"])

        input_ids = torch.tensor([ids], device=self.device)
        emb_layer = self.model.get_input_embeddings()
        R = np.zeros((n_a, n_p), dtype=np.float64)
        positions = list(range(n_p, n_p + n_a))  # token at t predicted from state t-1
        for cs in range(0, len(positions), self.positions_per_pass):
            chunk = positions[cs : cs + self.positions_per_pass]
            embeds = emb_layer(input_ids).detach().requires_grad_(True)
            logits = self.model(inputs_embeds=embeds).logits[0]
            for t in chunk:
                target = logits[t - 1, ids[t]]
                grad = torch.autograd.grad(target, embeds, retain_graph=True)[0]
                rel = (grad * embeds).sum(-1)[0].detach().float().cpu().numpy()
                row = rel[:n_p]
                if normalize == "abs_sum":
                    row = row / (np.abs(row).sum() or 1.0)
                R[t - n_p] = row
            del embeds, logits
        return R, list(p["offset_mapping"]), list(a["offset_mapping"])


def pool_to_units(
    R: np.ndarray,
    prompt_units: Sequence[SemanticUnit],
    prompt_offsets: Sequence[Tuple[int, int]],
    *,
    pool: str = "sum",  # "sum" | "mean" | "max" (paper: token -> semantic aggregation)
) -> np.ndarray:
    """Pool token columns of R into unit columns -> [n_answer_tokens x n_units]."""
    cols: List[np.ndarray] = []
    for u in prompt_units:
        idx = [j for j, (s, e) in enumerate(prompt_offsets) if s < u.end and e > u.start]
        if not idx:
            cols.append(np.zeros(R.shape[0]))
        elif pool == "sum":
            cols.append(R[:, idx].sum(axis=1))
        elif pool == "mean":
            cols.append(R[:, idx].mean(axis=1))
        else:
            cols.append(R[:, idx].max(axis=1))
    return np.stack(cols, axis=1) if cols else np.zeros((R.shape[0], 0))


# ------------------------------------------------------------ reasoning graph
@dataclass
class SentencePair:
    sentence: str
    start: int      # char span of the sentence within the answer
    end: int
    evidence: str   # top-attributed prompt units, relevance order


def build_sentence_pairs(
    answer: str,
    R_units: np.ndarray,                   # [n_answer_tokens x n_prompt_units]
    prompt_units: Sequence[SemanticUnit],
    answer_offsets: Sequence[Tuple[int, int]],
    *,
    spacy_model: str = "en_core_web_sm",
    top_k_evidence: int = 8,               # graph out-degree kept per sentence
    evidence_joiner: str = " ; ",
) -> List[SentencePair]:
    """
    One reasoning-graph row per answer sentence: pool the sentence's token
    rows, rank prompt units by pooled relevance, keep top_k as evidence text.
    """
    doc = _get_nlp(spacy_model)(answer)
    pairs: List[SentencePair] = []
    for sent in doc.sents:
        rows = [
            i
            for i, (s, e) in enumerate(answer_offsets)
            if s < sent.end_char and e > sent.start_char
        ]
        if not rows or R_units.shape[1] == 0:
            ev = ""
        else:
            scores = R_units[rows].sum(axis=0)
            order = np.argsort(scores)[::-1][:top_k_evidence]
            ev = evidence_joiner.join(
                prompt_units[int(i)].text for i in order if scores[int(i)] > 0
            )
        pairs.append(SentencePair(sent.text, sent.start_char, sent.end_char, ev))
    return pairs


# --------------------------------------------------------------- discriminator
def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def calibrate_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    target_pass_rate: float = 0.95,
) -> float:
    """
    Paper: "a threshold to dynamically adjust the pass rate of correct
    samples". Smallest threshold letting >= target_pass_rate of correct
    (label 0) samples score below it, i.e. pass as faithful.
    """
    correct = np.sort(probs[labels == 0])
    n = correct.size
    if n == 0:
        return 0.5
    k = min(n - 1, max(0, math.ceil(target_pass_rate * n) - 1))
    return float(min(1.0, correct[k] + 1e-6))


class SIRGDiscriminator:
    """
    Small-PLM pair classifier: P(hallucinated | evidence, sentence).
    Default init "roberta-base"; pass alignscore_ckpt (AlignScore-base .ckpt,
    RoBERTa 124M, https://github.com/yzha/AlignScore) to match the paper.
    """

    def __init__(
        self,
        init: str = "roberta-base",
        *,
        alignscore_ckpt: Optional[str] = None,
        device: str = "auto",
        max_length: int = 384,
    ):
        import torch  # lazy
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(init)
        self.model = AutoModelForSequenceClassification.from_pretrained(init, num_labels=2)
        if alignscore_ckpt:
            self._load_alignscore(alignscore_ckpt)
        self.model.to(self.device)

    def _load_alignscore(self, ckpt_path: str) -> None:
        """Best-effort init of the RoBERTa encoder from an AlignScore .ckpt."""
        state = self.torch.load(ckpt_path, map_location="cpu")
        state = state.get("state_dict", state)
        cleaned = {}
        for k, v in state.items():
            k2 = k
            for prefix in ("model.base_model.", "base_model.", "model.roberta.", "roberta."):
                if k2.startswith(prefix):
                    k2 = k2[len(prefix):]
                    break
            cleaned["roberta." + k2] = v
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if len(unexpected) >= len(cleaned):
            raise RuntimeError(f"AlignScore ckpt did not match the RoBERTa encoder: {ckpt_path}")

    def _encode(self, pairs: Sequence[Tuple[str, str]]):
        return self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

    def score(self, pairs: Sequence[Tuple[str, str]], *, batch_size: int = 32) -> np.ndarray:
        """P(hallucinated) for each (evidence, sentence) pair."""
        torch = self.torch
        self.model.eval()
        out: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch = self._encode(pairs[i : i + batch_size])
                probs = torch.softmax(self.model(**batch).logits, dim=-1)[:, 1]
                out.append(probs.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def train(
        self,
        pairs: Sequence[Tuple[str, str]],
        labels: Sequence[int],   # 1 = hallucinated sentence
        *,
        batch_size: int = 16,    # Appendix D
        epochs: int = 100,       # Appendix D: "100 iterations over the training set"
        lr: float = 1e-5,        # Appendix D (Adam)
        val_fraction: float = 0.1,
        seed: int = 42,
    ) -> Dict:
        """Adam training; the best-val-F1 checkpoint is kept (Appendix D)."""
        if len(pairs) < 2:
            raise ValueError("SIRG training needs at least 2 labelled pairs")
        torch = self.torch
        rng = random.Random(seed)
        idx = list(range(len(pairs)))
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_fraction))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        tr = [(pairs[i], int(labels[i])) for i in tr_idx]
        val_pairs = [pairs[i] for i in val_idx]
        val_y = np.array([int(labels[i]) for i in val_idx])

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        best_f1, best_state = -1.0, None
        for _ in range(epochs):
            self.model.train()
            rng.shuffle(tr)
            for i in range(0, len(tr), batch_size):
                chunk = tr[i : i + batch_size]
                batch = self._encode([c[0] for c in chunk])
                y = torch.tensor([c[1] for c in chunk], device=self.device)
                loss = torch.nn.functional.cross_entropy(self.model(**batch).logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            f1 = _f1(val_y, (self.score(val_pairs) >= 0.5).astype(int))
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        return {"best_val_f1": round(best_f1, 4), "n_train": len(tr_idx), "n_val": n_val}


# -------------------------------------------------------------------- detector
class SIRGDetector:
    """
    End-to-end SIRG scorer with a LettuceDetect-shaped predict() output.
    All paper-underspecified knobs are explicit constructor args.
    """

    def __init__(
        self,
        generator_model: str = "Qwen/Qwen2.5-7B-Instruct",  # HF twin of the served qwen2.5:7b
        *,
        discriminator_init: str = "roberta-base",
        alignscore_ckpt: Optional[str] = None,
        state_path: Optional[str] = None,       # trained discriminator + threshold
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        spacy_model: str = "en_core_web_sm",
        use_stanza: bool = False,
        threshold: float = 0.5,                 # P(halluc) >= threshold flags the sentence
        top_k_evidence: int = 8,
        pool: str = "sum",
        normalize: str = "abs_sum",
        device: str = "auto",
        dtype: str = "auto",
        max_input_tokens: int = 4096,
        positions_per_pass: int = 16,
        include_nouns: bool = True,
        include_entities: bool = True,
        include_noun_phrases: bool = True,
        include_negations: bool = True,
        trust_remote_code: bool = False,
    ):
        self.prompt_template = prompt_template
        self.spacy_model = spacy_model
        self.use_stanza = use_stanza
        self.threshold = threshold
        self.top_k_evidence = top_k_evidence
        self.pool = pool
        self.normalize = normalize
        self.include_nouns = include_nouns
        self.include_entities = include_entities
        self.include_noun_phrases = include_noun_phrases
        self.include_negations = include_negations
        self.attributor = SIRGAttributor(
            generator_model,
            device=device,
            dtype=dtype,
            max_input_tokens=max_input_tokens,
            positions_per_pass=positions_per_pass,
            trust_remote_code=trust_remote_code,
        )
        self.discriminator = SIRGDiscriminator(
            discriminator_init, alignscore_ckpt=alignscore_ckpt, device=device
        )
        self.trained = False
        if state_path and os.path.exists(state_path):
            self.load(state_path)

    # -- pipeline ----------------------------------------------------------
    def _pairs_for(self, question: str, context: str, answer: str) -> List[SentencePair]:
        prompt = self.prompt_template.format(context=context, question=question)
        R, p_off, a_off = self.attributor.attribute(prompt, answer, normalize=self.normalize)
        prompt_units = extract_semantic_units(
            prompt,
            spacy_model=self.spacy_model,
            include_nouns=self.include_nouns,
            include_entities=self.include_entities,
            include_noun_phrases=self.include_noun_phrases,
            include_negations=self.include_negations,
            use_stanza=self.use_stanza,
        )
        R_units = pool_to_units(R, prompt_units, p_off, pool=self.pool)
        return build_sentence_pairs(
            answer,
            R_units,
            prompt_units,
            a_off,
            spacy_model=self.spacy_model,
            top_k_evidence=self.top_k_evidence,
        )

    def predict(self, *, question: str, context: str, answer: str) -> Dict:
        """spans: [{"start","end","text","confidence"}] over flagged sentences."""
        if not self.trained:
            raise RuntimeError(
                "SIRG discriminator is untrained. Run "
                "`python -m veridic_eval.sirg train ...` or pass state_path."
            )
        pairs = self._pairs_for(question, context, answer)
        probs = self.discriminator.score([(p.evidence, p.sentence) for p in pairs])
        spans = [
            {"start": p.start, "end": p.end, "text": p.sentence, "confidence": float(pr)}
            for p, pr in zip(pairs, probs)
            if float(pr) >= self.threshold
        ]
        return {"spans": spans, "sentence_probs": [float(x) for x in probs]}

    # -- training ------------------------------------------------------------
    def fit(
        self,
        records: Sequence[Dict],
        *,
        batch_size: int = 16,
        epochs: int = 100,
        lr: float = 1e-5,
        val_fraction: float = 0.1,
        seed: int = 42,
        target_pass_rate: Optional[float] = 0.95,  # None keeps the preset threshold
    ) -> Dict:
        """
        records: {"question","context","answer"} plus either
          "spans": [[start,end], ...] gold hallucinated char ranges (RAGTruth-style), or
          "label": 0/1 answer-level (applied to every sentence of that answer).
        """
        pairs: List[Tuple[str, str]] = []
        labels: List[int] = []
        for rec in records:
            for p in self._pairs_for(rec["question"], rec["context"], rec["answer"]):
                if rec.get("spans") is not None:
                    y = int(any(s < p.end and e > p.start for s, e in rec["spans"]))
                else:
                    y = int(rec.get("label", 0))
                pairs.append((p.evidence, p.sentence))
                labels.append(y)
        info = self.discriminator.train(
            pairs, labels, batch_size=batch_size, epochs=epochs, lr=lr,
            val_fraction=val_fraction, seed=seed,
        )
        self.trained = True
        if target_pass_rate is not None:
            probs = self.discriminator.score(pairs)
            self.threshold = calibrate_threshold(
                probs, np.array(labels), target_pass_rate=target_pass_rate
            )
        info.update({"threshold": self.threshold, "n_pairs": len(pairs)})
        return info

    # -- persistence ---------------------------------------------------------
    def save(self, path: str) -> None:
        torch = self.discriminator.torch
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {"state_dict": self.discriminator.model.state_dict(), "threshold": self.threshold},
            path,
        )

    def load(self, path: str) -> None:
        torch = self.discriminator.torch
        blob = torch.load(path, map_location="cpu")
        self.discriminator.model.load_state_dict(blob["state_dict"])
        self.discriminator.model.to(self.discriminator.device)
        self.threshold = float(blob.get("threshold", self.threshold))
        self.trained = True


# ------------------------------------------------------------------------ CLI
def _cli(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m veridic_eval.sirg",
        description="Train the SIRG discriminator on a labelled JSONL benchmark.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train", help="train on JSONL (question/context/answer + spans|label)")
    tr.add_argument("--data", required=True)
    tr.add_argument("--out", required=True, help="output state file (*.pt)")
    tr.add_argument("--generator", default="Qwen/Qwen2.5-7B-Instruct")
    tr.add_argument("--discriminator-init", default="roberta-base")
    tr.add_argument("--alignscore-ckpt", default=None)
    tr.add_argument("--epochs", type=int, default=100)
    tr.add_argument("--batch-size", type=int, default=16)
    tr.add_argument("--lr", type=float, default=1e-5)
    tr.add_argument("--target-pass-rate", type=float, default=0.95)
    tr.add_argument("--limit", type=int, default=None, help="cap records (smoke runs)")
    args = ap.parse_args(argv)

    records = []
    with open(args.data, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    if args.limit:
        records = records[: args.limit]
    det = SIRGDetector(
        args.generator,
        discriminator_init=args.discriminator_init,
        alignscore_ckpt=args.alignscore_ckpt,
    )
    info = det.fit(
        records,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        target_pass_rate=args.target_pass_rate,
    )
    det.save(args.out)
    print(json.dumps({"saved": args.out, **info}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
