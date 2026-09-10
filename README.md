# eval

Logs-only, fully offline retrieval + LLM evaluation for the VERIDIC RAG app.

It reads the app's Postgres (`message_evidence`, `chunks`, `documents`,
`completion_logs`, `rag_logs`, `messages`) with no app-side schema or logging
change, and computes what `methodnow-rescoped.md` prescribes for the 2x2
(control, chunk-optimised, QDoRA, combined). The LLM is Ollama. All third-party
telemetry is disabled. Nothing replays the index: evaluation covers only what
the app already served.

Run procedure: **[WORKFLOW.md](WORKFLOW.md)**. This file is the reasoning behind it.

## You author

| File | Holds |
|---|---|
| `benchmark.yaml` | v1: the 10 questions, gold answers, `gold_evidence` as document + page + quoted text, answerable/unanswerable labels |
| `benchmarkv2.yaml` | v2: those 10 verbatim plus `q021` and `q022`, both unanswerable, so abstention is measured on 3 positives instead of 1. `benchmark.py:76-86` reads a fixed key set, so the marking rides on `category` (`unanswerable_v2`); a `version:` key would be dropped in silence |
| `conditions.yaml` | cells and their windows, grid/off-grid split, `device:`, `power:`, `ingest:`, `cost:` quantities, and each cell's `expect_policy:` chunking label, which `sit` and `run` read back and compare against the build the database actually holds |
| `human_validation.csv` | human labels for the detector spot-check. The name is yours: `agreement.agreement_from_csv(path, col_a, col_b)` reads whichever two columns you pass |

## What it computes

| Area | Metric | Tool |
|---|---|---|
| Retrieval | recall@5, mrr@5, ndcg@5, hit_rate@5 | ranx |
| Faithfulness | span-level ungrounded detection, faithful rate | LettuceDetect; RAGAS / Phoenix opt-in |
| Abstention | recall, precision, over-abstention (RGB *Rej*) | promptfoo `echo`, python fallback |
| Cost | GBP per answer, P(faithful.cited), CFCA (GBP) | LangMet + `cfca_cost.py` |
| Statistics | BCa 95% CIs, paired deltas, permutation p (Bonferroni) | scipy; deepsig.aso opt-in |
| Agreement | raw agreement rate, scorer vs runner-up detector | stdlib |
| Latency | p50/p95 end-to-end, retrieval/rerank | logs |

Only the served top_n evidence is persisted, so IR metrics cap at `k <= top_n`
(default 5), the math check fails a cell scored deeper than the pool its app
served, and retrieval-vs-rerank decomposition is out of scope.

LangMet (`mabrouka-abuhmida/LangMet`, MIT, no telemetry) supplies the
aggregation: its pure `analytics`/`cost` functions aggregate faithfulness
(`compute_raga_metrics`) and cost (`compute_cost_metrics`) from event objects
built out of VERIDIC's logs, which map almost 1:1 onto the app schema. Phoenix
(`Arize-ai/phoenix`) is wired only as an opt-in LLM-judge faithfulness backend
pointed at local Ollama with telemetry forced off, and stays off by default
because the methodology scores deterministically (RGB `Rej`, not LLM-judged
`Rej*`; LettuceDetect, not a judge).

## Layout

```
veridic_eval/
  telemetry_off.py  kill-switch for all third-party telemetry (import-time)
  config.py         Settings + Condition (one slice of the logs)
  db.py             read-only engine/session + ping()
  benchmark.py      BenchmarkQuery + load_benchmark()
  extract.py        THE ONLY DB READER: benchmark x condition -> QueryRecord[]
  gold_evidence.py  (document, page, text) -> chunk ids, per ingest
  retrieval_eval.py ranx qrels/run + evaluate_retrieval()
  faithfulness.py   lettucedetect | ragas | phoenix | sirg
  abstention.py     promptfoo echo + pure-python fallback
  agreement.py      raw agreement rate over two label columns
  cfca_metric.py    cost_overview() + compute_cfca()
  cfca_cost.py      GBP CFCA calculator (`veridic-eval cost`)
  devices.py        machine power profiles (omen, spark)
  power_meter.py    the meter log, sliced per cell (`veridic-eval meter`)
  power_log.py      one-shot poller (`veridic-eval power`)
  ingest_cost.py    one-time ingest hours, measured off the document stamps
  stats.py          bootstrap_ci, paired_delta_ci, permutation, bonferroni, aso
  cells.py          the 2x2 vocabulary: grid vs off-grid, contrast keys
  conditions.py     conditions.yaml parsed once
  sitting.py        window + chat id read off the logs (`veridic-eval sit`)
  templates.py      starter yamls for `veridic-eval init`
  report.py         evaluate_cell, evaluate_cell_runs, evaluate_experiment_split
  verify.py         post-run math check on a written report
  pipeline.py       run_experiment(): snapshot -> score -> CFCA -> check
  cells_app.py      cell dumps + the optional viewer
  cli.py            init | doctor | sit | run | verify | meter | view | cost | power
```

The unit passed between stages is a `QueryRecord`: served evidence in served
order with full chunk text, the answer, and the completion and rag log fields,
for one question in one run of one cell. Full flow: [Pipeline](#pipeline).

## DB join contract

Verified against `infra/postgres/init.sql` and `chat-service/routes/messages.py`.

| Eval need | Source | Note |
|---|---|---|
| Answer text | `messages.content` (assistant) | linked per cell |
| Question text | `messages.content` (user) -> next assistant | matched to the benchmark text |
| Served evidence + order | `message_evidence` | `ORDER BY rerank_score DESC NULLS LAST, retrieval_score DESC NULLS LAST`, mirrors `messages.py:254` |
| Faithfulness context | `message_evidence.chunk_id -> chunks.text` | full text, not the truncated snippet |
| Cost, tokens, latency | `completion_logs` by `message_id` | |
| RAG latency | `rag_logs` by `created_at` proximity | `rag_logs.message_id` is NULL (`retrieval.py:174`) |
| Pre-rerank pool | not persisted | IR capped at k <= top_n |

## Install

```powershell
python --version        # 3.11 or newer, LettuceDetect MIN
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

| Extra | Adds |
|---|---|
| `.[llm-judge]` | RAGAS faithfulness via Ollama |
| `.[phoenix]` | Arize Phoenix evals via Ollama, telemetry off |
| `.[sirg]` | SIRG white-box detector (`sirg.py`, arXiv 2601.03052) |
| `.[stats-extra]` | deepsig.aso significance (GPL-3.0) |
| `.[dev]` | pytest |

Abstention uses the promptfoo Node CLI when present, else an equivalent python
scorer. Copy `.env.example` to `.env`. The DB URL defaults to
`postgresql://postgres:admin@localhost:6432/veridic_db`, the compose-published
port. The first run pulls the LettuceDetect model
(`KRLabsOrg/lettucedect-base-modernbert-en-v1`, ~600MB) once; every run after is
offline. `sirg` needs training first
(`python -m veridic_eval.sirg train --data ragtruth.jsonl --out ./out/sirg_state.pt`)
and stays opt-in with the two non-deterministic judge backends.

## Design decisions

**Why each cell is snapshotted.** `message_evidence.chunk_id` cascades from
`documents`, so re-ingesting under the other chunker deletes the served evidence
of every cell already run. A cell leaves the database as JSON while its own
ingest is live, and the report scores those files. A dump writes one row per
question whether or not the app answered it, so only the linked count
distinguishes a measured cell from a window that matched nothing; a cell whose
questions were never asked is rejected rather than written as a 0-linked file
that later runs reuse. `--keep-empty` overrides that.

**Cells.** `prebaseline` runs first and sits below the grid: the app as it
stands, RCTS uncorrected, base model. `control` is the reference. The
`prebaseline` delta sizes the one-time baseline construction, a validity
prerequisite, so it reports outside the corrected family. Names in `cells.py`.

**conditions.yaml.** The app has no `condition` column and we make no schema
change, so a cell is a time window over the logs, or a list of conversation ids.
A cell asked more than once carries a `runs:` list, one window per sitting.
`sitting.py` writes those windows off the logs rather than by hand, running the
linker's own question comparison to find the chat that answered the cell, so a
window in the file is a window the linker can use and a question that exists
nowhere is named by id instead of collapsing into a silent `0/10 linked`.
A cell with no cost quantities gets no CFCA line instead of a fabricated 0, and
an unknown key raises instead of being dropped.

**The chunker is a process env, not a file.** `services/chunker.py:89` reads
`CHUNKING_MODE` out of the rag-service environment on every ingest, so the
builder is fixed before the stack starts and a duplicate line in
`env_local.env` resolves to whichever occurrence the parser keeps, silently.
`token_budget` sizes each chunk by the embedder's own tokenizer so no vector is
truncated at 256; `context_rag` sizes in words, `CHUNK_MAX_WORDS=310` with
`CHUNK_OVERLAP_WORDS=62` for the ~400 tok baseline
(`methodnow-rescoped.md:65`), and an empty pair falls back to the shipped
1000/200 in `chunking/chunk_builder.py`, which labels the cell
`context_rag_w1000_o200`, `prebaseline`'s own bucket. `MAX_SNIPPET_CHARS` stays
500 in every cell, so `chunk_opt` against `control` carries the chunker alone.
Each cell's `expect_policy:` is what catches a flip that never landed.

**Two benchmark markings.** Scoring reads a dump verbatim and never filters it
against the benchmark file, so v1 and v2 cannot share a cell directory: v2 runs
with `--cell-dir out/cells-v2 --out out-v2`, and its dumps hold 12 records. A
cell has a v2 number only if `q021` and `q022` were asked in its chat.
`prebaseline` and `prebaseline2` predate both items and their served evidence
rows are long cascaded away, so their v1 dumps are copied into the v2 directory
and reused untouched; their abstention stays a 1-positive number and is not
comparable to a v2 cell's 3.

**Repeats.** Each run is scored on its own, then every question is averaged
across the runs that answered it. The row and every paired delta use those
per-question means, so the CI is over questions, never over sittings, and a cell
run three times still pairs one-to-one against a reference run once.

**Gold as text, not ids.** Judgments are authored as `document` + 1-based `page`
+ the exact quoted `text`. Every ingest writes fresh `chunks.id` UUIDs and the
2x2 flips the chunker, so one cell's UUID list is wrong in the other three. Each
dump re-resolves the quotes against the ingest that served that cell, by
containment then 5-gram shingle coverage.

**Power.** One meter logs watts for the whole experiment; each cell is priced by
slicing that log. `busy` integrates only each answer's serving interval
(`answered_at` back by the retrieval, rerank and generation latencies, or by
`latency_ms` alone under `priced: generation`), `window` covers the sitting start
to end. An answer shorter than the poll cadence carries no sample of its own and
is integrated from the rows bracketing it (`power_meter.integrate_window`),
which needs `min_samples: 1`; the default 2 drops that span and lowers coverage.
A hole in the log lowers coverage instead of being invented; below
`min_coverage` the number falls back to the device nameplate and is labelled
`upper_bound`. `power_provenance` in the report says which happened per cell.

**Re-pricing after the fact.** A dump freezes the price of the moment it was
taken, under the `power:` block as it then read, so a block edited afterwards
reaches nothing already dumped. `reprice` re-reads the finished log over the
same answer spans and rewrites `meta.power`, `cost_inputs_measured` and
`serving_seconds` in place, keeping the first block under `power_at_dump`. It
touches no database, which is the point: a re-dump would read today's ingest,
and by then that is the next cell's corpus. Watts the meter never sampled are
not recoverable by either.

**The post-run math check.** `verify.py` re-derives every statistic from the
written report: Bonferroni divisors against the family size, CI ordering and
`significant` flags, the CFCA subtraction, every GBP row as its own cost over
its own all-pass rate, `k` against the width each cell served, off-grid
isolation from the grid. It runs after both files are written and exits
non-zero on a mismatch.

## Telemetry

`telemetry_off.py` runs at import, before any third-party module loads:
`HF_HUB_DISABLE_TELEMETRY=1`, `DO_NOT_TRACK=1`, `ANONYMIZED_TELEMETRY=false`,
`SCARF_NO_ANALYTICS=true`, `POSTHOG_DISABLED=true`, `OTEL_SDK_DISABLED=true`,
`LANGCHAIN_TRACING_V2=false`, `RAGAS_DO_NOT_TRACK=true`,
`PROMPTFOO_DISABLE_TELEMETRY=1`, `PHOENIX_DISABLE_TELEMETRY=true`,
`STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`, empty `SENTRY_DSN`. `HF_HUB_OFFLINE`
stays unset on purpose, so a first run can still fetch the LettuceDetect weights;
set it in `.env` once they are cached. LangMet does none.
`ANONYMIZED_TELEMETRY` is chromadb's variable, set defensively only:
chromadb is not a dependency and nothing here opens a vector store. The database
is opened read-only, so a run cannot mutate app data.

## Known soft spots

1. **LangMet install source.** Declared as PyPI `langmet>=0.1.0` and left
   unpinned, so pip takes the newest release; that is `0.3.0` (2026-06-02) here,
   and the line starts at `0.1.1` because no `0.1.0` was ever published. If
   resolution fails, switch to `git+https://github.com/mabrouka-abuhmida/LangMet`.
   Every call is guarded, so the deterministic path runs without it.
2. **Phoenix / RAGAS backends** are off by default; check their signatures first.
3. **promptfoo `results.json`** varies by version. On a parse failure the python
   scorer runs; `abstention.tool` in the report names which one did.
4. **CFCA is GBP off the measured watts**, so local runs price a real answer;
   `per_answer_cost:` only adds a provider's USD bill, converted at
   `cfca_cost.GBP_USD`.

## Limitations (restate in the thesis)

- IR capped at `k <= top_n`; no retrieval-vs-rerank decomposition.
- Served order reconstructed from the two scores; ranx gets rank-based scores.
- Faithfulness context rebuilt from `message_evidence -> chunks.text`, assuming
  a fixed prompt template. A template change between cells breaks comparability.
- `rag_logs` joined by timestamp proximity, since its `message_id` is NULL.
- Do not re-ingest mid-cell: the cascade wipes evidence history. Snapshot first.
- Shallow-pool qrels bias: judge blind to cell, settle the qrels before scoring.
- `cited` = at least one evidence link; `right-version` from served doc order.
  Both are logged proxies.

## Tests

```powershell
pytest -q     # offline: no DB, no Ollama, no network
```

Covers the telemetry kill-switch, stats (CI, delta, permutation, Bonferroni),
ratio stats, abstention, CFCA, cited and version logic, ranx IR, agreement, the
conditions parser, the grid/off-grid split, the snapshot pipeline, and the
post-run math check.

## Pipeline

```
sitting.open_sittings --> conditions.yaml   (veridic-eval sit, before any run:
                                             the real window and chat id, read
                                             off the logs of that sitting)

conditions.yaml --> conditions.load_conditions_file --> ConditionsFile
                      (windows, runs, grid/off-grid, snapshot, device, cost)
                                 |
                    pipeline.run_experiment
                                 |
                    0. meter log  out/power/power.csv     (started by hand,
                                 |   power_meter.log_power  runs all experiment)
                                 |
                    1. snapshot_cells --> out/cells/<cell>.json  (per-cell archive,
                                 |    cells_app.dump_cell_runs    written while that
                                 |    + gold_evidence               ingest is live)
                                 |    + power_meter.measure_cell_serving_power
                    2. load_cells <-- snapshots first, live DB as fallback
benchmark.yaml ---------> extract.extract_condition --> {cell: [QueryRecord]}
                                 |
                                 v
                          report.evaluate_experiment_split
                                 |  splits off the off-grid prebaseline window,
                                 |  then scores the 2x2. Per cell,
                                 |  evaluate_cell_runs averages the repeats per
                                 |  question, over evaluate_cell:
                                 |     retrieval_eval  (ranx)          -+
                                 |     faithfulness    (LettuceDetect)  | per-query
                                 |     abstention      (promptfoo)      | 0/1 columns
                                 |     cfca_metric     (LangMet)       -+
                                 |  stats: per-cell BCa CI + paired delta/perm vs reference
                                 |  off-grid: same deltas, plain BCa CI, no Bonferroni
                                 |
                    3. apply_measured_power --> measured watts into each cost block
                                 |    + ingest_cost.apply_measured_ingest
                                 |      the measured one-time ingest hours, same block
                    4. cfca_for_cells --> GBP cost per answer + CFCA per cell
                    5. verify.verify_report --> post-run math check
                                 v
                        out/report.json  +  out/report.md
```
