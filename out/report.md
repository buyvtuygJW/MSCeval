# VERIDIC eval report

- reference cell: **control**
- **PARTIAL REPORT**: reference cell 'control' has no scored answers; the delta tables return when that cell is scored and `run` is re-run
- conditions: qdora
- IR cap: k = 5 (logs-only; served evidence only), served top_n 5

## Per-cell summary

| cell | runs | linked | recall@5 | ndcg@5 | faithful (strict) | grounded chars | abst.recall | over-abst | CFCA (GBP) |
|---|---|---|---|---|---|---|---|---|---|
| qdora | 1 | 9/10 | 0.4556 | 0.4839 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |

## Per-answer faithfulness (what the strict flag is made of)

| cell | query | strict | grounded | spans | flagged frac | answer chars | ctx chars | tok/win | trunc | first flagged span |
|---|---|---|---|---|---|---|---|---|---|---|
| qdora | q001 | 1 | 1.0000 | 0 | 0.0000 | 281 | 26606 | 2393/8192 | 0 |  |
| qdora | q003 | 1 | 1.0000 | 0 | 0.0000 | 234 | 61383 | 3785/8192 | 0 |  |
| qdora | q004 | 1 | 1.0000 | 0 | 0.0000 | 802 | 5386 | 1459/8192 | 0 |  |
| qdora | q005 | 1 | 1.0000 | 0 | 0.0000 | 820 | 7122 | 1520/8192 | 0 |  |
| qdora | q006 | 1 | 1.0000 | 0 | 0.0000 | 799 | 15117 | 1925/8192 | 0 |  |
| qdora | q007 | 1 | 1.0000 | 0 | 0.0000 | 351 | 11215 | 1750/8192 | 0 |  |
| qdora | q008 | 1 | 1.0000 | 0 | 0.0000 | 257 | 5352 | 1386/8192 | 0 |  |
| qdora | q009 | 1 | 1.0000 | 0 | 0.0000 | 468 | 6706 | 1581/8192 | 0 |  |

- `strict` counts an answer faithful only at 0 confident ungrounded spans, so one flagged clause zeroes it, and a cell of near-clean answers still reports 0.0000; `grounded` is 1 - flagged/answer chars over the same answer.
- `tok/win` is the largest prompt the detector built against its own window; `trunc` > 0 means that prompt was cut to fit, and since the cut takes the context and keeps the answer, those spans were scored against evidence the model never read (`faithfulness.lettuce_window_read`).

## RAGA scores (LangMet `compute_raga_metrics`)

| metric | qdora |
|---|---|
| faithfulness | 1.0000 (8) |
| answer_relevancy | n/a (0) |
| context_precision | 0.7229 (8) |
| context_recall | 0.5125 (8) |
| context_relevancy | 0.8519 (9) |
| answer_correctness | 0.1769 (8) |
| answer_similarity | 0.2341 (8) |
| **overall** | 0.5831 (9) |

(one RagaEvaluationEvent per linked answer; `(n)` is the events that carried that metric, so `n/a (0)` means the source is absent, not zero: answer relevancy has no logged score and no judge runs by default)

## Limitations (logs-only; see methodnow Part B)

- IR metrics capped at k <= top_n; pre-rerank pool not logged.
- Served order reconstructed as rerank desc, retrieval desc.
- Faithfulness context rebuilt from message_evidence -> chunks.text.
- Shallow-pool qrels bias; settle qrels + judge blind to condition.
- 'cited' = >=1 evidence link; 'right-version' from served doc order.

## CFCA (GBP, declared cost inputs)

| cell | cost/answer | P_hat | CFCA | C_onetime/A | C_query | recurring/Q | dCFCA vs control |
|---|---|---|---|---|---|---|---|
| `qdora` | 0.000044 | 1.000000 | 0.000044 | 0.000000 | 0.000044 | 0.000000 | n/a |

CFCA = cost per answer / P(faithful.cited.right-version), GBP, from the `cost:` quantities in conditions.yaml.
