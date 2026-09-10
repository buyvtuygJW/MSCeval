# VERIDIC eval report

- reference cell: **control**
- conditions: control, chunk_opt, qdora, combined, prebaseline, prebaseline2
- IR cap: k = 5 (logs-only; served evidence only), served top_n 5

## Per-cell summary

| cell | runs | linked | recall@5 | ndcg@5 | faithful (strict) | grounded chars | abst.recall | over-abst | CFCA (GBP) |
|---|---|---|---|---|---|---|---|---|---|
| control | 1 | 12/12 | 0.5667 | 0.5610 | 0.7778 | 0.9837 | 0.6667 | 0.2222 | 0.0000 |
| chunk_opt | 1 | 12/12 | 0.6019 | 0.5324 | 1.0000 | 1.0000 | 1.0000 | 0.1111 | 0.0000 |
| qdora | 1 | 12/12 | 0.5667 | 0.5610 | 1.0000 | 1.0000 | 0.6667 | 0.1111 | 0.0000 |
| combined | 1 | 12/12 | 0.7130 | 0.6435 | 0.5556 | 0.8242 | 0.0000 | 0.0000 | 0.0001 |
| prebaseline | 1 | 10/10 | 0.8611 | 0.7833 | 0.6667 | 0.9728 | 1.0000 | 0.2222 | 0.0000 |
| prebaseline2 | 1 | 10/10 | 0.5667 | 0.5610 | 0.5556 | 0.8644 | 0.0000 | 0.0000 | 0.0000 |

(off-grid, measured but not scored in the 2x2: prebaseline, prebaseline2)

## Per-answer faithfulness (what the strict flag is made of)

| cell | query | strict | grounded | spans | flagged frac | answer chars | ctx chars | tok/win | trunc | first flagged span |
|---|---|---|---|---|---|---|---|---|---|---|
| control | q001 | 1 | 1.0000 | 0 | 0.0000 | 225 | 26606 | 2376/8192 | 0 |  |
| control | q002 | 0 | 0.9547 | 2 | 0.0453 | 287 | 6063 | 1577/8192 | 0 | July 21 |
| control | q003 | 1 | 1.0000 | 0 | 0.0000 | 158 | 61383 | 3765/8192 | 0 |  |
| control | q004 | 1 | 1.0000 | 0 | 0.0000 | 638 | 5386 | 1422/8192 | 0 |  |
| control | q005 | 1 | 1.0000 | 0 | 0.0000 | 706 | 7122 | 1497/8192 | 0 |  |
| control | q006 | 0 | 0.8988 | 1 | 0.1012 | 741 | 15117 | 1911/8192 | 0 | there are a total of 34 actions identified in the health and safety re... |
| control | q007 | 1 | 1.0000 | 0 | 0.0000 | 349 | 11215 | 1748/8192 | 0 |  |
| control | q008 | 1 | 1.0000 | 0 | 0.0000 | 279 | 5352 | 1390/8192 | 0 |  |
| control | q009 | 1 | 1.0000 | 0 | 0.0000 | 381 | 6706 | 1568/8192 | 0 |  |
| chunk_opt | q001 | 1 | 1.0000 | 0 | 0.0000 | 278 | 31885 | 5002/8192 | 0 |  |
| chunk_opt | q002 | 1 | 1.0000 | 0 | 0.0000 | 374 | 17170 | 4491/8192 | 0 |  |
| chunk_opt | q003 | 1 | 1.0000 | 0 | 0.0000 | 103 | 57019 | 4694/8192 | 0 |  |
| chunk_opt | q004 | 1 | 1.0000 | 0 | 0.0000 | 378 | 50229 | 5043/8192 | 0 |  |
| chunk_opt | q005 | 1 | 1.0000 | 0 | 0.0000 | 768 | 28562 | 4036/8192 | 0 |  |
| chunk_opt | q006 | 1 | 1.0000 | 0 | 0.0000 | 1078 | 47595 | 4586/8192 | 0 |  |
| chunk_opt | q007 | 1 | 1.0000 | 0 | 0.0000 | 334 | 42031 | 4707/8192 | 0 |  |
| chunk_opt | q008 | 1 | 1.0000 | 0 | 0.0000 | 557 | 31126 | 4565/8192 | 0 |  |
| chunk_opt | q009 | 1 | 1.0000 | 0 | 0.0000 | 812 | 30187 | 4820/8192 | 0 |  |
| qdora | q001 | 1 | 1.0000 | 0 | 0.0000 | 274 | 26606 | 2387/8192 | 0 |  |
| qdora | q002 | 1 | 1.0000 | 0 | 0.0000 | 267 | 6063 | 1575/8192 | 0 |  |
| qdora | q003 | 1 | 1.0000 | 0 | 0.0000 | 234 | 61383 | 3785/8192 | 0 |  |
| qdora | q004 | 1 | 1.0000 | 0 | 0.0000 | 802 | 5386 | 1459/8192 | 0 |  |
| qdora | q005 | 1 | 1.0000 | 0 | 0.0000 | 820 | 7122 | 1520/8192 | 0 |  |
| qdora | q006 | 1 | 1.0000 | 0 | 0.0000 | 799 | 15117 | 1925/8192 | 0 |  |
| qdora | q007 | 1 | 1.0000 | 0 | 0.0000 | 351 | 11215 | 1750/8192 | 0 |  |
| qdora | q008 | 1 | 1.0000 | 0 | 0.0000 | 257 | 5352 | 1386/8192 | 0 |  |
| qdora | q009 | 1 | 1.0000 | 0 | 0.0000 | 468 | 6706 | 1581/8192 | 0 |  |
| combined | q001 | 1 | 1.0000 | 0 | 0.0000 | 239 | 31885 | 5001/8192 | 0 |  |
| combined | q002 | 1 | 1.0000 | 0 | 0.0000 | 223 | 17170 | 4473/8192 | 0 |  |
| combined | q003 | 0 | 0.3189 | 4 | 0.6811 | 461 | 50229 | 5052/8192 | 0 | the |
| combined | q004 | 0 | 0.5723 | 1 | 0.4277 | 311 | 50229 | 5033/8192 | 0 | The report states that the next inspection and test should be conducte... |
| combined | q005 | 1 | 1.0000 | 0 | 0.0000 | 528 | 28562 | 4000/8192 | 0 |  |
| combined | q006 | 0 | 0.6529 | 3 | 0.3471 | 242 | 42332 | 4541/8192 | 0 | 15 actions |
| combined | q007 | 0 | 0.8736 | 4 | 0.1264 | 522 | 42031 | 4743/8192 | 0 | However |
| combined | q008 | 1 | 1.0000 | 0 | 0.0000 | 368 | 31126 | 4542/8192 | 0 |  |
| combined | q009 | 1 | 1.0000 | 0 | 0.0000 | 675 | 30187 | 4803/8192 | 0 |  |
| prebaseline | q001 | 1 | 1.0000 | 0 | 0.0000 | 434 | 38664 | 6500/8192 | 0 |  |
| prebaseline | q002 | 0 | 0.8971 | 8 | 0.1029 | 836 | 30319 | 7128/8192 | 0 | no junction |
| prebaseline | q003 | 0 | 0.9722 | 1 | 0.0278 | 288 | 238780 | 7485/8192 | 0 | contain |
| prebaseline | q004 | 1 | 1.0000 | 0 | 0.0000 | 490 | 124404 | 5532/8192 | 0 |  |
| prebaseline | q005 | 0 | 0.8862 | 3 | 0.1138 | 580 | 120419 | 4942/8192 | 0 | it does not |
| prebaseline | q006 | 1 | 1.0000 | 0 | 0.0000 | 871 | 52833 | 6970/8192 | 0 |  |
| prebaseline | q007 | 1 | 1.0000 | 0 | 0.0000 | 498 | 45141 | 5436/8192 | 0 |  |
| prebaseline | q008 | 1 | 1.0000 | 0 | 0.0000 | 550 | 39889 | 6884/8192 | 0 |  |
| prebaseline | q009 | 1 | 1.0000 | 0 | 0.0000 | 603 | 57723 | 6954/8192 | 0 |  |
| prebaseline2 | q001 | 1 | 1.0000 | 0 | 0.0000 | 133 | 26606 | 2359/8192 | 0 |  |
| prebaseline2 | q002 | 0 | 0.9858 | 2 | 0.0142 | 424 | 6063 | 1600/8192 | 0 | 21 |
| prebaseline2 | q003 | 0 | 0.7354 | 4 | 0.2646 | 378 | 61383 | 3818/8192 | 0 | pdf |
| prebaseline2 | q004 | 0 | 0.3333 | 1 | 0.6667 | 444 | 5386 | 1392/8192 | 0 | it is indicated that the report authorizes the next inspection on July... |
| prebaseline2 | q005 | 1 | 1.0000 | 0 | 0.0000 | 374 | 7122 | 1421/8192 | 0 |  |
| prebaseline2 | q006 | 1 | 1.0000 | 0 | 0.0000 | 913 | 5332 | 1652/8192 | 0 |  |
| prebaseline2 | q007 | 0 | 0.7247 | 3 | 0.2753 | 574 | 11215 | 1796/8192 | 0 | To determine |
| prebaseline2 | q008 | 1 | 1.0000 | 0 | 0.0000 | 279 | 5352 | 1399/8192 | 0 |  |
| prebaseline2 | q009 | 1 | 1.0000 | 0 | 0.0000 | 245 | 6706 | 1542/8192 | 0 |  |

- `strict` counts an answer faithful only at 0 confident ungrounded spans, so one flagged clause zeroes it, and a cell of near-clean answers still reports 0.0000; `grounded` is 1 - flagged/answer chars over the same answer.
- `tok/win` is the largest prompt the detector built against its own window; `trunc` > 0 means that prompt was cut to fit, and since the cut takes the context and keeps the answer, those spans were scored against evidence the model never read (`faithfulness.lettuce_window_read`).

## RAGA scores (LangMet `compute_raga_metrics`)

| metric | control | chunk_opt | qdora | combined | prebaseline | prebaseline2 |
|---|---|---|---|---|---|---|
| faithfulness | 0.9837 (9) | 1.0000 (9) | 1.0000 (9) | 0.8242 (9) | 0.9728 (9) | 0.8644 (9) |
| answer_relevancy | n/a (0) | n/a (0) | n/a (0) | n/a (0) | n/a (0) | n/a (0) |
| context_precision | 0.7074 (9) | 0.5747 (9) | 0.7074 (9) | 0.6858 (9) | 0.8395 (9) | 0.7074 (9) |
| context_recall | 0.5667 (9) | 0.6019 (9) | 0.5667 (9) | 0.7130 (9) | 0.8611 (9) | 0.5667 (9) |
| context_relevancy | 0.8549 (10) | 0.8667 (12) | 0.8549 (10) | 0.8649 (12) | 0.7044 (10) | 0.7115 (10) |
| answer_correctness | 0.2063 (9) | 0.1439 (9) | 0.2018 (9) | 0.2124 (9) | 0.1096 (9) | 0.1946 (9) |
| answer_similarity | 0.2625 (9) | 0.2027 (9) | 0.2683 (9) | 0.2808 (9) | 0.1627 (9) | 0.2562 (9) |
| **overall** | 0.5969 (12) | 0.5650 (12) | 0.5998 (12) | 0.5968 (12) | 0.6084 (10) | 0.5501 (10) |

(one RagaEvaluationEvent per linked answer; `(n)` is the events that carried that metric, so `n/a (0)` means the source is absent, not zero: answer relevancy has no logged score and no judge runs by default)


## Paired deltas vs control (BCa 95% CI; * = both CI ends one side of 0)

### chunk_opt_vs_control

| metric | delta | 95% CI | perm p (Bonf) | sig |
|---|---|---|---|---|
| recall@5 | 0.0352 | [-0.3704, 0.2333] | 1.0000 |  |
| mrr@5 | -0.1667 | [-0.5000, 0.0000] | 1.0000 |  |
| ndcg@5 | -0.0286 | [-0.4050, 0.1492] | 1.0000 |  |
| hit_rate@5 | -0.1111 | [-0.3333, 0.0000] | 1.0000 |  |
| faithful | 0.2222 | [0.0000, 0.5556] | 1.0000 |  |
| abstention_recall | 0.3333 | [0.0000, 1.0000] | 1.0000 |  |
| over_abstention_rate | -0.1111 | [-0.4444, 0.2222] | 1.0000 |  |
| cfca_joint_P | 0.2222 | [0.0000, 0.5556] | 1.0000 |  |
| CFCA (GBP) | 0.0000 | [-0.0000, 0.0000] | 0.1289 |  |

### qdora_vs_control

| metric | delta | 95% CI | perm p (Bonf) | sig |
|---|---|---|---|---|
| recall@5 | 0.0000 | [0.0000, 0.0000] | n/a |  |
| mrr@5 | 0.0000 | [0.0000, 0.0000] | n/a |  |
| ndcg@5 | 0.0000 | [0.0000, 0.0000] | n/a |  |
| hit_rate@5 | 0.0000 | [0.0000, 0.0000] | n/a |  |
| faithful | 0.2222 | [0.0000, 0.5556] | 1.0000 |  |
| abstention_recall | 0.0000 | [0.0000, 0.0000] | n/a |  |
| over_abstention_rate | -0.1111 | [-0.4444, 0.2222] | 1.0000 |  |
| cfca_joint_P | 0.2222 | [0.0000, 0.5556] | 1.0000 |  |
| CFCA (GBP) | 0.0000 | [-0.0000, 0.0000] | 0.5391 |  |

### combined_vs_control

| metric | delta | 95% CI | perm p (Bonf) | sig |
|---|---|---|---|---|
| recall@5 | 0.1463 | [0.0148, 0.2870] | 0.3750 | * |
| mrr@5 | -0.0556 | [-0.2222, 0.0370] | 1.0000 |  |
| ndcg@5 | 0.0825 | [-0.0305, 0.1956] | 0.6562 |  |
| hit_rate@5 | 0.0000 | [0.0000, 0.0000] | n/a |  |
| faithful | -0.2222 | [-0.5556, 0.2222] | 1.0000 |  |
| abstention_recall | -0.6667 | [-1.0000, 0.0000] | 1.0000 |  |
| over_abstention_rate | -0.2222 | [-0.5556, 0.0000] | 1.0000 |  |
| cfca_joint_P | -0.2222 | [-0.5556, 0.2222] | 1.0000 |  |
| CFCA (GBP) | 0.0001 | [0.0000, 0.0002] | 0.0234 | * |


## Off-grid before/after (validity check, uncorrected, not a 2x2 result)

The one-time baseline construction moved the pipeline from prebaseline, prebaseline2 to control: 256-tok truncation removed, embedder window >=512 tok, chunk size ~400 tok, hybrid BM25 and dense fused by RRF. This block sizes that move. It sits outside the Bonferroni family of 3 grid contrasts and carries no research claim.

### prebaseline_vs_control

| metric | delta | 95% CI | perm p (uncorrected) | n |
|---|---|---|---|---|
| recall@5 | 0.2944 | [0.0852, 0.5537] | 0.0938 | 9 |
| mrr@5 | 0.0741 | [-0.1667, 0.4444] | 0.7500 | 9 |
| ndcg@5 | 0.2223 | [0.0169, 0.5174] | 0.1406 | 9 |
| hit_rate@5 | 0.1111 | [0.0000, 0.5556] | 1.0000 | 9 |
| faithful | -0.1111 | [-0.5556, 0.2222] | 1.0000 | 9 |
| abstention_recall | 1.0000 | [1.0000, 1.0000] | n/a | 1 |
| over_abstention_rate | 0.0000 | [-0.4444, 0.4444] | 1.0000 | 9 |
| cfca_joint_P | -0.1111 | [-0.5556, 0.2222] | 1.0000 | 9 |
| CFCA (GBP) | 0.0000 | [0.0000, 0.0001] | 0.0312 | 9 |

### prebaseline2_vs_control

| metric | delta | 95% CI | perm p (uncorrected) | n |
|---|---|---|---|---|
| recall@5 | 0.0000 | [0.0000, 0.0000] | n/a | 9 |
| mrr@5 | 0.0000 | [0.0000, 0.0000] | n/a | 9 |
| ndcg@5 | 0.0000 | [0.0000, 0.0000] | n/a | 9 |
| hit_rate@5 | 0.0000 | [0.0000, 0.0000] | n/a | 9 |
| faithful | -0.2222 | [-0.5556, 0.2222] | 0.6250 | 9 |
| abstention_recall | 0.0000 | [0.0000, 0.0000] | n/a | 1 |
| over_abstention_rate | -0.2222 | [-0.5556, 0.0000] | 0.5000 | 9 |
| cfca_joint_P | -0.2222 | [-0.5556, 0.2222] | 0.6250 | 9 |
| CFCA (GBP) | 0.0000 | [-0.0000, 0.0001] | 0.1289 | 9 |

## Limitations (logs-only; see methodnow Part B)

- IR metrics capped at k <= top_n; pre-rerank pool not logged.
- Served order reconstructed as rerank desc, retrieval desc.
- Faithfulness context rebuilt from message_evidence -> chunks.text.
- Shallow-pool qrels bias; settle qrels + judge blind to condition.
- 'cited' = >=1 evidence link; 'right-version' from served doc order.

## CFCA (GBP, declared cost inputs)

| cell | cost/answer | P_hat | CFCA | C_onetime/A | C_query | recurring/Q | dCFCA vs control |
|---|---|---|---|---|---|---|---|
| `control` | 0.000013 | 0.777800 | 0.000017 | 0.000000 | 0.000013 | 0.000000 | - |
| `control` (wo warmup) | 0.000012 | 0.777800 | 0.000015 | 0.000000 | 0.000012 | 0.000000 | - |
| `chunk_opt` | 0.000024 | 1.000000 | 0.000024 | 0.000001 | 0.000023 | 0.000000 | 0.000007 |
| `chunk_opt` (wo warmup) | 0.000025 | 1.000000 | 0.000025 | 0.000001 | 0.000024 | 0.000000 | 0.000010 |
| `qdora` | 0.000022 | 1.000000 | 0.000022 | 0.000000 | 0.000022 | 0.000000 | 0.000005 |
| `qdora` (wo warmup) | 0.000022 | 1.000000 | 0.000022 | 0.000000 | 0.000022 | 0.000000 | 0.000007 |
| `combined` | 0.000043 | 0.555600 | 0.000077 | 0.000001 | 0.000042 | 0.000000 | 0.000060 |
| `combined` (wo warmup) | 0.000025 | 0.555600 | 0.000045 | 0.000001 | 0.000024 | 0.000000 | 0.000030 |
| `prebaseline` | 0.000029 | 0.666700 | 0.000044 | 0.000004 | 0.000025 | 0.000000 | 0.000027 |
| `prebaseline` (wo warmup) | 0.000026 | 0.666700 | 0.000039 | 0.000004 | 0.000022 | 0.000000 | 0.000024 |
| `prebaseline2` | 0.000015 | 0.555600 | 0.000027 | 0.000000 | 0.000015 | 0.000000 | 0.000010 |
| `prebaseline2` (wo warmup) | 0.000016 | 0.555600 | 0.000029 | 0.000000 | 0.000016 | 0.000000 | 0.000014 |

CFCA = cost per answer / P(faithful.cited.right-version), GBP, from the `cost:` quantities in conditions.yaml.

A `(wo warmup)` row re-prices one input and nothing else: the first 1 priced answer of each sitting is dropped, so `query_gpu_seconds` = (total_s - dropped_s) / (n_answers - 1), and the same P_hat divides it. A cold model load is billed to whichever answer waits for it, which is a property of the sitting and not of the condition. Dropped: `control` q001 10.01 s (20%), `chunk_opt` q001 4.12 s (4%), `qdora` q003 4.87 s (6%), `combined` q001 73.69 s (47%), `prebaseline` q002 16.75 s (21%), `prebaseline2` q001 2.13 s (5%). Watts, tokens, storage, the one-time build and every score are the run's own; nothing here is re-scored, and the reported rows stand.
