# VERIDIC eval report

- reference cell: **control**
- conditions: control, chunk_opt, combined, prebaseline, prebaseline2
- IR cap: k = 5 (logs-only; served evidence only), served top_n 5

## Per-cell summary

| cell | runs | linked | recall@5 | ndcg@5 | faithful (strict) | grounded chars | abst.recall | over-abst | CFCA (GBP) |
|---|---|---|---|---|---|---|---|---|---|
| control | 1 | 12/12 | 0.5667 | 0.5610 | 0.6667 | 0.9544 | 0.6667 | 0.2222 | 0.0000 |
| chunk_opt | 1 | 12/12 | 0.6019 | 0.5324 | 1.0000 | 1.0000 | 1.0000 | 0.1111 | 0.0000 |
| combined | 1 | 10/12 | 0.6019 | 0.5324 | 0.4444 | 0.7997 | 0.0000 | 0.1111 | 0.0001 |
| prebaseline | 1 | 10/10 | 0.8611 | 0.7833 | 0.7778 | 0.9607 | 1.0000 | 0.2222 | 0.0000 |
| prebaseline2 | 1 | 10/10 | 0.5667 | 0.5610 | 0.6667 | 0.8642 | 0.0000 | 0.0000 | 0.0000 |

(off-grid, measured but not scored in the 2x2: prebaseline, prebaseline2)

## Per-answer faithfulness (what the strict flag is made of)

| cell | query | strict | grounded | spans | flagged frac | answer chars | ctx chars | tok/win | trunc | first flagged span |
|---|---|---|---|---|---|---|---|---|---|---|
| control | q001 | 1 | 1.0000 | 0 | 0.0000 | 225 | 26507 | 2325/4096 | 0 |  |
| control | q002 | 0 | 0.9268 | 2 | 0.0732 | 287 | 5956 | 1522/4096 | 0 | report |
| control | q003 | 0 | 0.7658 | 1 | 0.2342 | 158 | 61260 | 3702/4096 | 0 | Copperfield Way, St Mellons Cardiff. |
| control | q004 | 1 | 1.0000 | 0 | 0.0000 | 638 | 5263 | 1359/4096 | 0 |  |
| control | q005 | 1 | 1.0000 | 0 | 0.0000 | 706 | 6999 | 1434/4096 | 0 |  |
| control | q006 | 0 | 0.8974 | 1 | 0.1026 | 741 | 14994 | 1852/4096 | 0 | there are a total of 34 actions identified in the health and safety re... |
| control | q007 | 1 | 1.0000 | 0 | 0.0000 | 349 | 11092 | 1689/4096 | 0 |  |
| control | q008 | 1 | 1.0000 | 0 | 0.0000 | 279 | 5229 | 1331/4096 | 0 |  |
| control | q009 | 1 | 1.0000 | 0 | 0.0000 | 381 | 6583 | 1505/4096 | 0 |  |
| chunk_opt | q001 | 1 | 1.0000 | 0 | 0.0000 | 278 | 31786 | 4951/4096 | 1 |  |
| chunk_opt | q002 | 1 | 1.0000 | 0 | 0.0000 | 374 | 17063 | 4436/4096 | 1 |  |
| chunk_opt | q003 | 1 | 1.0000 | 0 | 0.0000 | 103 | 56896 | 4631/4096 | 1 |  |
| chunk_opt | q004 | 1 | 1.0000 | 0 | 0.0000 | 378 | 50106 | 4980/4096 | 1 |  |
| chunk_opt | q005 | 1 | 1.0000 | 0 | 0.0000 | 768 | 28439 | 3973/4096 | 0 |  |
| chunk_opt | q006 | 1 | 1.0000 | 0 | 0.0000 | 1078 | 47472 | 4523/4096 | 1 |  |
| chunk_opt | q007 | 1 | 1.0000 | 0 | 0.0000 | 334 | 41908 | 4648/4096 | 1 |  |
| chunk_opt | q008 | 1 | 1.0000 | 0 | 0.0000 | 557 | 31003 | 4506/4096 | 1 |  |
| chunk_opt | q009 | 1 | 1.0000 | 0 | 0.0000 | 812 | 30064 | 4757/4096 | 1 |  |
| combined | q001 | 1 | 1.0000 | 0 | 0.0000 | 288 | 31786 | 4960/4096 | 1 |  |
| combined | q002 | 0 | 0.9788 | 1 | 0.0212 | 424 | 17063 | 4465/4096 | 1 | SATISFACT |
| combined | q003 | 0 | 0.7593 | 4 | 0.2407 | 428 | 50106 | 4976/4096 | 1 | I'm sorry for |
| combined | q004 | 0 | 0.0000 | 1 | 1.0000 | 185 | 50106 | 4950/4096 | 1 | According to the EICR 2.pdf document, page 1, the next inspection and ... |
| combined | q005 | 1 | 1.0000 | 0 | 0.0000 | 499 | 28439 | 3946/4096 | 0 |  |
| combined | q006 | 0 | 0.9619 | 4 | 0.0381 | 867 | 47472 | 4483/4096 | 1 | report |
| combined | q007 | 0 | 0.4974 | 2 | 0.5026 | 774 | 41908 | 4738/4096 | 1 | The fire door issue falls under "High Priority," which, according to t... |
| combined | q008 | 1 | 1.0000 | 0 | 0.0000 | 184 | 31003 | 4444/4096 | 1 |  |
| combined | q009 | 1 | 1.0000 | 0 | 0.0000 | 618 | 30064 | 4726/4096 | 1 |  |
| prebaseline | q001 | 1 | 1.0000 | 0 | 0.0000 | 434 | 38565 | 6449/4096 | 1 |  |
| prebaseline | q002 | 1 | 1.0000 | 0 | 0.0000 | 836 | 30212 | 8578/4096 | 1 |  |
| prebaseline | q003 | 0 | 0.8472 | 2 | 0.1528 | 288 | 238657 | 14356/4096 | 1 | do |
| prebaseline | q004 | 1 | 1.0000 | 0 | 0.0000 | 490 | 124281 | 9973/4096 | 1 |  |
| prebaseline | q005 | 1 | 1.0000 | 0 | 0.0000 | 580 | 120296 | 8570/4096 | 1 |  |
| prebaseline | q006 | 1 | 1.0000 | 0 | 0.0000 | 871 | 52710 | 6911/4096 | 1 |  |
| prebaseline | q007 | 0 | 0.7992 | 4 | 0.2008 | 498 | 45018 | 5377/4096 | 1 | Report |
| prebaseline | q008 | 1 | 1.0000 | 0 | 0.0000 | 550 | 39766 | 6825/4096 | 1 |  |
| prebaseline | q009 | 1 | 1.0000 | 0 | 0.0000 | 603 | 57600 | 6891/4096 | 1 |  |
| prebaseline2 | q001 | 1 | 1.0000 | 0 | 0.0000 | 133 | 26507 | 2308/4096 | 0 |  |
| prebaseline2 | q002 | 1 | 1.0000 | 0 | 0.0000 | 424 | 5956 | 1545/4096 | 0 |  |
| prebaseline2 | q003 | 0 | 0.7196 | 4 | 0.2804 | 378 | 61260 | 3755/4096 | 0 | the address of the electrical installation is not explicitly stated in... |
| prebaseline2 | q004 | 0 | 0.3333 | 1 | 0.6667 | 444 | 5263 | 1329/4096 | 0 | it is indicated that the report authorizes the next inspection on July... |
| prebaseline2 | q005 | 1 | 1.0000 | 0 | 0.0000 | 374 | 6999 | 1358/4096 | 0 |  |
| prebaseline2 | q006 | 1 | 1.0000 | 0 | 0.0000 | 913 | 5209 | 1593/4096 | 0 |  |
| prebaseline2 | q007 | 0 | 0.7247 | 3 | 0.2753 | 574 | 11092 | 1737/4096 | 0 | To determine |
| prebaseline2 | q008 | 1 | 1.0000 | 0 | 0.0000 | 279 | 5229 | 1340/4096 | 0 |  |
| prebaseline2 | q009 | 1 | 1.0000 | 0 | 0.0000 | 245 | 6583 | 1479/4096 | 0 |  |

- `strict` counts an answer faithful only at 0 confident ungrounded spans, so one flagged clause zeroes it, and a cell of near-clean answers still reports 0.0000; `grounded` is 1 - flagged/answer chars over the same answer.
- `tok/win` is the largest prompt the detector built against its own window; `trunc` > 0 means that prompt was cut to fit, and since the cut takes the context and keeps the answer, those spans were scored against evidence the model never read (`faithfulness.lettuce_window_read`).

## RAGA scores (LangMet `compute_raga_metrics`)

| metric | control | chunk_opt | combined | prebaseline | prebaseline2 |
|---|---|---|---|---|---|
| faithfulness | 0.9544 (9) | 1.0000 (9) | 0.7997 (9) | 0.9607 (9) | 0.8642 (9) |
| answer_relevancy | n/a (0) | n/a (0) | n/a (0) | n/a (0) | n/a (0) |
| context_precision | 0.7074 (9) | 0.5747 (9) | 0.5747 (9) | 0.8395 (9) | 0.7074 (9) |
| context_recall | 0.5667 (9) | 0.6019 (9) | 0.6019 (9) | 0.8611 (9) | 0.5667 (9) |
| context_relevancy | 0.8549 (10) | 0.8667 (12) | 0.8719 (10) | 0.7044 (10) | 0.7115 (10) |
| answer_correctness | 0.2063 (9) | 0.1439 (9) | 0.1750 (9) | 0.1096 (9) | 0.1946 (9) |
| answer_similarity | 0.2625 (9) | 0.2027 (9) | 0.2425 (9) | 0.1627 (9) | 0.2562 (9) |
| **overall** | 0.5920 (12) | 0.5650 (12) | 0.5443 (10) | 0.6064 (10) | 0.5501 (10) |

(one RagaEvaluationEvent per linked answer; `(n)` is the events that carried that metric, so `n/a (0)` means the source is absent, not zero: answer relevancy has no logged score and no judge runs by default)


## Paired deltas vs control (BCa 95% CI; * = both CI ends one side of 0)

### chunk_opt_vs_control

| metric | delta | 95% CI | perm p (Bonf) | sig |
|---|---|---|---|---|
| recall@5 | 0.0352 | [-0.3704, 0.2333] | 1.0000 |  |
| mrr@5 | -0.1667 | [-0.5000, 0.0000] | 0.7500 |  |
| ndcg@5 | -0.0286 | [-0.4050, 0.1492] | 1.0000 |  |
| hit_rate@5 | -0.1111 | [-0.3333, 0.0000] | 1.0000 |  |
| faithful | 0.3333 | [0.1111, 0.6667] | 0.5000 | * |
| abstention_recall | 0.3333 | [0.0000, 1.0000] | 1.0000 |  |
| over_abstention_rate | -0.1111 | [-0.4444, 0.2222] | 1.0000 |  |
| cfca_joint_P | 0.3333 | [0.1111, 0.6667] | 0.5000 | * |
| CFCA (GBP) | 0.0000 | [-0.0000, 0.0000] | 1.0000 |  |

### combined_vs_control

| metric | delta | 95% CI | perm p (Bonf) | sig |
|---|---|---|---|---|
| recall@5 | 0.0352 | [-0.3704, 0.2333] | 1.0000 |  |
| mrr@5 | -0.1667 | [-0.5000, 0.0000] | 0.7500 |  |
| ndcg@5 | -0.0286 | [-0.4050, 0.1492] | 1.0000 |  |
| hit_rate@5 | -0.1111 | [-0.3333, 0.0000] | 1.0000 |  |
| faithful | -0.2222 | [-0.5556, 0.0000] | 1.0000 |  |
| abstention_recall | 0.0000 | [0.0000, 0.0000] | n/a |  |
| over_abstention_rate | -0.1111 | [-0.4444, 0.2222] | 1.0000 |  |
| cfca_joint_P | -0.2222 | [-0.5556, 0.0000] | 1.0000 |  |
| CFCA (GBP) | 0.0001 | [0.0000, 0.0004] | 0.0078 | * |


## Off-grid before/after (validity check, uncorrected, not a 2x2 result)

The one-time baseline construction moved the pipeline from prebaseline, prebaseline2 to control: 256-tok truncation removed, embedder window >=512 tok, chunk size ~400 tok, hybrid BM25 and dense fused by RRF. This block sizes that move. It sits outside the Bonferroni family of 2 grid contrasts and carries no research claim (methodnow Section 5).

### prebaseline_vs_control

| metric | delta | 95% CI | perm p (uncorrected) | n |
|---|---|---|---|---|
| recall@5 | 0.2944 | [0.0852, 0.5537] | 0.0938 | 9 |
| mrr@5 | 0.0741 | [-0.1667, 0.4444] | 0.7500 | 9 |
| ndcg@5 | 0.2223 | [0.0169, 0.5174] | 0.1406 | 9 |
| hit_rate@5 | 0.1111 | [0.0000, 0.5556] | 1.0000 | 9 |
| faithful | 0.1111 | [-0.2222, 0.5556] | 1.0000 | 9 |
| abstention_recall | 1.0000 | [1.0000, 1.0000] | n/a | 1 |
| over_abstention_rate | 0.0000 | [-0.4444, 0.4444] | 1.0000 | 9 |
| cfca_joint_P | 0.1111 | [-0.2222, 0.5556] | 1.0000 | 9 |
| CFCA (GBP) | 0.0000 | [0.0000, 0.0000] | 0.0391 | 9 |

### prebaseline2_vs_control

| metric | delta | 95% CI | perm p (uncorrected) | n |
|---|---|---|---|---|
| recall@5 | 0.0000 | [0.0000, 0.0000] | n/a | 9 |
| mrr@5 | 0.0000 | [0.0000, 0.0000] | n/a | 9 |
| ndcg@5 | 0.0000 | [0.0000, 0.0000] | n/a | 9 |
| hit_rate@5 | 0.0000 | [0.0000, 0.0000] | n/a | 9 |
| faithful | 0.0000 | [-0.4444, 0.4444] | 1.0000 | 9 |
| abstention_recall | 0.0000 | [0.0000, 0.0000] | n/a | 1 |
| over_abstention_rate | -0.2222 | [-0.5556, 0.0000] | 0.5000 | 9 |
| cfca_joint_P | 0.0000 | [-0.4444, 0.4444] | 1.0000 | 9 |
| CFCA (GBP) | 0.0000 | [-0.0000, 0.0000] | 0.6289 | 9 |

## Limitations (logs-only; see methodnow Part B)

- IR metrics capped at k <= top_n; pre-rerank pool not logged.
- Served order reconstructed as rerank desc, retrieval desc.
- Faithfulness context rebuilt from message_evidence -> chunks.text.
- Shallow-pool qrels bias; settle qrels + judge blind to condition.
- 'cited' = >=1 evidence link; 'right-version' from served doc order.

## CFCA (GBP, declared cost inputs)

| cell | cost/answer | P_hat | CFCA | C_onetime/A | C_query | recurring/Q | dCFCA vs control |
|---|---|---|---|---|---|---|---|
| `control` | 0.000013 | 0.666700 | 0.000020 | 0.000000 | 0.000013 | 0.000000 | - |
| `chunk_opt` | 0.000024 | 1.000000 | 0.000024 | 0.000001 | 0.000023 | 0.000000 | 0.000004 |
| `combined` | 0.000049 | 0.444400 | 0.000111 | 0.000001 | 0.000048 | 0.000000 | 0.000091 |
| `prebaseline` | 0.000029 | 0.777800 | 0.000037 | 0.000004 | 0.000025 | 0.000000 | 0.000018 |
| `prebaseline2` | 0.000015 | 0.666700 | 0.000023 | 0.000000 | 0.000015 | 0.000000 | 0.000003 |

CFCA = cost per answer / P(faithful.cited.right-version), GBP, from the `cost:` quantities in conditions.yaml (methodnow Sections 10 and 11).
