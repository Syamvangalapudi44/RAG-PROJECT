# ERROR ANALYSIS (run: hash embeddings + extractive generator; see results/comparison.md for every answer)

Failure types: stale / conflict / fragment / negation / fabricated (task definitions). `conflict` below means my grader saw the correct chunk inside the retrieved set but a wrong one was used.

## Before (naive top-3), 5/8 for both chunkers
| Q | fixed | structure |
|---|---|---|
| 1 intl per-diem | **conflict** - answered 6,500 from 2019 policy | **conflict** - 6,500 from 2019 |
| 2 home-office limit | correct (memo chunk happened to win - luck, not design) | **conflict** - 25,000 from the original clause; the memo was in top-3 |
| 3 contractor laptop | correct (but answer text cut mid-word: "...m") | correct |
| 4 Lead notice | **fragment** - table split; returned the header row, no "90 days" | correct (row chunk carries column names) |
| 5 receipts | **conflict** - 1,000 from 2019 | **conflict** - 1,000 from 2019 |
| 6 June 2019 | correct (2019 chunk happened to win; the system had no date logic) | correct (same) |
| 7 business class | correct | correct *content*, but cites only the 2019 file (the superseded document) |
| 8 parental leave | correctly abstained | correctly abstained |

No `negation` or `fabricated` failures occurred in this configuration. Caveat: the extractive generator cannot hallucinate; an LLM given Q8 and a weak context often will, and an LLM shown only clause 5.1 for Q3 would say "yes". Re-run with an LLM before claiming these types are absent.

## After (resolved): fixed 7/8, structure 8/8
Still wrong: **fixed chunking, Q4 (fragment)**. The 300-char window cuts the notice table so no chunk holds the "Lead / Manager | 90 days" row together with its header; the extractive answer returns the title and header line. Conflict resolution cannot repair a retrieval failure.
Fix with more time: never ship fixed-size for tabular docs; or, if forced to, add parent-document expansion (retrieve small, return the enclosing table/section).

## Residual defects that the accuracy number hides
- Q7 (structure, naive) and some resolved answers cite the 2019 file as a source for text identical in 2024; harmless here, wrong in principle - prefer in-force source.
- Resolved answers sometimes print a "Resolution:" note about an unrelated amendment (e.g. memo note on Q7) because the memo chunk landed in the top-3 pool. Noise, not an error; fix by emitting notes only when the amended clause is relevant to the question.
- Q6 is ambiguous (domestic vs international). I graded "give both" as correct.
