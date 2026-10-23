# Data specification

## Dataset and scope

The repository uses the **Standard** Soccer-GMR split from Ding et al.; it does not introduce a new dataset. Soccer-GMR is a generalized temporal retrieval benchmark with no-target, single-target and multi-target query–video pairs. See the [benchmark paper](https://arxiv.org/abs/2605.02623) for the original annotation-construction procedure. Statistics below were computed directly from this repository's JSONL files, rather than copied from the benchmark's larger Full split.

| Split | Queries | Unique videos | No target | Single | Multi | Annotated moments | Maximum moments/query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 4,138 | 1,369 | 2,002 | 1,423 | 713 | 3,038 | 6 |
| Validation | 465 | 127 | 210 | 165 | 90 | 365 | 4 |
| Test | 1,036 | 461 | 492 | 384 | 160 | 750 | 5 |

All annotated intervals satisfy `0 <= start < end <= duration`. Every split has unique `qid` values. The union contains 1,957 distinct video IDs. Durations vary; 150 seconds is an upper bound in the active model interfaces, not the duration of every record. Complete histograms and file hashes are in [dataset_statistics.json](research/generated/dataset_statistics.json).

## Annotation contract

Each UTF-8 line in [train.jsonl](<../Phase2(addbackbone_with_S4_minusFusion)/shared/labels/train.jsonl>), [val.jsonl](<../Phase2(addbackbone_with_S4_minusFusion)/shared/labels/val.jsonl>) or [test.jsonl](<../Phase2(addbackbone_with_S4_minusFusion)/shared/labels/test.jsonl>) is a JSON object. The model-facing fields are:

| Field | Type and meaning |
| --- | --- |
| `qid` | Query-instance identifier; use it to join predictions to ground truth |
| `vid` | Video feature identifier; media extensions are stripped by the FlashVTG loader |
| `query` | Original natural-language expression |
| `duration` | Video duration in seconds |
| `relevant_windows` | Complete list of `[start_seconds, end_seconds]`; `[]` is a valid no-target label |
| `dataset_source` | Source grouping, currently `worldcup2022` or `sportsmoments` |
| `action_type`, `match_info`, `moment` | Retained provenance/annotation fields; presence and refinement use `relevant_windows` |

Existence is derived as `int(len(relevant_windows) > 0)`. Cardinality is derived from the complete list: zero, one, or at least two. An empty list must never be replaced by a zero-length dummy moment or removed during GMR training/evaluation. The loader uses an empty `(0,2)` span tensor for negative supervision.

Two Train records contain six intervals. The FlashVTG loader's `get_span_labels` shuffles and truncates the **span-label tensor** to `max_windows=5`; its in-place shuffle may change the metadata list's order but does not remove its entries. The S4 refinement loss reads the full metadata list, and A3 protects the full original union. Consequently, “all moments are supervised identically by every native backbone loss” is not an accurate description. Validation and Test contain no records above five intervals.

## Feature and batch contract

The original feature extractors are external assets. The full source workspace stores annotations and predictions, and its original Phase 2 archive additionally contains binary checkpoints; the GitHub publication omits those binaries; it does not store the large visual/text feature banks. The portable root FlashVTG API obtains checkpoint locations from external-path configuration. These two packaging arrangements should not be conflated.

| Input | Before projection | Preparation |
| --- | --- | --- |
| SlowFast video | `T x 2304` | Float32; normalize each clip vector |
| CLIP video | `T x 512` | Float32; normalize each clip vector |
| TEF | `T x 2` | Append recipient-relative `[i/T, (i+1)/T]` |
| Concatenated video | `T x 2818` | **SlowFast → CLIP → TEF**, maximum 75 clips |
| CLIP text | `L x 512` | `last_hidden_state`, normalized per token |
| Hidden text/video | Width 256 | Obtained from the backbone projections/encoder |

The nominal clip step is two seconds. Video streams are cropped to the configured maximum and aligned to their minimum available length before concatenation. Root FlashVTG text is cropped to 40 tokens; the archived DETR configurations use their native limits, commonly 32. Exact per-backbone settings remain in each `configuration/CONFIG.json`.

The portable FlashVTG layout is `slowfast/<vid>.npz`, `clip/<vid>.npz` with `features` arrays, and `clip_text/qid<qid>.npz` with `last_hidden_state`. Video loaders also support documented `.pt`/`.npy` fallbacks. Although query loading includes a `.pt` fallback, the preceding availability filter requires the query `.npz`; a `.pt`-only query bank can therefore be filtered out before loading. Use the expected `.npz` layout and run the row-count preflight.

Batches provide `src_vid`, `src_txt` and corresponding masks, with `True/1` meaning valid. Attention readers invert these masks for PyTorch's `key_padding_mask`. Existence labels cover every query. Negative saliency placeholders are all zero and use identical positive/negative indices. These placeholders do not imply temporal ground truth for absent events.

The Dataset can silently filter examples with missing features. The root environment's `make_dataset` asserts that the loaded count equals the JSONL count. A successful label-only documentation audit does not establish feature availability on another machine; run the asset-dependent preflight before training.

## Source distribution and split interpretation

| Split/source | Queries | No target | Single | Multi |
| --- | ---: | ---: | ---: | ---: |
| Train / WorldCup2022 | 3,435 | 2,002 | 720 | 713 |
| Train / SportsMoments | 703 | 0 | 703 | 0 |
| Val / WorldCup2022 | 384 | 210 | 84 | 90 |
| Val / SportsMoments | 81 | 0 | 81 | 0 |
| Test / WorldCup2022 | 820 | 492 | 168 | 160 |
| Test / SportsMoments | 216 | 0 | 216 | 0 |

All negative and multi-target records come from WorldCup2022; all SportsMoments records are positive singles. This source/label association permits source-conditioned behavior to contribute to aggregate discrimination. It does not by itself prove the trained model uses a shortcut. Report WorldCup-only AUROC and positive recall at high negative rejection alongside full-Test metrics. SportsMoments-only AUROC is undefined because that subset has one class.

| Split pair | Shared qids | Shared video IDs | Shared exact query strings | Shared exact `match_info` values |
| --- | ---: | ---: | ---: | ---: |
| Train / Val | 0 | 0 | 320 | 27 |
| Train / Test | 0 | 0 | 599 | 31 |
| Val / Test | 0 | 0 | 138 | 27 |

The partition is disjoint by recorded video ID. Exact match metadata overlaps where supplied, so it must not be described as match-disjoint. Repeated query templates are expected possible occurrences across different videos; their overlap alone is not a train/test record duplication. This audit does not infer unrecorded match identity from filenames or assess overlap between distinct video crops from the same broadcast. It also does not support “unseen actions,” “unseen teams,” or “unseen queries” merely from the workspace name.

## Optional A3 QCD augmentation

A3 is a separate exploratory experiment inspired by CVA's [Query-aware Context Diversification](https://arxiv.org/abs/2603.24934). The main five-backbone archive has augmentation disabled.

A3 keeps all 4,138 original Train records. Only the 1,433 WorldCup positive records (720 single + 713 multi) can be augmented, with probability 0.5. It protects every GT interval plus one adjacent clip on both sides. From the remaining eligible background, it replaces at most `round(0.30 * non_GT_background)` positions. Python's ties-to-even rounding, destination sampling without replacement and donor sampling with replacement are part of the implementation.

A donor is a distinct WorldCup Train video. Its clips are screened against the **recipient query** using projected, normalized CLIP sentence embeddings. Inclusive bounds are `theta_min=0.21965927` and `theta_max=0.26803598`, estimated respectively from Train non-GT and GT similarity distributions at percentiles 10 and 60. Empty candidate pools skip augmentation without retry. “Non-GT” denotes annotation membership, not a verified semantic negative.

SlowFast and CLIP vectors are copied synchronously from the same donor clip. The recipient's TEF, valid mask, query and complete labels remain. The original donor bank is immutable. Sentence projection uses the original cached EOT token and matching CLIP `text_projection`, before the model's text crop; mean token pooling is not equivalent. See [QCD_A3.md](<research/source_materials/provenance/QCD_A3.md>), [qcd.py](<../Phase2(addbackbone_with_S4_minusFusion)/flashvtg/code/qcd.py>) and [qcd_a3.json](<../Phase2(addbackbone_with_S4_minusFusion)/flashvtg/configs/qcd_a3.json>).

## Data provenance and reporting

The generated JSON includes SHA256 hashes for all three label files. Prediction tables include hashes of the exact archived files used. Original extraction and checkpoint identities remain in [provenance](../provenance/). The root MIT license is not evidence of rights to redistribute underlying broadcasts or externally supplied feature banks; no dataset licensing terms beyond the local evidence are asserted here.
