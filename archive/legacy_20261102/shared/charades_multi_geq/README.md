# Charades GMR + Multi + GEQ, seed 2023

This is the shared implementation for four backbones and six methods each. `multi` means **native GMR + Multi**, not a separate non-GMR backbone. Flash `multi` means **FlashVTG-GMR + Multi-TQAux**, a structural adaptation using a training-only two-layer temporal query decoder. CG-DETR, Shared, Adaptive and other modules are not experimental variants in this runner.

## Data and environment

Use Python 3.10 with the versions recorded in `requirements.txt`; the local run uses CUDA on two RTX 3090 GPUs. Native sources and the official evaluator are included under `vendor/`. No model checkpoint, video or feature file is distributed.

Set `CHARADES_PROJECT` to a directory containing `charades/d_gmr_two_stage/dataset/charades/`, with the four JSONL files under `splits/strict_seed2026/` and pre-extracted features under `features/lighthouse_bn45/{clip,slowfast,clip_text}/`. `strict_seed2026` is the existing split name; model training uses only seed2023. Optionally set `CHARADES_OUTPUT` to a new experiment directory. The original annotations/features are read-only inputs.

```bash
export CHARADES_PROJECT=/path/to/ImprovingGeneralizedMomentRetrieval
export CHARADES_OUTPUT="$CHARADES_PROJECT/charades/multiroute_geq_seed2023_20261101"
python prepare.py
```

`prepare.py` verifies the 21 contradictory negative exclusions, constructs a 27,788-row shared training view, creates immutable row evaluation IDs, and scans feature payloads for shape and finite values. Text features always use `original_qid`. All 7,440 Test-OOD rows, including duplicate appended records, retain their original weights. GT normalization uses real duration. Complete videos fit max_v_l=128 (observed maximum97).

## Preflight before formal training

First ensure GPU slots are available: every physical GPU may have at most two CUDA jobs, including jobs outside this experiment. The following diagnostic commands can be run sequentially, or assigned to free slots. They never update formal initialization or selected checkpoints.

```bash
CUDA_VISIBLE_DEVICES='' python test_flash_zero_point_guard.py
# Repeat the following three commands for moment_detr, eatr, qd_detr and flashvtg.
CUDA_VISIBLE_DEVICES=0 python debug_matrix.py moment_detr
CUDA_VISIBLE_DEVICES=0 python debug_resume_matrix.py moment_detr
CUDA_VISIBLE_DEVICES=0 python resume_audit.py moment_detr
python seal_preflight.py
```

All 24 configurations check real positives/ID negatives/OOD negatives, all-null batches, longest/tail-GT inputs, the physical batch, finite losses/gradients, actual GEQ optimizer updates, Full native encoder/existence/auxiliary gradients, Frozen parameter and buffer invariance, eval auxiliary suppression, export/metric computation, strict loading and a real process exit/restart. The separate real zero-regression-point check covers a positive Flash annotation whose dense head assigns no regression points. Diagnostic updates are restored. CUDA backward reductions can differ at floating-point roundoff; RNG and sampler state comparisons remain exact.

## Persistent queue

```bash
# Use tmux or an equivalent persistent service.
python queue_manager.py
```

The locked dispatcher accounts for existing CUDA processes. Each task has both task.lock and .lock, with atomic best/last saves and optimizer/RNG/sampler state. GEQ branches start only after the selected parent and same-backbone GMR Val reference are ready. Full and Frozen start independently from that same parent best. DataLoader workers=2, CPU-only feature work, AMP off. Physical batches: Moment8, EaTR32, QD32, Flash8; Flash eval1. No DDP or extra training seeds.

Stage A uses AdamW lr3e-5, weight decay1e-4, clip0.1, at most200 epochs, ungated positive-Val mAP selection, and20 non-improving training epochs. On 2026-11-01 the user authorized a prospective cadence change for unfinished Stage A runs: validate at epoch multiples of5, and additionally at the20-epoch patience boundary or final epoch before stopping. Historical every-epoch validations and existing best checkpoints are retained. Each changed run records its effective epoch, immutable legacy-history prefix hash, and resume checkpoint hash in `validation_schedule.json`; completed runs keep their original protocol. Skipped epochs record null Val metrics and do not compete for best. Thus this experiment has explicitly recorded mixed historical/prospective validation schedules, not a uniform every-five-epoch schedule from initialization.

Stage B continues to validate every epoch, using at most20 epochs; Full native lr1e-5 and GEQ lr3e-5, Frozen GEQ-only lr3e-5. Its selection score is complete-Val G-mIoU@1 averaged at0.4/0.5/0.6, with6 non-improving epochs. Only strict mean growth resets patience; equal means prefer all three values above GMR, then the earlier epoch. Test never selects epochs or thresholds.

EaTR Multi was restored to the original all-query auxiliary supervision on 2026-11-01 at the user's request; the interim positive-only experiment is archived separately. Original epochs1–33, best epoch13 and the epoch34 optimizer/RNG/sampler checkpoint are retained. Because that original run had already early-stopped, a hashed `USER_CONTINUE_REQUEST.json` authorizes one additional20-epoch training window through epoch53. Patience counts are preserved rather than rewritten; after this window the normal20-epoch no-improvement rule applies, with the overall200-epoch cap unchanged. The audit verifies this scoped exception. EaTR GMR native localization remains positive-only and existence BCE covers all queries; Multi auxiliary classification again supervises negative queries as background. LR3e-5 and batch32 are retained, so this does not reproduce Soccer's LR5e-5/effective-batch8 training setup.

The user subsequently authorized continuing the positive-only variant as well. It resumes independently under `runs/eatr/multi_positive_only`, using `--method multi --output-suffix _positive_only --auxiliary-query-scope positive_only_v1`. The existing dispatcher enables this additional Stage A ablation through `scheduling_policy.json: train_eatr_multi_positive_only=true`; it shares the two-jobs-per-GPU limit and never writes the canonical all-query run. Its Val history is shown separately. Main counts remain24 training/48 formal Test evaluations, with this additional training reported separately and required for final acceptance. Its original20-epoch patience remains unchanged.

Full retains native localization, native existence BCE and Multi auxiliary losses, plus GEQ BCE1.0. Frozen keeps the trunk in eval and detaches its evidence. Frozen exports copy the parent's fixed candidates exactly and replace only presence scores, enforcing unchanged boundaries/scores/order and mAP. Presence uses strict score > threshold. Flash Frozen computes native eval features one sample at a time while optimizing an eight-sample GEQ BCE batch.

## Metrics, plots, audit and release

The runner evaluates each final Val-best once per Test split and verifies all7,440 unique evaluation IDs. G-mIoU uses0–1; mAP/mR/R@1/AUROC use percent. Single-GT data have N/A mR+ and multi-GT coverage. Both localization and presence results are retained, including mAP decreases.

```bash
python plot_report.py
python completion_audit.py --checkpoints --recompute
# Re-render a saved figure without GPUs, checkpoints or features:
python plot_report.py --snapshot /path/to/full.snapshot.json --output /path/to/figure
```

The audit does not equate file existence with completion. It checks selection histories, checkpoint/parent hashes, identity coverage, recomputed metrics, frozen candidate invariance,101-point curves, CSV/snapshot provenance and required figure files. Final completion also requires visual review and verified GitHub publication. Timing in tables is explicitly labelled: recorded completed-epoch training plus Val wall time; interrupted partial attempts may be excluded. Inference time includes feature loading and export.

`prepare_release.py` writes only Charades-owned paths into the separate publication worktree. `publish_queue.py` retries Git operations with timeouts independently of CUDA scheduling. It never stages weights/features/videos or authentication files; all Git URLs remain credential-free. Final PUBLISHED requires a successful push and matching remote commit. Interim releases remain explicitly incomplete.

The existing SoccerGMR results in the repository are separate experiments and are preserved. Use the root Charades result page for this experiment; do not mix the two datasets or training budgets.

`PROFILE.json` records a Val-only CUDA forward benchmark (25 warmups,100 measured iterations). GEQ overhead compares against a separate native trunk with identical trained tensors and no GEQ feature hooks. The total Test inference metric additionally includes I/O/export. Timing does not select checkpoints or change hyperparameters.

Temporal export is fixed before Test: DETR normalized center/width uses real duration, Flash uses its native dense seconds output; native2s grid rounding is retained. The existing Charades adapter clamps to real duration (no fixed150s truncation), repairs only nonpositive-width windows inside the video, retains FP32 confidence, and uses native confidence order plus Flash native NMS. `export_protocol_audit.py` records these adaptations explicitly and verifies the native rounding functions; it does not modify outputs or checkpoints.
