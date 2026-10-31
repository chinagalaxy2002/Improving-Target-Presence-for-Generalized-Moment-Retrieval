# QD-DETR baseline deployment

Status: smoke passed; original Soccer-GMR single-seed training deployment. Reproduction metrics are pending full training and final test. This is not a QD-DETR-GMR experiment.

## Verified source and prior experiment audit

Official repository https://github.com/wjun0830/QD-DETR pinned commit `f8628f79f7c651b586300b142dbe9b85e43857cc`. Complete relevant official model/data/train source is archived under `backbonefolder/qd_detr/official_snapshot` via GitHub Contents API; `deployment` contains recorded local modifications. Direct Git transport failed and no commit was invented. Model uses original QD-DETR PReLU transformer, original global saliency token, 10 query slots, 2 encoder/decoder layers, hidden dimension256, video-conditioned query decoder. Model has7,585,573 parameters with no existence adapter or CAPS.

Inspected `experiments/caps_qd_detr_20261020`: prior E0 evaluation loaded a smoke-only checkpoint trained with seed2023 and the migrated QD-GMR architecture. Existing migration also uses ReLU rather than original PReLU. Prior CAPS scores/checkpoint are not used. Only verified dataset naming conventions inform this deployment.

## Protocol

Original official `data/label/Standard` train4138 / validation465 / held-out test1036 queries. All naturally empty samples and all annotated moments are preserved, no missing-file filtering or random maximum-window truncation. Maximum6 training moments fits original10 decoder proposals. Shared precomputed video representations load SlowFast2304 then CLIP512 then TEF2, with native per-stream L2 normalization. CLIP text tokens512, maximum32 tokens. Shared verified video features have75 vectors per150-second clip (effective2-second stride, despite frames sampled1fps); variable video lengths are retained. Temporal targets and predictions consistently use each video's actual duration.

Seed2026 for Python, NumPy, Torch CPU/CUDA, DataLoader generator; workers seed Python/NumPy from Torch's worker seed. Checkpoints preserve RNG states and DataLoader generator state. Native dropout remains. Dataset augmentation, query swap, QCD, synthetic negatives, background replacement, augmented merging, optional text token dropout all disabled.

Original QD parser supplies model/loss/optimization defaults: batch32,200 epochs, AdamW weight decay1e-4, gradient norm0.1, StepLR400 gamma0.1; paper reference learning rate3e-5 overrides QD's QVHighlights default1e-4. Span/GIoU/class/saliency coefficients10/1/4/1. No arbitrary early stopping during200 epochs. Validation mAP alone selects best checkpoint. Threshold0.4 is frozen before test. Final test inference/evaluation executes once after best checkpoint has been frozen; no test loader opens during optimization/checkpoint selection.

## Required local changes and limits

1. The user's ban on Query Swap/synthetic negatives conflicts with official QD's cyclic text-swapping saliency branch. That branch and its negative-pair objective are disabled. Native in-video saliency ranking/contrastive supervision remains, using original annotations only. This is a documented protocol deviation from unmodified QD training, mandated by the user; architecture is preserved.
2. Original criterion returns NaN for all-negative batches due to means over empty span matches. Empty span/GIoU terms return graph-connected zero; native unmatched-query background classification remains active. Empty matched-class logging returns zero.
3. Correct original attention mask head batching: `repeat_interleave(nheads,dim=0)` follows PyTorch's batch-major head ordering. The original `repeat(nheads,...)` interleaves examples incorrectly when masks differ.
4. Soccer saliency supervision uses union of all annotated intervals; all-negative samples contribute no positive-ranking terms. No additional negative query is generated.
5. Official QD inference uses clipping followed by2-second endpoint rounding and score rounding to4 decimals, without window-length clipping. Deployment preserves that order and adds actual-duration clipping after rounding for shorter videos. No NMS (official default-1). Exact prediction qid coverage is asserted for every split.
6. Exact QD Soccer hyperparameters/checkpoint were not released in official GMR repository. Dataset-independent QD defaults and paper reference learning rate are recorded explicitly; successful deployment does not establish Table2 metric reproduction.

## Smoke evidence

Shared GPU scheduler job `qd_detr_smoke_v1` passed on physical GPU1/logical cuda0. Verified separate multi-positive and natural-negative forwards, one mixed train step, one all-negative train step, full batch32 backward/AdamW step, finite losses and gradients, validation inference and official GMR evaluator format. Mixed loss6.010913, all-negative loss0.434012, batch32 loss5.245914. Peak allocated456.667MiB, reserved496MiB. `smoke.json` contains complete evidence, including strict checkpoint reload with exactly matching logits/spans and exact qid coverage. The first preliminary training reached epoch0 and was archived under `attempts/pre_postprocess_correction_20261022_1618` before correcting native endpoint postprocessing; it is excluded from final model selection. Full run reserves2000MiB through scheduler.

## Artifacts and acceptance

`effective_config.json`, `provenance.json`, `training.log`, `history.jsonl`, `last.ckpt`, validation-selected `best.ckpt`, best validation predictions/metrics, frozen checkpoint SHA, final test predictions and official GMR JSON are produced automatically by launcher. Test numerical evaluator is coordinator's pinned `shared/official_eval` package.

Initial diagnostic tolerance: absolute difference <=2 percentage points for localization/G-mIoU metrics, with all8 metrics reviewed including AUROC/Rej-F1. Differences outside tolerance require investigation; no reproduction classification will be claimed before final test evidence. Target QD row: AUROC64.71, Rej-F10.00, mAP9.62, G-mIoU@1 8.48, G-mIoU@3 4.22. Remaining mR metrics are taken from coordinator's paper table extraction, not invented.

Persistent scheduler runs in tmux `table2_scheduler_20261022`. Submit through shared scheduler; do not launch GPU workload directly. See `launch_command.txt` for exact command. Safe interrupted-run resume: submit a new/retried scheduler job with `train.sh --resume /absolute/output/last.ckpt`; successful completed runs require archival before replacement.

Resume preserves Python/NumPy/Torch CPU/CUDA RNG, training DataLoader generator state, optimizer and scheduler; worker processes are recreated per epoch with deterministic worker seeds. Full trajectory equivalence is not separately tested.
