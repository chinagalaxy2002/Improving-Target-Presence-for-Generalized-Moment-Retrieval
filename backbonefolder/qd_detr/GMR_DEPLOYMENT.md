# QD-DETR-GMR queued extension

Prepared locally at user request. QD-DETR-GMR has no published Table2 row; this is a reconstructed paper-style extension, not a published-target reproduction. No GPU workload has been submitted for this extension. Full smoke is deferred to the durable scheduler queue and gates training.

Future direct scheduler command (single tracked PID):

```bash
/home/guoxiangyu/miniconda3/envs/gmr/bin/python -u /home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/qd_detr/train_gmr.py --variant gmr --smoke-then-train
```

Run with assigned physical GPU1 via scheduler, logical cuda0; reserve2000MiB initially based on original measured522MiB reserved plus adapter margin. Root coordinator owns dependency barriers and scheduling. The original queued jobs precede this extension.

Actual outputs: `backbonefolder/qd_detr/gmr/`. Model/code isolation: `gmr_deployment/` is copied from the frozen original deployment, with only the local adapter/criterion modifications; `train_gmr.py` is a separate copy of original runner. Active original sources and processes are untouched.

The model retains official QD-DETR PReLU encoder/decoder, slot matching and saliency modules. The new head applies elementwise maximum across10 final decoder slot embeddings, then Linear256→256, ReLU, Linear256→1. Sigmoid of the head logit supplies existence confidence; native foreground scores still rank proposals. No feature, adapter or checkpoint from CAPS is used. Initialization is seed2026 from scratch.

Criterion: native VMR losses and auxiliary decoder losses operate only on original queries with annotated moments; BCE-with-logits existence loss sees every original positive and naturally negative query with coefficient1. Null-only batches have zero VMR loss and train the existence head/backbone. No natural negative is discarded; all annotated moments remain. Optional augmentation and native Query Swap branch remain disabled, as documented for the original run.

Optimization retains native batch32,200 epochs, AdamW lr3e-5/weight decay1e-4, gradient norm0.1, StepLR400 gamma0.1. Checkpoint selection preserves validation positive-query mAP; G-mIoU is monitored at frozen existence threshold0.4. Test is opened only after the selected checkpoint/threshold have been frozen. Original2-second endpoint/4decimal score postprocessing, exact-duration bounds, split qid coverage checks, RNG/optimizer/scheduler resumption remain.

Deferred GPU smoke checks separate multi-positive/null forwards; mixed, null-only and fullbatch32 training with finite loss/gradients; finite nonzero gradients specifically in the existence head; strict save/reload with exactly equal proposal logits, spans and existence logits; official val evaluation and proposal bounds/qid coverage; and peak allocated/reserved VRAM. The paper mask invariant is checked by perturbing only null-query predictions and asserting native VMR loss is unchanged, while existence BCE exactly includes both queries. Decoder-slot permutation invariance checks max pooling. Smoke failures stop the process before training. After a successful smoke, the same tracked PID reseeds and reconstructs fresh model/optimizer so full training starts from scratch.

No GPU smoke results or reproduced metrics are claimed yet. On launch, the runner writes `smoke.json`, smoke checkpoint, full effective config and source provenance, training/validation history, best and last checkpoints, best validation predictions/metrics, frozen checkpoint SHA, final held-out test predictions and official evaluation JSON into the new local output directory. Queue stdout/stderr log is chosen by the coordinator in that directory.
