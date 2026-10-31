# CG-DETR-GMR requested extension

Status: prepared for the coordinator's dependency queue. No GPU job has been launched for this extension. CPU shape/BCE/gradient checks passed; full GPU smoke remains mandatory and is the first launcher action.

This is a locally reconstructed GMR adapter extension requested after the original CG-DETR launch. Table 2 has no published CG-DETR-GMR row; no published target or reproduction success is claimed. `gmr_source/` is an isolated snapshot of official CGDETR commit `212b2e49a0893512930d63b9294a8c066347e606`; original source and running original runner were not edited. Its model.py differs by only the existence head and its forward output. The head max-pools the ten last-decoder proposal slots and applies Linear(256,256), ReLU, Linear(256,1). It is initialized after the native modules so their seeded initialization is retained.

`train_gmr.py` applies every native VMR loss and every auxiliary decoder loss exclusively to positive queries. BCEWithLogits supervises existence on all naturally positive and null queries with coefficient 1. Native VMR losses are zero for an all-null batch. Every natural query and every annotated moment is retained. Synthetic pairing is disabled with vid=None. Dataset augmentation and text dropping are disabled; native architectural dropout remains.

Retained backbone settings: native 32 batch, 200 epochs, AdamW weight decay 1e-4, StepLR drop at 400, gradient clipping 0.1, ten proposal slots and native module depths/dimensions; requested learning rate 3e-5 and seed 2026. Soccer-GMR Standard CLIP+SlowFast/CLIP text inputs use existing two-second vectors, native feature normalization, 75 video vectors max and original max query tokens 32. Validation positive-query mAP selects the checkpoint each epoch, as the existing original deployment. Explicit sigmoid existence probability is emitted alongside original scored proposals for official evaluator gating at threshold 0.4. The checkpoint is frozen and hashed before test data are loaded; final test occurs once.

Launcher command (physical GPU binding is exclusively the central scheduler's job):

```
/home/guoxiangyu/miniconda3/envs/gmr/bin/python -u /home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/cg_detr/train_gmr.py --smoke-then-train
```

Run with cwd `/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/cg_detr`. Alternative wrapper: `bash /home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/cg_detr/launch_gmr.sh`. The direct Python PID runs mandatory smoke first, then os.execv replaces the same PID with a fresh trainer, resetting all model/optimizer/RNG/GPU state before scratch seed-2026 training. Smoke failure prevents training. Existing latest.pt resumes safely with optimizer/scheduler and Python/NumPy/torch/CUDA/DataLoader RNG state.

Full GPU smoke covers positive-only and null-only steps plus mixed batch32 with short/long and multi-moment inputs. It checks finite losses and all parameter gradients, nonzero existence-head gradients for positive/null/mixed cases, zero native VMR derivatives on null query outputs including all auxiliary proposals, strict checkpoint reload, full official evaluation of validation-format outputs with explicit existence scores, exact qid coverage, and peak GPU allocation. CPU checks do not replace this smoke. The original batch32 peak was 953 MiB; use at least the original 4000 MiB scheduler reservation until measured GMR peak is available.

Actual checkpoints, smoke evidence, training logs, final predictions, metrics, provenance, configuration, and snapshots belong to `backbonefolder/cg_detr/gmr/`. The original variant belongs to sibling `original/`. Runtime effective configuration and provenance record runner/model SHA256, source commit, environment versions, seed and CUDA binding. `gmr/source_hashes.json` pins prepared Python source and launcher content. Runtime logs must be directed by the coordinator to this actual gmr directory.
