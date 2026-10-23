# DeRoI-GMR component experiments (2026-10-23)

This isolated experiment follows `REQUEST.txt`. Original checkpoints, training code, and archived predictions are preserved. `SOURCE.json` identifies the supplied GitHub repository's local checkout and the source implementations copied here.

Five backbones: Moment-DETR, QD-DETR, CG-DETR, EaTR, FlashVTG. The formal component ablation contains exactly five backbones x six configurations = 30 experiments, with these independently initialized rows:

| Row | Configuration | LoRA | Refiner | 6D saliency | Candidate presence fusion |
|---:|---|---|---|---|---|
| 1 | Frozen coupled GMR Adapter | Off | Off | Off | Native GMR head |
| 2 | Decoupled Global Event Query | Off | Off | Off | Off |
| 3 | Row 2 + rank-8 LoRA | On | Off | Off | Off |
| 4 | Row 3 + hard rectangular RoI | On | Hard | Off | Off |
| 5 | Row 3 + Gaussian RoI | On | Gaussian | Off | Off |
| 6 | Full DeRoI-GMR / minusFusion | On | Gaussian | On | Off |

Separately, five additional fusion-enabled diagnostic runs (execution row 7) are stored under `reports/extra_diagnostics/` and are excluded from formal tables and counts. This diagnostic adds the reference candidate cross-attention, candidate matching MLP, local head, and zero-initialized residual fusion gate. It uses the same pre-refinement moment tokens, text state, six gate inputs, and residual equation as `MomentConditionedVerifier`. Rows 6 and 7 test removal of candidate presence fusion under matched training. The manuscript defines Full DeRoI-GMR as proposal-free presence, so row 6 is the full model. The fusion control is an additional experiment.

All added rows use the exact corresponding GMR baseline checkpoint, seed 2026, a fresh AdamW optimizer/scheduler, and four full epochs (fixed before training). They are not warm-started from the preceding row. Original backbone parameters remain frozen. All modules are materialized identically before switching behavior. Inactive parameter branches are frozen; native refiners in rows 1--3 are replaced by parameter-free identity modules. LoRA has scaling zero when off. Rows 4 and 5 disable the entire saliency projection output, including its bias. Hard pooling uses frame centers inside the exact candidate interval, with nearest-frame fallback for empty intervals, unlike the historical uniform_box implementation's extra one-second margins.

Training: original 4,138 Standard queries, no augmentation or synthetic negatives. Val: 465. Test: 1,036. Each new checkpoint is fixed at epoch 4, reloaded strictly, then evaluated on Val and Test. There is no model or threshold selection using Test. Historical experiments on these labels are already known, so these remain exploratory single-seed comparisons. Row 1 is a frozen baseline evaluation (zero new training epochs).

Metrics are the official evaluator's N-acc (negative recall), Rej-F1, positive-query ungated mAP, multi-positive mR+@5, and G-mIoU@1. Primary tau is 0.4; 0.5/0.6 are frozen sensitivity analyses. Every row uses its backbone's same export/postprocessing: native rounded 10-candidate export for DETR-style models; native FlashVTG NMS, top-10 metric limit. Thus a row gain cannot come from enabling NMS only on that row. Different backbone postprocessing is not mixed within a component comparison.

Run artifacts: `runs/<backbone>/row_XX/` contains config and source hashes, initialization provenance, preflight, losses, checkpoint hash, Val/Test query predictions, official reports, and completion record. `QUEUE.json` records actual subprocess PIDs and exit status; a running label alone does not prove a process is live. Failed runs are never silently restarted. `preflights/` preserves initial smoke attempts, including the corrected FlashVTG evaluation batch-size and CG-DETR output-key adaptations.

Launch and inspect:

```bash
/home/guoxiangyu/miniconda3/envs/gmr/bin/python launch.py
/home/guoxiangyu/miniconda3/envs/gmr/bin/python summarize.py
```

The launcher uses two GPU workers and refuses to overwrite run directories. Do not launch a second instance while the first is active. Summaries update after each completed run in `reports/RESULTS.md` and `reports/component_ablation.csv`; only genuinely completed runs enter these tables.

Table 4 of `../../write/DeRoI_GMR_Paper.tex` is filled by `table4.py` from the exact historical selected FlashVTG validation predictions. `table4/EVIDENCE.json` records archive hashes and complete official reports; `table4/DeRoI_GMR_Paper.before.tex` preserves the input manuscript. The native pre-NMS noPH G-mIoU@1 is corrected to 16.47: the old 16.49 came from post-NMS always-accept. Single-only mR+ and negative-only localization remain undefined. These inference-stage rows all use the same trained LoRA checkpoint and cannot establish training-component causality; the new matched ladder supplies that evidence.


`CONFIG.json` records the construction job that materializes the complete architecture; its common flags are not the effective ablation switches. Historical epoch-3--6 plotting/evaluation fields inherited from the source job are inert in this trainer; `PROTOCOL.json`, `run.py`, and the sidecar effective training record define the actual fixed-epoch-4 procedure. After independent reconstruction, `EFFECTIVE_CONFIG.json` records the actual switches, active parameter groups, learned parameter groups, shared initialization checks, and frozen tensor checks. `verify_components.py` also checks hard-window locality, Gaussian sensitivity to nearby outside context, padding masks, and disabled saliency output. `verify_fusion.py` checks proposal-free presence under candidate perturbation and checks native fusion outputs against the complete reference verifier equation.

`finalize.py --wait` waits for the existing queue, then reconstructs every completed variant, runs the behavioral controls, recomputes the 180 formal split/threshold evaluations and, separately, 30 additional diagnostic evaluations, and generates a revised manuscript copy and PDF in `paper/`. It never launches training or chooses a checkpoint. The revised copy replaces Figures 3--4 with the formal 30 paired validation/test component results and a separately labelled additional fusion diagnostic; the original manuscript keeps its existing main-results layout and receives only the Table 4 corrections. `paper/EVIDENCE_MAP.json` records the scope of the new text. Existing bibliography and historical main-result claims are outside this ablation subsection's review scope.

The local Tectonic compiler is recorded in `tools/TECTONIC_SOURCE.json`. `ENVIRONMENT.json` records the Python, CUDA, and evaluation/training package versions. The paper package is separate from the training checkpoints and raw prediction archives.

Native FlashVTG exports contain some zero-width candidates, including a few within the raw top 10. These are preserved to keep the original export and official metric semantics unchanged: G-mIoU cleans degenerate windows before gating, while localization metrics use their own raw top-K rules. The final numerical audit records their positions in `reports/NATIVE_DEGENERATE_WINDOWS.json`; it does not silently repair or remove prediction rows. Other backbones use the shared positive-width export documented above.

FlashVTG's native exporter also retains its existing soft presence-score calibration (scale candidate confidence when presence is below 0.5), three-decimal rounding, and NMS. Replacing the presence head leaves all raw `pred_relevant_windows_pre_exist` lists exactly unchanged but can alter submitted confidence ties/order and mAP. In rows 1--2, Val/Test mAP is 29.16/25.43 versus 28.96/25.40; mR+@5 is unchanged. The four DETR-style exports are exactly equal in rows 1--2. The audit checks the correct raw-localizer invariant for FlashVTG and does not assume that its final exported scores are independent of presence calibration. No exporter or prediction files were changed to force invariance.
