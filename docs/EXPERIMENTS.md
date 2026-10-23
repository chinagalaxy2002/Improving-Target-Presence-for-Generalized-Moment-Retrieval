# Experimental evidence and interpretation

## Protocols must remain separate

All documented results use seed 2026 and Standard Test's 1,036 queries, with 544 positives and 492 negatives. This is one dataset and one seed. A comparison against a checkpoint trained with a different schedule is a complete-system comparison, not a controlled attribution to an individual module.

| Experiment | Initialization and training | Checkpoint reporting | Test exposure |
| --- | --- | --- | --- |
| Root FlashVTG E0 | Native baseline from scratch on fixed features; batch 8; AdamW 3e-5, weight decay 1e-4; budget 100 epochs; StepLR 400; stale >23 stopping | Epoch 32, chosen by **native pre-NMS Val mAP** | Frozen-checkpoint reporting and threshold diagnostics |
| Root FlashVTG S4 | Same-seed E0 Val-best; new optimizer/scheduler; base LR 3e-5, LoRA LR 8e-5; batch 8; budget 40; StepLR 8; stale ≥15 | Epoch 4, chosen by **Val omni_spectrum** | Frozen-checkpoint reporting and threshold diagnostics |
| Phase 2 Moment/QD/CG/EaTR | Respective archived GMR baseline; six adaptation epochs; native batches 8/32/32/32; new AdamW and StepLR | Fixed epoch 4; all six epochs retained; no best-checkpoint selector | Test diagnostic evaluation at epochs 3–6 |
| FlashVTG + A3 | Fresh initialization from E0; same active S4 objective; 10 epochs, 5,180 updates | Epoch 4 by Val omni_spectrum | Epochs 4–10 inspected on Test; exploratory |

Gradient clipping is 0.1 and adaptation weight decay is 1e-4. The complete native baseline schedules differ across backbones; refer to each archived `configuration/CONFIG.json` rather than claiming one universal baseline budget. CG-DETR-GMR is a local extension, not an independently published Soccer-GMR Table 2 result.

The root S4 selector uses percentage-valued metrics:

$$
S_{\mathrm{Val}}=\frac{G_1(.4)+G_1(.5)+G_1(.6)}3
+0.4G_1(\tau_{G,\mathrm{Val}})
+0.5\operatorname{rawmAP}
+0.1\min(\operatorname{PosRec}(.5),\operatorname{Nacc}(.5)).
$$

The Val threshold scan uses .01–.99 at .01 spacing. Thresholds selected from Val can be applied to Test; the maximum of a Test sweep must not be presented as deployment performance. The Phase 2 fixed-epoch archive does not use this selector, even though it inherits some common configuration fields.

The original FlashVTG Test protocol predates the later adaptation runs. The four additional backbones and A3 have repeated Test diagnostics. They were not selected by the implemented Test metric, but they do not constitute a newly untouched Test evaluation. This limits the strength of confirmatory claims.

## Metrics and decision rule

Accept a query when `existence_score > tau`. Rejection accuracy is `TN/(TN+FP)` and positive recall is `TP/(TP+FN)`. Rejection treats absent events as its positive decision, so

$$
\operatorname{RejF1}=\frac{2TN}{2TN+FN+FP}.
$$

AUROC measures ranking of positive versus negative query–video pairs before choosing an operating threshold. A higher rejection rate at one threshold need not imply higher AUROC.

Localization mAP averages per-query temporal AP over positive queries and tIoU thresholds .50:.05:.95, using at most ten predictions. It is evaluated **without hard existence gating**. FlashVTG uses its native timestamp processing and NMS; the four DETR outputs retain native timestamp processing without an added NMS stage. Hard-gated mAP is a separate diagnostic. Preservation of ungated mAP does not guarantee preservation of accepted positive results.

For G-mIoU@k, reject to an empty set, then compare the first k predictions to the complete GT set. Both-empty scores one, exactly-one-empty scores zero, and two nonempty sets use the evaluator's greedy matching and `sum(matched tIoU)/(|pred|+|GT|-|matches|)`. The implementation permits zero-IoU matches at its zero threshold. Preserve original ranked positions, including degenerate candidates, before top-k slicing; cleaning/repacking may change results. This is set-level scoring, not ordinary top-1 overlap on positives.

For reference, always rejecting all Test queries yields **47.49% G-mIoU@k** for every k (`492/1036`). Thus a large gain over an always-accepting or poorly calibrated baseline requires simultaneous inspection of positive recall and localization. A high G score alone does not establish complete multi-moment recovery.

## Five-backbone comparison at τ=0.4

Values are percentages. Baselines below are local archived GMR checkpoints, not the numerical rows of a published paper. Current adaptation predictions were recomputed through the existing evaluator; baseline rows come from archived curve summaries with their own provenance hashes.

| Backbone | Model | AUROC | mAP | G-mIoU@1 | G-mIoU@3 | Negative rejection | Positive recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Moment-DETR | GMR baseline | 70.20 | 6.68 | 5.83 | 2.43 | 0.00 | 100.00 |
| | DeRoI-GMR | 70.26 | 7.59 | 51.92 | 48.97 | 100.00 | 39.71 |
| QD-DETR | GMR baseline | 77.01 | 14.62 | 36.51 | 31.47 | 56.71 | 75.55 |
| | DeRoI-GMR | 76.21 | 16.18 | 47.35 | 42.37 | 81.50 | 58.09 |
| CG-DETR | GMR baseline | 76.40 | 18.65 | 22.84 | 17.07 | 23.17 | 93.20 |
| | DeRoI-GMR | 76.72 | 19.33 | 50.52 | 45.59 | 89.02 | 52.02 |
| EaTR | GMR baseline | 76.56 | 16.20 | 33.32 | 28.34 | 48.58 | 80.51 |
| | DeRoI-GMR | 76.03 | 17.01 | 48.58 | 43.90 | 85.37 | 54.04 |
| FlashVTG | E0 GMR baseline | 74.30 | 25.43 | 28.10 | 21.75 | 32.72 | 86.76 |
| | DeRoI-GMR | 75.15 | 25.84 | 51.54 | 46.08 | 89.43 | 51.10 |

G-mIoU@1 improvements range from **10.84 to 46.09 percentage points**; ungated mAP improvements range from **0.41 to 1.56 points**. Positive recall decreases for all five backbones. AUROC changes by +0.06, −0.80, +0.32, −0.53 and +0.85 points respectively. The evidence supports improvements at the reported operating point, not uniformly better score ranking or calibrated probabilities.

Moment-DETR's archived baseline rejects no negatives at .4, while its adaptation rejects all negatives. The 46.09-point G@1 gain accompanies only .06 points of AUROC change. This makes threshold/score-distribution behavior a particularly important alternative explanation for its aggregate improvement.

| Backbone | mR@1 | mR@5 | mR+@5 | Rej-F1 |
| --- | ---: | ---: | ---: | ---: |
| Moment-DETR | 3.46 | 12.40 | 0.98 | 75.00 |
| QD-DETR | 9.75 | 23.77 | 12.26 | 71.54 |
| CG-DETR | 13.08 | 25.77 | 11.87 | 73.55 |
| EaTR | 11.44 | 22.48 | 9.63 | 72.29 |
| FlashVTG | 15.98 | 35.86 | 18.88 | 73.46 |

These mR values are the archived official positive-query localization diagnostics, not existence-gated end-to-end recall. Their exact definitions are in the original evaluator. A broad claim of “excellent multi-moment localization” would exceed the evidence, particularly for Moment-DETR.

## Threshold sensitivity on FlashVTG

| Model | τ | G-mIoU@1 | Rej-F1 | Negative rejection | Positive recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| E0 | .4 | 28.10 | 44.41 | 32.72 | 86.76 |
| S4 | .4 | 51.54 | 73.46 | 89.43 | 51.10 |
| E0 | .5 | 43.57 | 66.35 | 70.93 | 61.21 |
| S4 | .5 | 53.94 | 74.80 | 95.93 | 45.22 |
| E0 | .6 | 49.41 | 71.56 | 85.16 | 52.21 |
| S4 | .6 | 54.28 | 74.67 | 97.36 | 42.65 |

At .4, S4 changes TP/TN/FP/FN from **472/161/331/72** to **278/440/52/266**. G-mIoU@1 gains 23.44 points, but positive acceptance falls by 35.66 points. At .5, ungated mAP changes from 25.43 to 25.84, while hard-gated mAP falls from **16.07 to 11.16**. Report these observations together. See [fixed_threshold_results.csv](research/generated/fixed_threshold_results.csv) and the existing [full threshold curves](<research/source_materials/reports/threshold_grid_test_with_qcd.csv>).

The archived [subset report](<research/source_materials/reports/subsets_test_tau05.csv>) further limits the preservation claim. At .5, multi-target positive acceptance falls from **41.88% to 13.75%**, and multi-target G-mIoU@3 falls from **11.27% to 3.92%**. On WorldCup alone, positive acceptance falls from **35.67% to 9.15%**, although AUROC rises from 57.38% to 58.80%. SportsMoments positives are all accepted by both systems at this threshold. These are observations from the saved subset report, not newly trained comparisons.

## A3 is exploratory and threshold-specific

| Model | mAP | AUROC | G@1 at .4 | N-acc at .4 | PosRec at .4 | G@1 at .5 | N-acc at .5 | PosRec at .5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S4 | 25.84 | 75.15 | 51.54 | 89.43 | 51.10 | 53.94 | 95.93 | 45.22 |
| S4 + A3 | 26.37 | 75.77 | 50.99 | 88.01 | 51.47 | 53.67 | 95.12 | 45.77 |

A3 gains .53 mAP and approximately .62 AUROC points. At .4 its G@1 difference is **−.55**, and its rejection difference is **−1.42** points. The root README's nearby narrative differences of −.27 G@1 and −.81 N-acc correspond to **.5**, not .4. The documentation preserves the original files and resolves the ambiguity by labeling both thresholds explicitly.

The archived WorldCup-only comparison reports AUROC 59.85 versus 58.80, and empirical positive recall at N-acc≥95% of 10.67 versus 9.76. These are exploratory ROC diagnostics, not thresholds selected for deployment. They do not establish that augmentation solves positive rejection at high specificity.

## Supported claims and missing evidence

The implementation supports a global presence readout without direct proposal inputs and a separate query-conditioned refinement path. The experiment supports adaptation across five implemented backbones and the numerical comparisons above. Code structure alone does not establish that reused localization representations cause errors or that Gaussian pooling, the event query or LoRA independently cause the gains.

Before making stronger claims, obtain controlled global-query/refiner/LoRA/fusion ablations, equal-budget and matched-selection comparisons, repeated seeds with uncertainty, positive/multi/source-specific end-to-end results, and measured compute/memory/latency. These are future evidence requirements, not experiments completed in this documentation run. A new untouched holdout would be needed for a fresh confirmatory evaluation after the repeated Test diagnostics.
