# Completed DeRoI-GMR component experiment

The formal 5 backbones x 6 configurations = 30 experiments are complete: 60 validation/test prediction sets and 180 threshold evaluations. Separately, five additional fusion diagnostics are complete: 10 prediction sets and 30 threshold evaluations. Both scopes pass official recomputation. Initial checkpoints are identical within each backbone; frozen parameters and persistent buffers are unchanged.

The formal component table reports the formal 30 experiments (five backbones and six configurations), using matched single-seed runs on the original Standard split (4,138 training, 465 validation, and 1,036 test queries). Each added configuration starts independently from the corresponding trained GMR Adapter checkpoint with a fresh optimizer and a fixed budget of four adaptation epochs. The baseline checkpoint is evaluated without further training. We fix tau=0.4 and retain the same postprocessing for every row within each backbone; neither test scores nor threshold scans select the checkpoint. These exploratory runs are reported separately from the historical full-system results in the historical main-results table. The shared export yields frozen CG-DETR/EaTR baseline mAP of 18.66/16.22%, compared with 18.65/16.20% in the archived main table; all conditional differences use the newly exported baselines.

Adding only the global event query changes test N-acc by +23.58 to +100.00 percentage points, Rej-F1 by +9.38 to +75.00 points, and G-mIoU@1 by +10.09 to +46.19 points across the five backbones. For the four DETR-style backbones, submitted candidates remain exactly equal to the baseline for every validation and test query, so mAP and mR+@5 are unchanged. FlashVTG preserves its raw pre-existence candidates exactly, but its unchanged native exporter rescales confidence with the new presence score before rounding and NMS: validation/test mAP changes from 29.16/25.43 to 28.96/25.40%, while mR+@5 stays unchanged. Thus its exported confidence scores are not independent of presence calibration.

Adding LoRA alone changes test mAP by -0.91 to +0.23 points, with increases on 2 backbones and decreases on 3. The corresponding mR+@5 changes range from -0.89 to +0.03 points. Adding the hard-RoI refiner changes test mAP by +0.59 to +2.78 points, with increases on 5 backbones and decreases on 0. The corresponding mR+@5 changes range from -0.15 to +4.57 points. Replacing hard with Gaussian RoI pooling changes test mAP by -0.91 to +1.43 points, with increases on 3 backbones and decreases on 2. The corresponding mR+@5 changes range from -0.74 to +0.78 points. Adding 6D saliency changes test mAP by -1.03 to +0.76 points, with increases on 2 backbones and decreases on 3. The corresponding mR+@5 changes range from -0.59 to +0.44 points. These conditional differences are limited to this single seed and fixed training budget.

Table 4 supplies a complementary inference-stage check on the historical FlashVTG checkpoint. Holding its trained LoRA weights fixed, pre-NMS refinement changes validation mAP from 26.55% to 26.81% (+0.26 points), while NMS raises it further to 28.63% (+1.82 points). The stage comparison therefore separates refinement from NMS at fixed weights; the matched training rows above assess the additional effects of LoRA, pooling, and saliency.

## Additional fusion diagnostics (outside the formal 30)

The five fusion-enabled runs in the additional diagnostic table are additional diagnostics outside the formal 30. Removing candidate presence fusion changes test G-mIoU@1 by -1.07 to +0.89 points and Rej-F1 by -0.67 to +0.99 points relative to the matched fusion-enabled control. G-mIoU@1 increases on 3 backbones and decreases on 2. This comparison isolates the inclusion of candidate presence fusion within the stated adaptation protocol; it does not imply a seed-independent ranking.

[Complete metric tables](reports/RESULTS.md) | [CSV](reports/component_ablation.csv) | [Conditional deltas](reports/component_deltas_pp.csv) | [Additional fusion diagnostics](reports/extra_diagnostics/README.md) | [Additional minus-fusion deltas](reports/extra_diagnostics/minusFusion_control_pp.csv)

[Revised manuscript source](paper/DeRoI_GMR_Paper_with_ablations.tex) | [Revised PDF](paper/DeRoI_GMR_Paper_with_ablations.pdf) | [Reproducible paper package](paper/DeRoI_GMR_with_ablations.zip)

The revised manuscript copy replaces Figures 3--4 and expands Table 4 for readability. The original source receives the Table 4 corrections only. Existing main-result comparisons and bibliography are historical material and were not independently re-reviewed in this ablation task.

[Historical baseline export differences](reports/HISTORICAL_BASELINE_COMPARISON.json)

[Numerical audit](reports/AUDIT.json) | [Table 4 evidence](table4/EVIDENCE.json) | [Matched inference stages](table4/matched_inference_stages.csv) | [Protocol](PROTOCOL.json)
