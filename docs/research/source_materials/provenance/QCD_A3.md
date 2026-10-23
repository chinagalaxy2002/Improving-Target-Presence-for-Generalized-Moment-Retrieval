# A3: Multi-aware Query-aware Context Diversification

A3 applies CVA's Query-aware Context Diversification (QCD) to the existing S4-Fusion model. It changes only loaded training video features. The architecture, trainable parameter set, E0 initialization and original losses stay as in S4-Fusion.

## Method and adaptation

- Scope: original WorldCup2022 Train positives, both Single (720) and Multi (713). All 4,138 original Train annotations remain; other-source and no-target samples are not augmented.
- Activation probability: 0.5. Replacement ratio: 0.30. Context: one clip on each side of **every** GT window, merged by Boolean union.
- Protection uses the full JSONL `relevant_windows`, including the sixth window when `max_windows=5` truncates span supervision. Intersecting clips are `floor(start/2):ceil(end/2)` with clipping to effective feature length. Overlapping windows/context are not counted twice.
- Replace `min(eligible_background, round(0.30 × non_GT_background))` positions. Python `round` uses ties-to-even. Destination clips are sampled without replacement; source clips with replacement. Empty pools skip augmentation, without donor retry.
- Pick one uniformly sampled, distinct WorldCup Train video. Similarity is computed against the **recipient query**, not the donor's annotation. Threshold interval is inclusive.
- All WorldCup Train clips × unique exact query strings enter dataset similarity statistics. GT membership is the union of annotated same-video/same-query windows; other pairs are non-GT **as annotated**, not proven semantic negatives. Negative-query annotations contribute non-GT pairs. No Val/Test feature or label enters thresholds or donor sampling.
- CLIP sentence representation uses the original Lighthouse cache's final non-padding EOT token after `ln_final`, multiplied by the original ViT-B/32 `text_projection`, then L2 normalized. The cached 77-token sequence is used before the model's 40-token crop. Do not use averaged token states as projected sentence embeddings.
- Thresholds: `Percentile_10(S_non)=0.21965927`, `Percentile_60(S_gt)=0.26803598`. These are estimated similarities, not fixed percentile values supplied as similarities.
- Copy SlowFast2304 and CLIP512 from the same original donor clip into the batch's first 2,816 dimensions. Keep the recipient TEF2, padding, query, presence target and all original supervision. The immutable original Train bank is never overwritten.

## Reproduce

Use the root README's external-path setup. The pretrained CLIP file is required only to compute its projection; it must match the cached CLIP feature space. No downloads or machine-specific paths are embedded in the published scripts.

```bash
python code/test_qcd.py
python code/build_qcd_cache.py --clip-model /path/to/ViT-B-32.pt
python code/train_qcd.py --run-name A3_new --device cuda:0
```

The default is 10 epochs, no early stopping, 5,180 optimizer updates, all epoch checkpoints saved. Default training uses Train/Val only. Initialization seed=2026, DataLoader seed=2026, training RNG seed=102026, separate QCD NumPy Generator seed=302026. The independent generator avoids consuming model/dropout RNG.

To reproduce the **explicitly requested exploratory per-epoch Test diagnostic**:

```bash
python code/train_qcd.py --run-name A3_test_diagnostic --device cuda:0 --epochs 10 --test-from 4
```

This saves `epoch01.ckpt`–`epoch10.ckpt`, including optimizer, scheduler, global/loader/QCD RNG and QCD counters, and Test outputs in `test/epoch04/`–`test/epoch10/`. Test evaluation preserves training RNG; same-epoch Val supplies calibration. Test is not used by the implemented Val selection rule. No automatic resume is implemented.

## Published result and limitations

The published checkpoint is **epoch 4**, the original Val omni_spectrum selection. All epochs 4–10 were subsequently inspected on Test at the user's request; this is an exploratory comparison, not a new untouched Test evaluation. Model binary and external features remain outside Git; the exact epoch-4 checkpoint SHA256 and configuration are in [QCD_A3.json](QCD_A3.json). Gzip prediction archives and complete evaluation reports are in `results/S4Fusion_QCD_A3_ep04/`.

At epoch 4 on all Test samples, A3 versus unaugmented S4-Fusion: G@1 53.67 vs 53.94, mAP 26.37 vs 25.84, raw mAP 25.49 vs 24.95, N-acc 95.12 vs 95.93, and AUROC 75.77 vs 75.15. On WorldCup-only Test, AUROC is 59.85 vs 58.80; recall at empirical N-acc≥95% is 10.67 vs 9.76. These single-seed differences do not establish that QCD solves high-specificity positive rejection.

## Attribution

Moon et al., *CVA: Context-aware Video-text Alignment for Video Temporal Grounding*, [arXiv:2603.24934, §3.1](https://arxiv.org/html/2603.24934). Query-aware context diversification is the original method. WorldCup-only statistics/donors, full multi-GT protection, synchronized SlowFast+CLIP replacement with retained TEF, activation probability, explicit rounding/pool behavior and A3 configuration are this repository's adaptation, not a claim of the authors' original implementation.
