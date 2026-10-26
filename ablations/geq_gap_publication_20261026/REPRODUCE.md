# Reproducibility

## Figure-only reproduction

From the repository root, run `python ablations/geq_gap_publication_20261026/reproduce_figures.py` (numpy, scikit-learn, matplotlib). No CUDA, private paths, checkpoints, or feature files are needed. Evidence is gzip-compressed JSONL with exact query IDs. Original official G-mIoU evaluator snapshots are unchanged. Output includes PDF/SVG/PNG, exact score summaries, density bins and all 101-threshold curves for all three methods. The script checks all 303 method/threshold pairs against the official evaluator; ties use strict score > threshold.

## Training / checkpoint evaluation

The implementation reuses these repository directories:

- `ablations/end_to_end_geq_pv_20261026/{adapters,model,run,evaluation,common,pv,independent_metrics}.py`: native factories, attached GEQ, original native criterion and official export.
- `ablations/geq_mean_separation_20261026/train.py`: shared pretrained constructor, native correctness preflight, initial tensor equality to E1.
- `ablations/geq_gap_search_20261026/search_train.py`: parameterized attached class-mean GAP; fixed 4 epochs; Val-only selection.

Download the original GMR checkpoint from the Release, then set `GMR_FLASH_CHECKPOINT=/absolute/path/flashvtg_gmr_original_initialization.ckpt`. Set `GMR_ASSETS_DIR=/absolute/path/to/Soccergmr`, containing slowfast/, clip/, clip_text/. These external offline encoders are not part of the training graph. Native dependencies include PyTorch, nncore, easydict and the repository's FlashVTG requirements. The exact splits are in `ablations/end_to_end_geq_pv_20261026/data/`.

On a fresh checkout (existing experiment output directories must be moved aside rather than overwritten):

```bash
CUDA_VISIBLE_DEVICES=0 python ablations/geq_gap_search_20261026/search_train.py --trial GAP_m60_w200 --preflight-only
CUDA_VISIBLE_DEVICES=0 python ablations/geq_gap_search_20261026/search_train.py --trial GAP_m60_w200
```

The train entry deliberately starts from the original pretrained checkpoint, not the released fine-tuned weights; native model and GEQ initial tensors must equal the saved E1 initial hashes. No old optimizer state is restored. The published historical original-GAP loader hashes serve as reference; all four epoch orders must match. Preflight gradient checks and optimizer smoke steps are discarded before formal fresh initialization. All original trainable modules are allowed to update; unused native MR-only saliency / text-position branches are recorded rather than falsely claimed to receive gradients.

For inference, construct `Adapter('flashvtg','cuda:0')`, wrap `JointGMR(adapter.base,'flashvtg',pv=False)`, load `torch.load(weights, map_location='cpu', weights_only=False)['model']` with `strict=True`, then call the existing `evaluation.predict(adapter,model,adapter.dataset('test'))`. `pred_exist_score` is solely sigmoid(GEQ logit); preserve native soft calibration and NMS. Do not restore native existence as the final presence predictor. Using the original initialization file to construct the adapter before strictly overwriting with released weights is an architecture-building dependency, not training from E1.

The full checkpoint includes optimizer/scheduler; the compact checkpoint includes the identical model tensors and native factory configuration. SHA256 and byte counts are in CHECKPOINT_MANIFEST.json. Pre-existing local absolute paths in historical configs are provenance, not required portable destinations; use the environment overrides above.

## Selection / evidence boundaries

Six new trials: margin={0.4,0.6} × weight={0.5,1,2}; the existing original GAP margin=.5/weight=1 is a reused candidate. All use seed2026 and epoch4. Choose by exact Val AUROC, then AUPRC/S/mAP/lexical trial ID. `tables/SELECTION.json` was locked before evaluating the selected new model's Test. The user subsequently requested Test for every trial; those are fully disclosed in `tables/search_all_test.csv`, without changing the prior selection. The selected model is not claimed to maximize every Test metric. S and mean gap are algebraically equivalent; improvements are not independent evidence. No statistical significance or cross-seed robustness is claimed.
