# FlashVTG E1 weight-space presence repair

This experiment explicitly uses the historical FlashVTG E1 implementation and its fixed epoch-4 checkpoint, not the current branch's ten-patience training protocol. The current branch does not register FlashVTG. The scripts import the legacy E1 runtime by its explicit path; neither training protocol nor the official evaluator is modified. The original GMR and E1 checkpoints remain immutable.

Only Train and Val are loaded. Train is divided by SHA-256 of video ID into discovery and development (60/40); Val video IDs must be disjoint from Train. Hard null means a top candidate score at least the 75th percentile of discovery null queries. Presence acceptance is strictly `score > 0.5` for categorization and confusion; official G-mIoU is additionally evaluated at 0.4 and 0.6. Top-1 IoU is against any GT window using the raw pre-existence candidate order. Discovery results are exploratory; development determines selection; Val is an independent report and never selects an edit.

The legacy export scales candidate confidence according to existence, so rerunning that export unmodified would change candidate scores even when `base.*` is identical. `infer_fixed_candidates.py` supplies the original E1 head's probability to the unmodified exporter for candidate score fusion and publishes the edited head's probability as the GEQ decision. The metric experiment equivalently freezes the original E1 candidate export. The bridge also supports a GEQ FFN edit by running the original GEQ and edited GEQ separately. `verify_native.py` checks native output tensors on all Val queries for each edit; `verify_bridge.py` verifies exported candidates and official mAP. The FFN path is gated by an independent activation intervention; a failing causal test produces no FFN edit. FFN validation requires actual full-model Train and Val forwards, rather than the analytic head shortcut.

All outputs must be new directories. The commands below use the original local assets; adapt paths and select GPUs as needed. Installation requires PyTorch, NumPy, scikit-learn, matplotlib and the FlashVTG legacy dependencies (`nncore` included). `python tests/weight_space/test_core.py` runs the pure diagnostic and intervention algebra tests, without gradients.

```bash
cd /home/guoxiangyu/SoccerGMR-GEQ
LEGACY_CODE=/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_standard/d_gmr_two_stage/stage_1_geq/code
E1_CHECKPOINT=/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_standard/d_gmr_two_stage/stage_1_geq/runs/flashvtg/E1/epoch_004.ckpt
FEATURE_ROOT=/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval/Soccergmr
export GMR_ASSETS_DIR="$FEATURE_ROOT"
EXPERIMENT_DIR=/absolute/path/to/a/new/experiment
mkdir "$EXPERIMENT_DIR"
python tests/weight_space/test_core.py
python scripts/weight_space/audit_assets.py --legacy-code "$LEGACY_CODE" --features "$FEATURE_ROOT" --output "$EXPERIMENT_DIR/feature_audit.json"
python scripts/weight_space/extract_activations.py --legacy-code "$LEGACY_CODE" --checkpoint "$E1_CHECKPOINT" --output "$EXPERIMENT_DIR/train" --split train --device cuda:0
python scripts/weight_space/extract_activations.py --legacy-code "$LEGACY_CODE" --checkpoint "$E1_CHECKPOINT" --output "$EXPERIMENT_DIR/val" --split val --device cuda:0
python scripts/weight_space/analyze_directions.py --train "$EXPERIMENT_DIR/train" --val "$EXPERIMENT_DIR/val" --output "$EXPERIMENT_DIR/analysis"
python scripts/weight_space/causal_ffn.py --legacy-code "$LEGACY_CODE" --checkpoint "$E1_CHECKPOINT" --train "$EXPERIMENT_DIR/train" --analysis "$EXPERIMENT_DIR/analysis" --output "$EXPERIMENT_DIR/causal_ffn" --device cuda:0
python scripts/weight_space/edit_weights.py --checkpoint "$E1_CHECKPOINT" --analysis "$EXPERIMENT_DIR/analysis" --train "$EXPERIMENT_DIR/train" --output "$EXPERIMENT_DIR/edits"
python scripts/weight_space/validate_edits.py --legacy-code "$LEGACY_CODE" --train "$EXPERIMENT_DIR/train" --val "$EXPERIMENT_DIR/val" --analysis "$EXPERIMENT_DIR/analysis" --edits "$EXPERIMENT_DIR/edits" --output "$EXPERIMENT_DIR/validation"
python scripts/weight_space/verify_native.py --legacy-code "$LEGACY_CODE" --checkpoint "$E1_CHECKPOINT" --edits "$EXPERIMENT_DIR/edits" --val "$EXPERIMENT_DIR/val" --output "$EXPERIMENT_DIR/native_check" --device cuda:0
python scripts/weight_space/verify_head_math.py --legacy-code "$LEGACY_CODE" --checkpoint "$E1_CHECKPOINT" --edits "$EXPERIMENT_DIR/edits" --output "$EXPERIMENT_DIR/head_math" --device cuda:0
python scripts/weight_space/infer_fixed_candidates.py --legacy-code "$LEGACY_CODE" --reference "$E1_CHECKPOINT" --edited "$EXPERIMENT_DIR/edits/direction_a0.1/edited.ckpt" --split val --output "$EXPERIMENT_DIR/bridge_check" --device cuda:0
python scripts/weight_space/verify_bridge.py --legacy-code "$LEGACY_CODE" --reference "$EXPERIMENT_DIR/val" --bridge "$EXPERIMENT_DIR/bridge_check"
python scripts/weight_space/verify_analytic.py --reference "$EXPERIMENT_DIR/val" --bridge "$EXPERIMENT_DIR/bridge_check" --delta "$EXPERIMENT_DIR/edits/direction_a0.1/delta.pt" --output "$EXPERIMENT_DIR/bridge_check/analytic_verification.json"
python scripts/weight_space/report.py --root "$EXPERIMENT_DIR" --output "$EXPERIMENT_DIR/report"
```

To request FFN checkpoint generation, supply `--ffn-causal "$EXPERIMENT_DIR/causal_ffn/causal.json"` to a fresh `edit_weights.py` output directory. The command refuses if the activation test has no passing strength. For each causally gated FFN candidate, run `infer_fixed_candidates.py` once per split with output paths `$EXPERIMENT_DIR/ffn_forwards/<candidate>/train` and `.../val`. Then pass `--ffn-predictions-root "$EXPERIMENT_DIR/ffn_forwards"` to `validate_edits.py`; it verifies the forward hashes and frozen candidates before scoring. Run `verify_native.py` on the combined directory to check native outputs. `verify_head_math.py` is only for head edits, and refuses FFN candidates.
