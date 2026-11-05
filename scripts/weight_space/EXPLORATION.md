The expanded run reads https://rikiyat.github.io/blog/weight-space-programming/ as inspiration, particularly its MLP output-column intervention. It adds 588 analytic head hypotheses, all created on discovery videos, screened on Train development videos, and eight preselected families evaluated on reused Val. This is adaptive development and does not establish confirmatory generalization. The selected sparse head fails to generalize. Full failure records are retained.

Run from `/home/guoxiangyu/SoccerGMR-GEQ`, using fresh output paths:

```bash
EXPERIMENT_BASE=/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/weight_space_flashvtg_e1
EXPLORATION_DIR=/absolute/path/to/new/exploration
E1_CHECKPOINT=/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_standard/d_gmr_two_stage/stage_1_geq/runs/flashvtg/E1/epoch_004.ckpt
LEGACY_CODE=/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_standard/d_gmr_two_stage/stage_1_geq/code
OPENBLAS_NUM_THREADS=2 python scripts/weight_space/explore_heads.py --root "$EXPERIMENT_BASE" --checkpoint "$E1_CHECKPOINT" --output "$EXPLORATION_DIR"
python scripts/weight_space/build_exploration_controls.py --root "$EXPERIMENT_BASE" --exploration "$EXPLORATION_DIR"
python scripts/weight_space/validate_edits.py --legacy-code "$LEGACY_CODE" --train "$EXPERIMENT_BASE/train" --val "$EXPERIMENT_BASE/val" --analysis "$EXPERIMENT_BASE/analysis" --edits "$EXPLORATION_DIR/edits" --output "$EXPLORATION_DIR/validation"
python scripts/weight_space/validate_edits.py --legacy-code "$LEGACY_CODE" --train "$EXPERIMENT_BASE/train" --val "$EXPERIMENT_BASE/val" --analysis "$EXPERIMENT_BASE/analysis" --edits "$EXPLORATION_DIR/controls" --output "$EXPLORATION_DIR/control_validation"
python scripts/weight_space/verify_native.py --legacy-code "$LEGACY_CODE" --checkpoint "$E1_CHECKPOINT" --edits "$EXPLORATION_DIR/edits" --val "$EXPERIMENT_BASE/val" --output "$EXPLORATION_DIR/native_check" --device cuda:1
for layer in 0 1; do
  python scripts/weight_space/causal_ffn.py --legacy-code "$LEGACY_CODE" --checkpoint "$E1_CHECKPOINT" --train "$EXPERIMENT_BASE/train" --analysis "$EXPERIMENT_BASE/analysis" --layer "$layer" --columns 32 --strengths 0.5 1.0 --output "$EXPLORATION_DIR/ffn_layer$layer" --device cuda:0
done
python scripts/weight_space/report_exploration.py --root "$EXPLORATION_DIR"
```

No FFN checkpoint is permitted unless the corresponding activation intervention passes. `sweep.json` and `directions.npz` preserve every virtual intervention; only shortlisted edits and controls are materialized as checkpoints. All native output checks compare against the unedited E1 model at identical inputs. Candidate exports for official metrics remain fixed as documented in README.md.

When layer 1 passes the causal gate, materialize and validate actual FFN weights:

```bash
python scripts/weight_space/edit_weights.py --checkpoint "$E1_CHECKPOINT" --analysis "$EXPERIMENT_BASE/analysis" --train "$EXPERIMENT_BASE/train" --ffn-causal "$EXPLORATION_DIR/ffn_layer1/causal.json" --ffn-only --output "$EXPLORATION_DIR/ffn_edits"
for candidate in ffn_a0.5 ffn_a1; do
  mkdir -p "$EXPLORATION_DIR/ffn_forwards/$candidate"
  for split in train val; do
    python scripts/weight_space/infer_fixed_candidates.py --legacy-code "$LEGACY_CODE" --reference "$E1_CHECKPOINT" --edited "$EXPLORATION_DIR/ffn_edits/$candidate/edited.ckpt" --split "$split" --output "$EXPLORATION_DIR/ffn_forwards/$candidate/$split" --device cuda:0
  done
done
python scripts/weight_space/validate_edits.py --legacy-code "$LEGACY_CODE" --train "$EXPERIMENT_BASE/train" --val "$EXPERIMENT_BASE/val" --analysis "$EXPERIMENT_BASE/analysis" --edits "$EXPLORATION_DIR/ffn_edits" --ffn-predictions-root "$EXPLORATION_DIR/ffn_forwards" --output "$EXPLORATION_DIR/ffn_validation"
python scripts/weight_space/verify_native.py --legacy-code "$LEGACY_CODE" --checkpoint "$E1_CHECKPOINT" --edits "$EXPLORATION_DIR/ffn_edits" --val "$EXPERIMENT_BASE/val" --output "$EXPLORATION_DIR/ffn_native" --device cuda:0
python scripts/weight_space/finalize_exploration.py --root "$EXPLORATION_DIR"
```

The FFN strength list above matches this run's passing interventions; never hardcode a passing result for another dataset or checkpoint. The initial run materialized all head controls as well under `ffn_all_edits`, then used a baseline-plus-FFN index in `ffn_edits`; `--ffn-only` now supplies that same scope directly.
