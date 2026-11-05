# Adding a DETR backbone

The common CLI owns data, optimizer, checkpointing, validation selection and post-training evaluation. A backbone implements native moment localization and its original GMR baseline. GEQ Full is a shared head attached to projected text and encoder video features.

1. Add a self-contained, licensed native source snapshot under `src/soccergmr/backbones/vendor/<name>/`. Record upstream provenance and file hashes in `SOURCE_MANIFEST.json`. Include transitive imports. Do not import an external checkout or the archive.
2. Add `configs/backbones/<name>.json` containing all architecture options and no machine paths. Add the matching experiment template in `configs/experiments/`.
3. Register a `Spec` in `backbones/registry.py`: builder, feature order, text masking, encoder video offset and number of encoder passes. `build()` returns the native model and common GMR loss wrapper. Preserve the baseline's trained existence head and native losses; do not substitute a different pooling rule while claiming the same baseline.
4. Adapt forward arguments in `registry.forward()` only if required. Common inputs are `src_txt`, `src_txt_mask`, `src_vid`, `src_vid_mask`. Masks are float 0/1; implementations may cast to bool. CG-DETR also receives metadata and training targets, without synthetic negative pairing.
5. Expose `pred_logits` (B × Q × 2, foreground channel 0), `pred_spans` (B × Q × 2, center/width normalized to true duration), and native existence logits. `encoder_root()` must expose `input_txt_proj` and `transformer.encoder.layers`. Encoder hooks return sequence-first features with a declared video offset. Sim-DETR uses the first of its two encoder passes, matching the archived GEQ attachment point.
6. Add a native loss adapter if needed. It must handle a positive/null mixture, multiple GT windows, all-null batches and padded variable lengths without NaN. Auxiliary outputs must be sliced consistently when native losses see only positives. Padding must never enter attention BCE as a negative probability.
7. Run the bounded real-data verifier in a fresh process. Check finite gradients for native and GEQ parameters, exact epoch-boundary resume against uninterrupted training, changed native tensors during GEQ Full, strict state loading, complete prediction coverage and both Test layouts including a video longer than 75 tokens. Then run formal Standard training and report its own results.

Native snapshots may share top-level `models`, `utils` or `misc` package names. The registry permits one backbone family per process and rejects namespace contamination. Launch each experiment as a separate CLI process. There is no need to copy or edit the trainer for each new backbone.

Selection is fixed: GMR uses Val mAP; GEQ Full uses the mean Val G-mIoU@1 at `.4/.5/.6`, with ten consecutive non-improving epochs. Test scores cannot justify changing learning rates, stopping epochs, thresholds or the selected checkpoint.

The verifier checks that imported project modules are from its checkout. Python dependencies may come from the environment; labels/features may come from configured external paths. A clean source copy with no archive must still train and evaluate.
