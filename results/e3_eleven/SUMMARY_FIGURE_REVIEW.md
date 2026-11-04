# Summary figure review

- Figure type: experimental results; paired method comparison on two Test datasets.
- Paradigm: horizontal grouped bars, two panels per dataset (mAP and mean G1).
- Layout: Standard and Full figures use the same backbone order and matching axes per metric. Exact-value table remains available in the README.
- Encoding: neutral grey squares for GMR baseline, blue circles for GEQ+aux; method order is fixed. All 11 backbones and decreases remain visible.
- Labels: 88 numeric labels match CSV values to two decimal places; plotted bar widths match original unrounded CSV values. G1 explicitly uses fixed 0.4/0.5/0.6 thresholds.
- Exports: native vector PDF and SVG plus PNG for GitHub. Source font sizes >=9pt; text bounds and footer/axis-label separation checked by the plotting script.
- Axes: zero origin; mAP 0–29 and G1 0–66, matching between Standard and Full. Axes show score units and higher-is-better direction.
- Caption: main README reports improvement counts and exceptions; figures define the auxiliary-trained baseline and reused Moment-DETR results.
- Scope: single seed; no invented uncertainty bars. High-retention calibration is explicitly distinguished from fixed-threshold mean G1.
- Visual inspection: Standard and Full PNGs inspected; title, row labels, values and footer are readable with no overlaps after layout adjustment.
- Integrity gate: experimental-results type, bar encoding, concrete labels, reproducible Matplotlib code, dual encoding and honest axes all pass. No critical or major issue found at native export size.

See SUMMARY_PLOT_AUDIT.json for original-input and figure hashes.
