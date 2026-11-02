# Figure audit: frozen Shared / Multi + GEQ

## Four-curve reference layout

The four backbone figures each contain exactly GMR baseline, GMR baseline + GEQ,
multi + GEQ, and shared + GEQ. Their six panels follow the main branch reference: threshold
versus G-mIoU@1, threshold versus G-mIoU@3, G-mIoU@1 difference from GMR at the
same threshold, negative accuracy versus positive recall, rejection F1, and
G-mIoU@1 at fixed thresholds 0.4/0.5/0.6. A separate two-panel vector export
provides the G-mIoU@1 / @3 curves for each backbone.

All absolute performance axes span 0–1; the difference panel uses signed 0–1
units. mAP is reported in percent in the shared legend, README and CSV and uses
the official positive-query localization calculation without presence gating.
All 16 model series cover the same 1,036 Test queries; the 12 GEQ curves match
their archived 101-point sweeps exactly, and every model's full metrics were
recomputed from its frozen predictions and matched the archived metrics.
Empirical rejection/retention curves use every unique score, with the three
prespecified thresholds marked. Rendering does not load weights or select
thresholds. Four additional baseline + GEQ controls were independently trained
for fixed epoch 4 from the exact displayed baselines, with unchanged localization
and mAP verified. EaTR's completed historical baseline seed 2026 is labeled separately from
the GEQ seed 2023; source hashes and protocol metadata are retained.

The two-column / three-column panel exports use native vector graphics, a
consistent gray/orange/green/blue palette, and distinct line styles and marker shapes.
Legends explicitly identify each model and its mAP. Standalone exports were
inspected for axis and legend readability, missing labels and clipping.

## Earlier paired comparisons

The fixed-threshold G-mIoU@3 improves in six of eight comparisons and decreases
in both Moment-DETR comparisons. All eight AUROCs decrease slightly; localization
outputs and metrics are exactly preserved. This is an exploratory single-seed,
fixed-four-epoch comparison on the same SoccerGMR Standard Test split.

The overview has four backbone rows and Shared / Multi columns. Every panel
uses the same before/after color, line style and marker combination. Both axes
span 0–1; each curve contains 101 original threshold measurements without
smoothing. The gray reference is the score obtained by rejecting all queries,
492 / 1,036 = 0.4749. It does not represent a trained comparison model.

The separate annotated heatmap normalizes only the two entries within each
backbone/method row. Color identifies that row's lower/higher value, not the
magnitude of gain or an absolute score across rows. Printed values remain
G-mIoU in 0–1 units. Equal rows use the middle color and missing rows are gray.

PDF and SVG are native vector exports; PNG is a preview. Each backbone has its
own two-panel vector figure; use those panels when paper scaling would make the
eight-panel overview too small. Labels are 9–12 pt at the native output size.
Legend text, marker shapes and line styles supplement color. No seed variance,
confidence intervals or statistical significance are fabricated.

The overview and row-colored heatmap were visually inspected for clipping,
readable legends, honest axes and consistent before/after labels. The CPU audit
verified all eight source checkpoints, four training epochs, frozen states,
strict head loading, complete Test predictions, unchanged localization metrics,
baseline metric reproduction and all 101-point 0–1 sweeps. The results include
the two decreases; neither checkpoints nor thresholds were selected on Test.

Audit: 0 critical, 0 major. Final paper insertion must preserve at least 8 pt
text; use the per-backbone files when needed.
