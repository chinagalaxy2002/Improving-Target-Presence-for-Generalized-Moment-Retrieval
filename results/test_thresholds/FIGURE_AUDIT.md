# Figure design and integrity audit

## 1. Figure type

Experimental-results / parameter-sensitivity figures. The requested design follows the supplied five-backbone extension of the reference: one row per backbone and G-mIoU@1/@3 as two columns.

## 2. Chosen design

Line plots match the continuous existence-threshold axis. A bar chart would hide threshold dependence, while a heatmap would make the individual operating curves harder to compare. The two methods in the current branch scope are GMR and GEQ Full; archived alternative methods are not introduced into the main comparison.

## 3. Layout

Two main canvases, Standard and Full, each 12.4 × 18 inches with five rows and two columns. Backbone order is Moment-DETR, EaTR, QD-DETR, CG-DETR, Sim-DETR. Both canvases share the same x/y limits. Each row has its own mAP legend and three changes relative to its matching GMR. Additional PDFs provide a page per backbone and a two-page main-figure bundle.

## 4. Labels and visual encodings

GMR uses grey solid lines and circle markers; GEQ Full uses green dash-dot lines and triangles. Shapes and line styles distinguish methods without colour. Deltas include explicit signs: positive green, negative red, zero grey. Each axis names the metric and scale; legends report mAP in percent; deltas use percentage points. Guides at .4/.5/.6 correspond to the fixed existence thresholds. Oracle envelope and its GT-assisted meaning are defined in the caption.

## 5. Tool and artifacts

Matplotlib script `scripts/plot_test_thresholds.py` reads the portable `PLOT_DATA.json`. PNG files support GitHub README display; PDF/SVG exports contain vector curves/text. The figures are not edited screenshots and no generative image tool is used. All raw points and per-method oracle values are included in CSV.

## 6. Universal rule audit

- Vector format: **pass**, PDF and SVG exports; SVG contains no raster `<image>` element.
- Typography: **pass for standalone/README viewing**, source labels and annotation sizes 9–15 pt, with 220-DPI PNG and zoomable vector copies. A five-row overview should not be reduced to a small paper column; use the per-backbone pages and adjust final publication sizing if needed.
- Colour-blind accessibility: **pass**, colour plus line style plus marker, and signed deltas.
- Self-contained meaning: **pass**, title, split sizes, train split, historical stopping protocol, units, checkpoint selection, oracle definition and Full coverage/input limits are shown or linked.
- Honest axes: **pass**, shared x range 0–1 and shared y range 0–0.7; no hidden axis breaks, clipping or smoothed curves.
- Chartjunk: **pass**, light grids, no decorative fills, gradients or 3D effects.
- Uncertainty: each line represents one fixed checkpoint, not repeated seeds; no fabricated confidence intervals.

## 7. Integrity checks

Both complete PNG figures were opened and visually inspected after generation. All ten panels, legends, oracle labels, delta text, x labels and footnotes are visible without overlaps. The Standard figure shows the Moment-DETR negative ΔmAP in red; the Full figure shows the Moment-DETR zero delta in grey. The displayed range preserves endpoint behavior at τ=0 and τ=1.

The collector verifies 20 complete runs, 1,036/2,970 unique qids per run, immutable checkpoint hashes, paired checkpoint identity across splits, official mAP/brief recomputation, G1/G3 spot checks at 0/.4/.5/.6/1, and independently gated GT-presence oracle values. Standard fixed-point metrics exactly reproduce the original formal records. No Test threshold or checkpoint is chosen.

The supplied example's four-backbone Full plot used different GMR initializations. Its numbers were not copied into the new matched comparison. Missing paired Full results were evaluated from the exact five historical Standard checkpoint pairs using repository-owned native source. The old 75-token profile is explicitly retained for comparability; it is not the new trainer's 128-token contract.

## 8. Severity

**0 critical, 0 major, 0 minor display defects observed at the inspected standalone size.** Historical-protocol and Full-coverage limitations are scientific scope labels, retained in the main figures and README. Universal effectiveness across thresholds, seeds or unseen backbones is not claimed.
