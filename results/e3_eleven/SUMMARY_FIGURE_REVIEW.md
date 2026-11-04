# Summary figure review

- Figure type: experimental results; paired G-mIoU@1 comparison on two Test datasets.
- Paradigm: one horizontal grouped-bar panel per dataset. A single metric avoids mixing ungated mAP with the Val-selected existence-gate protocol.
- Layout: Standard and Full use the same nine-backbone order and the same 0–45 axis. Exact thresholds and values remain available in the README and CSV.
- Encoding: neutral grey plus square for GMR baseline; colour-blind-safe blue plus circle for GEQ+aux. Method order is fixed and also distinguishable in greyscale.
- Labels: 36 numeric labels match unrounded CSV bar widths; the 18 GEQ+aux labels also state the paired G1 change.
- Selection disclosure: the subtitle and footer say that the 9/11 subset was selected after Test inspection for positive G1 change on both Tests. Moment-DETR and LD-DETR are named as omitted; all 11 remain in the source CSV.
- Threshold disclosure: every backbone/method independently selects the largest 0.01-grid Standard-Val threshold retaining at least 220/255 positives, then applies that same threshold to both Tests.
- Exports: native vector PDF and SVG plus PNG for GitHub. Source font sizes are at least 8.8pt; text bounds and footer/axis-label separation are checked in code.
- Axes: zero origin and a common honest upper limit just beyond the maximum value; score units and higher-is-better direction are explicit.
- Scope: single seed and exploratory post-Test selection; no uncertainty bars or independent-confirmation claim is invented.
- Visual inspection: both native PNGs were inspected; titles, row labels, values, deltas, legend and footer are readable with no overlaps.
- Integrity gate: experimental-results type, grouped bars, concrete labels, reproducible Matplotlib code, dual encoding, vector output and honest axes pass. No critical or major issue remains at native export size.

See SUMMARY_PLOT_AUDIT.json for original-input and figure hashes.
