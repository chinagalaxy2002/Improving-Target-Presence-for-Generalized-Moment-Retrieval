# All-backbone retention figure review

- Both figures include all 11 published backbones, in the same order as the main Test summary; Standard is left and Full is right.
- Figure A uses the 0.01 Val grid; Figure B uses all observed Val score breakpoints. Every method uses its own Val threshold, fixed across both Tests.
- Horizontal paired bars retain all gains and declines. Grey squares identify GMR baseline; blue circles identify GEQ+aux. Both figures use the same zero-origin axis range.
- Numeric labels give G1 scores rather than only gains. All 88 bars/labels match METRICS.csv; full mAP/G1/G3 and threshold/retention tables cover all 11 backbones and both Tests.
- Native vector PDF/SVG and GitHub PNG exports are provided. Text bounds and footer separation are checked in code; both rendered PNGs were visually inspected.
- Figure captions and README explain auxiliary-trained Stage 1, reused Moment-DETR, and Val-only threshold selection. No Test-based backbone filtering or threshold tuning.
- Interpretation includes BAM-DETR and LD-DETR declines on both Tests and the QD-DETR Full decline in the breakpoint analysis.
- Integrity check: suitable experimental-results chart type, concrete labels, reproducible plotting, dual encoding, honest axes; no critical or major issue found at native export size.
