# Reference-style curve figure review

- Type: experimental threshold-sensitivity curves; same 11-row × 2-column structure as the supplied reference.
- Dataset separation: STANDARD_TEST contains only Standard; FULL_TEST contains only Full-Decodable. Both include all 11 backbones, left G1/right G3.
- Encoding: grey solid circles for GMR baseline; green dash-dot triangles for GEQ+aux. Hollow diamonds mark each method's Val ≥95% retention operating point from the 0.01 grid.
- Labels: method mAP and τ95 per row; signed changes in mAP and G1/G3 at τ95, with gains and losses retained. Changes are in percentage points; the curve axis uses 0–1 units.
- Oracle: recomputed from each current method's fixed candidates and GT presence. The horizontal envelope is the maximum of the two model-specific scores, and is explicitly analysis-only.
- Layout: matching axes throughout Standard/Full; non-overlapping legends, deltas and footnotes. Native vector PDF/SVG, PNG, combined two-page PDF and 11-page per-backbone PDFs are included.
- Numerical integrity: 8,888 curve values match the existing verified CSV, 88 operating points match the retention report, and 88 Oracle G1/G3 values match the official implementation. All 44 prediction files are hash-checked.
- Visual inspection: full Standard figure, Standard top rows and Full bottom rows inspected; labels, markers, Oracle annotations, legends, signed changes and footnotes are legible at native export size. Use per-backbone PDF pages when reducing for manuscript placement.
- Integrity gate: curve paradigm matches threshold sensitivity; concrete labels, dual encoding, honest zero-origin axes and reproducible Matplotlib exports pass. No critical or major visual issue found at native size.
