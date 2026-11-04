# Retention-target scan and selected-subset figure review

- Scan overview shows the complete 0–100% named-target range, mean G1 gain and number of positive backbones on both Tests; both threshold candidate rules are displayed with color and line-style differences.
- Standard / Full heatmaps include all 11 backbones, focus explicitly on 80–100%, and use the same symmetric zero-centered color scale. Numeric annotations and full CSV values preserve losses.
- Selected-subset figures show nine rows, G1/G3 columns, fixed model curves, recomputed-from-original-candidates Oracle references, mAP, and signed changes at the shared 86% Val constraint.
- Figure titles and captions explicitly identify the nine-backbone subset as selected after Test scanning and name the excluded Moment-DETR and LD-DETR. This is exploratory presentation, not independent confirmation.
- mAP losses remain visible. All 11-backbone scan data and original 95% figures are preserved and linked.
- Both selected PNGs and scan figures were visually inspected, including the Full final rows; labels, legends, footnotes and signed changes are legible at native size. Vector PDF/SVG and per-backbone PDFs are supplied.
- Numerical QA: 1,144 representative official metric checks; all 67,584 dense gain calculations checked independently by prefix-sum and direct masking; all 44 original 95% rows matched; 72 selected-subset working points matched the scan.
- Charts use appropriate line/heatmap encodings, concrete labels, native vector exports, honest axes and explicit scope. No critical or major issue found at native export size.
