# Figure design and inspection audit

1. **Figure type:** experimental results. The data compare two methods on six fixed backbones in two evaluation settings.
2. **Paradigm:** grouped bars for paired method comparison. Raw scores are shown directly; percentage annotations alone would hide the baseline scale. A second figure repeats the encoding for exact-threshold sensitivity.
3. **Layout:** 7.2 × 3.8 inches, two side-by-side dataset panels, six paired groups per panel. GMR is always left and GEQ Full right. No arrows. Both panels have the same 0–27 axis limits and 5-point ticks.
4. **Labels:** full backbone identities are represented by split-line labels. Every axis names G-mIoU@1 and its 0–100 units. Above-pair labels are absolute GEQ−GMR percentage-point changes. Panel notes report the equal-backbone mean gain and relative gain. Footer identifies the exploratory subset and threshold rule.
5. **Tool:** reproducible Matplotlib script, `scripts/report_high_retention.py`. Grey GMR bars/square markers and Okabe–Ito blue GEQ bars/circle markers provide colour and shape encoding.
6. **Universal audit:** PDF/SVG vectors PASS; no raster image elements in SVG PASS; 8–10pt text at the native full-width size PASS; zero-origin common axes PASS; colour-blind-safe blue/grey with square/circle encoding PASS; no 3D, gradients or decorative effects PASS. Captions state the result, define units, and explain the Val-selected thresholds. Native-width exports should be used rather than reducing both panels to one narrow column.
7. **Integrity gates:** 1–4 and 7 PASS. Gate5 PASS after viewing both complete PNG previews. Gate6 (motivated-example/Introduction alignment) is not applicable. All displayed values come from verified CSV rows, with the same six-model order in both figures. Single-checkpoint comparisons do not display invented seed error bars.
8. **Severity summary:** 0 CRITICAL, 0 MAJOR, 0 MINOR. Both PRIMARY.png and EXACT_THRESHOLDS.png were visually inspected; captions, labels, mean annotations and bar differences are legible and not clipped.

This follows the figure-designer skill's experimental-results, universal-design and integrity guidance. It is a design audit, not an independent research review or statistical-significance result.
