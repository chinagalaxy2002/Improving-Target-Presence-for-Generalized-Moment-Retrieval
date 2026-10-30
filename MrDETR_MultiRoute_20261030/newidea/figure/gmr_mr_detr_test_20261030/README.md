# GMR × Mr. DETR test evaluation figure

This directory contains the reproducible evaluation figure for the six-route SoccerGMR ablation.

- `gmr_mr_detr_test_evaluation.pdf`: paper-ready vector figure.
- `gmr_mr_detr_test_evaluation.svg`: editable vector figure.
- `gmr_mr_detr_test_evaluation.png`: preview image.
- `figure_data.csv`: exact values shown in the figure.
- `gmiou_threshold_sweep.csv`: G-mIoU@3 at 101 thresholds from 0.00 to 1.00.
- `build_gmiou_sweep.py`: deterministic full-domain metric generator.
- `render.py`: deterministic Matplotlib renderer.
- `CAPTION.md`: self-contained paper caption.

The renderer reads `mr_experiments/test_summary/results.json`. Regenerate from the project root with:

```bash
python newidea/figure/gmr_mr_detr_test_20261030/build_gmiou_sweep.py
python newidea/figure/gmr_mr_detr_test_20261030/render.py
```
