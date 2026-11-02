# Figure design and verification

The experimental-result figures compare all four backbones and six routes.
No route is highlighted as a universal winner: localization, additional-moment
recall and existence metrics do not show the same ordering.

## Design

- The cross-backbone figure uses four annotated heatmaps: mAP, mR+@5, AUROC,
  and G-mIoU@3. Every heatmap has the same four-by-six layout; cells without a
  completed Test are gray and explicitly marked with a dash.
- Each backbone row is independently min–max normalized for color encoding.
  Color compares completed methods only within that row. Missing cells remain
  gray and do not contribute to the row's bounds; constant rows use the middle
  color. Printed cell values always remain the original Test measurements.
- Each backbone has a four-panel bar figure and a separate 101-point threshold
  curve. The bars have distinct marker shapes; the curves combine color,
  marker shape and line style. The palette is color-blind friendly.
- The status figure uses categorical colors plus explicit text and actual
  completed epochs. A queue status is not a zero-valued Test measurement.
- Font sizes are 9–13 pt at the exported native size. Use the downloadable
  vector panels at a readable size when inserting them into a paper.
- PDF/SVG are native vector exports; PNG is a preview. G-mIoU curves show both
  axes from 0 to 1. All bars start at zero. Heatmap color bars explicitly label
  within-row relative scores from Low to High, rather than absolute metrics.
  The numbers printed in cells allow absolute comparisons across backbones.
- Captions and the unified README specify the Test split, gate threshold,
  units, incomplete runs and mixed training protocols. There are no fabricated
  error bars for these single-seed runs, no smoothing, and no Test-based
  checkpoint or threshold selection.

## Verification

The published snapshot contains exactly four backbones × six methods. Each
completed Test value was matched to its source JSON and SHA256; G-mIoU is
divided by 100 once. Each completed run has exactly 101 threshold measurements,
and the value at threshold 0.4 agrees with its summary metric within the CSV's
four-decimal rounding precision.

The report was also reproduced offline from `snapshot.json`, with identical
CSV and snapshot content. Markdown links and all PDF/SVG/PNG files were checked.
The overview, status plot, per-backbone bars and threshold curves were visually
inspected for label readability, missing-value handling and honest axes.
