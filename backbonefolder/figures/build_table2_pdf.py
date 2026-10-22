#!/usr/bin/env python3
"""Typeset the archived experiment results using the paper's Table 2 layout.

This only formats existing results; it never evaluates or selects checkpoints.
Run with: /home/guoxiangyu/miniconda3/envs/gmr/bin/python <this file>
"""
from pathlib import Path
import argparse
import csv
from decimal import Decimal
import hashlib
import html
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[1]
METRICS = ["AUROC", "Rej-F1", "mAP", "mR@1", "mR@5", "mR+@5", "G-mIoU@1", "G-mIoU@3"]
GROUPS = [("Null-Set Rejection", 0, 2), ("Temporal Localization", 2, 6), ("Overall GMR", 6, 8)]
FONTDIR = Path("/usr/share/fonts/truetype/liberation")


def font(size=10.5, style="Regular"):
    path = FONTDIR / f"LiberationSerif-{style}.ttf"
    if path.is_file():
        return FontProperties(fname=str(path), size=size)
    return FontProperties(family="serif", size=size,
                          weight="bold" if "Bold" in style else "normal",
                          style="italic" if "Italic" in style else "normal")


def threshold_source(threshold):
    """Extract an already evaluated operating point from the frozen sweep data."""
    folder = ROOT / "figures/complete_comparison"
    original = folder / "TABLE2_OUR_RESULTS.csv"
    if threshold == 0.4:
        return original
    suffix = f"_tau_{threshold:g}"
    source = folder / f"TABLE2_OUR_RESULTS{suffix}.csv"
    with original.open(newline="") as handle:
        original_rows = list(csv.reader(handle))
    output = [original_rows[0][:]]
    output[0][2] = f"Rej-F1@{threshold:g}"
    model_names = {"Moment-DETR": "moment_detr", "QD-DETR": "qd_detr",
                   "CG-DETR": "cg_detr", "EaTR": "eatr", "FlashVTG": "flashvtg"}
    records = []
    for row in original_rows[1:]:
        name = row[0]
        model = model_names[name.split("-GMR")[0]]
        variant = "s4" if "S4_minus_Fusion" in name else "gmr" if "-GMR" in name else "original"
        if variant == "s4":
            cached = folder / "data" / f"{model}_s4_epoch4.json"
        else:
            cached = ROOT.parent / "Phase2(addbackbone_with_S4_minusFusion)/shared/plot_baselines" / f"{model}_{variant}.json"
        data = json.loads(cached.read_text())
        predictions = Path(data["source_predictions"])
        if not predictions.is_absolute():
            predictions = ROOT.parent / predictions
        prediction_hash = hashlib.sha256(predictions.read_bytes()).hexdigest()
        if prediction_hash != data["source_sha256"]:
            raise ValueError(f"Prediction source changed: {predictions}")
        point = next(point for point in data["grid"] if point["threshold"] == threshold)
        base = data["tau_0.4"]
        values = [base["AUROC"], point["Rej-F1"], base["mAP"], base["mR@1"],
                  base["mR@5"], base["mR+@5"], point["G-mIoU@1"], point["G-mIoU@3"]]
        output.append([name] + [f"{value:.2f}" for value in values])
        records.append({"model": name, "cached_sweep": str(cached),
                        "cached_sweep_sha256": hashlib.sha256(cached.read_bytes()).hexdigest(),
                        "predictions": str(predictions), "predictions_sha256": prediction_hash,
                        "threshold_point": point, "prior_evaluation_checks": data["checks"]})
    with source.open("w", newline="") as handle:
        csv.writer(handle).writerows(output)
    source.with_name(source.stem + "_SOURCE.json").write_text(json.dumps({
        "threshold": threshold, "test_queries": 1036,
        "method": "Extract the requested point from the saved, official-evaluator-checked 101-point threshold sweep; verify prediction hashes.",
        "threshold_dependent": ["Rej-F1", "G-mIoU@1", "G-mIoU@3"],
        "threshold_independent": ["AUROC", "mAP", "mR@1", "mR@5", "mR+@5"],
        "runs": records,
    }, indent=2) + "\n")
    return source


def main(threshold=0.4):
    SOURCE = threshold_source(threshold)
    suffix = "" if threshold == 0.4 else f"_tau_{threshold:g}"
    DEST = ROOT / ("TABLE2_COMPLETE_RESULTS" + suffix)
    with SOURCE.open(newline="") as handle:
        source_rows = list(csv.reader(handle))
    rows = []
    for row in source_rows[1:]:
        name = row[0]
        block = 2 if "S4_minus_Fusion" in name else 1 if "-GMR" in name else 0
        rows.append({"source_name": name,
                     "name": name.replace(" (epoch 4)", "").replace(" (E0)", ""),
                     "values": [Decimal(value) for value in row[1:]], "block": block})
    rows.sort(key=lambda row: row["block"])
    our_row_count = len(rows)
    paper_path = ROOT / "figures/complete_comparison/PAPER_TABLE4_RESULTS.json"
    paper = json.loads(paper_path.read_text())
    for entry in paper["rows"]:
        rows.append({"source_name": entry["model"] + " (paper Table 4; τ=0.4)",
                     "name": entry["model"] + " [Table 4]", "block": 3,
                     "values": [Decimal(value) if value is not None else None for value in entry["values"]]})
    gmr_rows = {row["name"]: row for row in rows if row["block"] == 1}
    for row in rows:
        row["gains"] = None
        if row["block"] == 2:
            reference = gmr_rows[row["name"].split(" + ")[0]]
            row["reference"] = reference["source_name"]
            row["gains"] = [value - base for value, base in zip(row["values"], reference["values"])]
    ranks = [sorted({row["values"][col] for row in rows if row["values"][col] is not None}, reverse=True)
             for col in range(8)]

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    # Extend the page vertically to keep all 20 rows legible in one table.
    # All coordinates are in typographic points; width stays 297 mm.
    width, height = 841.89, 780.28
    offset = height - 595.28
    fig = plt.figure(figsize=(width / 72, height / 72), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.axis("off")

    def text(x, y, value, size=10.5, style="Regular", ha="left"):
        return ax.text(x, y, value, ha=ha, va="center", fontproperties=font(size, style))

    def rule(y, left=32, right=width - 32, lw=0.55):
        ax.plot([left, right], [y, y], color="black", lw=lw, solid_capstyle="butt")

    text(32, 557 + offset, "Table 2. Complete results on the Soccer-GMR test set, with MLLM references", 13, "Bold")
    text(32, 537 + offset, f"Our 14 runs: 1,036 queries (544 positive / 492 negative), τ = {threshold:g}. Six paper Table 4 rows retain τ = 0.4. All values: %.", 10.5)

    left, right, model_width = 32, width - 32, 260
    metric_left = left + model_width
    step = (right - metric_left) / 8
    xs = [metric_left + (col + 0.5) * step for col in range(8)]
    rule(514 + offset, lw=1.15)
    text(left + 5, 482 + offset, "Model", 11, "Bold")
    for label, start, stop in GROUPS:
        text(metric_left + (start + stop) / 2 * step, 501 + offset, label, 11, "Italic", "center")
        rule(490 + offset, metric_left + start * step + 4, metric_left + stop * step - 4, 0.5)
    for x, label in zip(xs, METRICS):
        text(x, 478 + offset, label, 10.5, "Regular", "center")
    rule(464 + offset, lw=0.75)

    underline_artists = []
    y = 448 + offset
    previous = 0
    for row in rows:
        if row["block"] != previous:
            rule(y + 7, lw=0.65)
            y -= 9
            if row["block"] == 3:
                text(left + 5, y, "MLLM references: paper Table 4 (reported at τ = 0.4)", 10, "Italic")
                y -= 19
        value_y = y + 3 if row["gains"] is not None else y
        text(left + 5, value_y, row["name"], 10.8)
        for col, (x, value) in enumerate(zip(xs, row["values"])):
            if value is None:
                continue
            artist = text(x, value_y, f"{value:.2f}", 11,
                          "Bold" if value == ranks[col][0] else "Regular", "center")
            if value == ranks[col][1]:
                underline_artists.append(artist)
            if row["gains"] is not None:
                text(x, y - 8.5, f'({row["gains"][col]:+.2f})', 8.5, ha="center")
        previous = row["block"]
        y -= 31 if row["gains"] is not None else 20
    rule(y + 8, lw=1.15)

    text(32, y - 12, "First 14 rows: our baseline / GMR / GMR + S4_minus_Fusion runs. Bold: best; underline: second-best among reported values.", 9.5)
    text(32, y - 27, "All S4_minus_Fusion rows use epoch 4. FlashVTG-GMR uses the archived E0 run; its S4 variant uses the historical no-QCD epoch 4.", 9.5)
    text(32, y - 42, f"Our Rej-F1 and G-mIoU use τ = {threshold:g}; AUROC is threshold-free; localization uses positives. No adapter-free FlashVTG run.", 9.5)
    text(32, y - 57, "Parentheses: S4_minus_Fusion minus same-backbone GMR, in percentage points (pp); negative values indicate decreases.", 9.5)
    text(32, y - 72, "Last 6 rows: published Table 4 values at τ = 0.4, not our reruns. Unreported mR@5 and G-mIoU cells are blank.", 9.5)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in underline_artists:
        bbox = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
        rule(bbox.y0 + 0.4, bbox.x0, bbox.x1, 0.5)
    for extension in ("pdf", "svg", "png"):
        kwargs = {"dpi": 220} if extension == "png" else {}
        if extension == "pdf":
            kwargs["metadata"] = {"Title": "Table 2 - Complete experimental results",
                                  "Subject": f"14 experiment results at threshold {threshold:g}, plus 6 paper Table 4 references at threshold 0.4",
                                  "Creator": "build_table2_pdf.py (Matplotlib)"}
        fig.savefig(DEST.parent / (DEST.name + "." + extension), **kwargs)
    plt.close(fig)

    # Editable HTML retains the merged headers and ranking marks.
    html_rows = []
    for i, row in enumerate(rows):
        cls = ' class="block"' if i and row["block"] != rows[i - 1]["block"] else ""
        if row["block"] == 3 and rows[i - 1]["block"] != 3:
            html_rows.append('<tr class="block"><th colspan="9"><em>MLLM references: paper Table 4 (reported at τ = 0.4)</em></th></tr>')
        cells = [f'<th scope="row">{html.escape(row["name"])}</th>']
        for col, value in enumerate(row["values"]):
            if value is None:
                cells.append("<td></td>")
                continue
            label = f"{value:.2f}"
            if value == ranks[col][0]:
                label = f"<strong>{label}</strong>"
            elif value == ranks[col][1]:
                label = f"<u>{label}</u>"
            if row["gains"] is not None:
                label += f'<br><span class="gain">({row["gains"][col]:+.2f})</span>'
            cells.append(f"<td>{label}</td>")
        html_rows.append(f"<tr{cls}>" + "".join(cells) + "</tr>")
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Table 2 — Complete experimental results</title>
<style>
body{font-family:"Liberation Serif","Times New Roman",serif;margin:36px;color:#000;background:white}
table{border-collapse:collapse;width:100%;max-width:1280px;margin:auto;border-top:2px solid;border-bottom:2px solid}
caption{text-align:left;padding-bottom:14px;line-height:1.5;font-size:16px}
th,td{padding:8px 10px;white-space:nowrap;font-weight:normal}
td{text-align:center;font-variant-numeric:tabular-nums}
.gain{font-size:85%}
thead th{text-align:center}thead tr:first-child th:not(:first-child){font-style:italic;border-bottom:1px solid}
thead{border-bottom:1px solid}thead th:first-child,tbody th{text-align:left}
.block{border-top:1px solid}tfoot{border-top:2px solid;font-size:13px}tfoot td{text-align:left;white-space:normal;padding-left:0}
@page{size:297mm 275.27mm;margin:12mm}@media print{body{margin:0}th,td{padding:5px 7px}}
</style><table>
<caption><strong>Table 2. Complete experimental results on the Soccer-GMR test set.</strong><br>
Our 14 runs on all 1,036 test queries (544 positive / 492 negative); τ = THRESHOLD. Six paper Table 4 rows retain τ = 0.4. All values are percentages.</caption>
<thead><tr><th rowspan="2">Model</th><th colspan="2">Null-Set Rejection</th><th colspan="4">Temporal Localization</th><th colspan="2">Overall GMR</th></tr>
<tr>''' + "".join(f"<th>{metric}</th>" for metric in METRICS) + "</tr></thead><tbody>\n" + "\n".join(html_rows) + '''
</tbody><tfoot><tr><td colspan="9">
First 14 rows: our baseline / GMR / GMR + S4_minus_Fusion runs. Bold: best; underline: second-best among reported values.<br>
All S4_minus_Fusion rows use epoch 4. FlashVTG-GMR uses the archived E0 run; its S4 variant uses the historical no-QCD epoch 4.<br>
No adapter-free FlashVTG result is available. Our Rej-F1 and G-mIoU use τ = THRESHOLD; AUROC is threshold-free; localization uses positives.<br>
Parentheses: S4_minus_Fusion minus same-backbone GMR, in percentage points (pp); negative values indicate decreases.<br>
Last 6 rows: published Table 4 values at τ = 0.4, not our reruns. Unreported mR@5 and G-mIoU cells are blank.
</td></tr></tfoot></table></html>'''
    # Path.with_suffix would truncate the decimal threshold in the file stem.
    def export_path(extension):
        return DEST.parent / (DEST.name + extension)

    page = page.replace("THRESHOLD", f"{threshold:g}")
    export_path(".html").write_text(page)

    latex_rows = []
    for i, row in enumerate(rows):
        if i and row["block"] != rows[i - 1]["block"]:
            latex_rows.append(r"\midrule")
            if row["block"] == 3:
                latex_rows.append(r"\multicolumn{9}{l}{\textit{MLLM references: paper Table 4 (reported at $\tau=0.4$)}} \\")
        cells = [row["name"].replace("_", r"\_")]
        for col, value in enumerate(row["values"]):
            if value is None:
                cells.append("")
                continue
            label = f"{value:.2f}"
            if value == ranks[col][0]:
                label = r"\textbf{" + label + "}"
            elif value == ranks[col][1]:
                label = r"\underline{" + label + "}"
            if row["gains"] is not None:
                label = (r"\shortstack{" + label + r" \\ {\scriptsize ("
                         + f'{row["gains"][col]:+.2f}' + r")}}")
            cells.append(label)
        latex_rows.append(" & ".join(cells) + r" \\")
    latex = r"""\documentclass{article}
\usepackage[paperwidth=297mm,paperheight=275.27mm,margin=12mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{booktabs,graphicx}
\pagestyle{empty}
\begin{document}
\begin{table}[ht]
\centering
\caption{Complete results on the Soccer-GMR test set. Our 14 runs use all 1,036 queries (544 positive / 492 negative), $\tau=THRESHOLD$. Six published Table 4 rows retain $\tau=0.4$. All values are percentages.}
\label{tab:complete-gmr}
\renewcommand{\arraystretch}{1.25}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrr}
\toprule
& \multicolumn{2}{c}{\textit{Null-Set Rejection}} & \multicolumn{4}{c}{\textit{Temporal Localization}} & \multicolumn{2}{c}{\textit{Overall GMR}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-7}\cmidrule(lr){8-9}
Model & AUROC & Rej-F1 & mAP & mR@1 & mR@5 & mR$^+$@5 & G-mIoU@1 & G-mIoU@3 \\
\midrule
""" + "\n".join(latex_rows) + r"""
\bottomrule
\end{tabular}}
\par\smallskip\begin{minipage}{\textwidth}\footnotesize
First 14 rows: our baseline / GMR / GMR + S4\_minus\_Fusion runs. Bold: best; underline: second-best among reported values.\\
All S4\_minus\_Fusion rows use epoch 4. FlashVTG-GMR uses the archived E0 run; its S4 variant uses the historical no-QCD epoch 4.\\
No adapter-free FlashVTG result is available. Our Rej-F1 and G-mIoU use $\tau=THRESHOLD$; AUROC is threshold-free; localization uses positives.\\
Parentheses: S4\_minus\_Fusion minus same-backbone GMR, in percentage points (pp); negative values indicate decreases.\\
Last 6 rows: published Table 4 values at $\tau=0.4$, not our reruns. Unreported mR@5 and G-mIoU cells are blank.
\end{minipage}
\end{table}
\end{document}
"""
    # Retain Table 2 numbering when compiled as a standalone document.
    latex = latex.replace(r"\begin{document}", "\\begin{document}\n\\setcounter{table}{1}")
    latex = latex.replace("THRESHOLD", f"{threshold:g}")
    export_path(".tex").write_text(latex)
    with export_path(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(source_rows[0] + ["Source", "Reported Rej-F1 threshold"])
        for row in rows:
            writer.writerow([row["source_name"]] + [f"{value:.2f}" if value is not None else "" for value in row["values"]]
                            + (["Paper Table 4", 0.4] if row["block"] == 3 else ["Our experiment", threshold]))
    gains_path = export_path("_GAINS.csv")
    with gains_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Model", "GMR reference"] + [metric + " delta (pp)" for metric in METRICS])
        for row in rows:
            if row["gains"] is not None:
                writer.writerow([row["source_name"], row["reference"]] + [f"{gain:+.2f}" for gain in row["gains"]])

    # Keep the existing Markdown as a single table, in its backbone-grouped order.
    markdown_header = ["Model（S4 行括号为相对同主干 GMR 的变化，单位 pp）", "AUROC (%)",
                       f"Rej-F1 (本实验 τ={threshold:g}；论文 T4 τ=0.4, %)", "mAP (%)", "mR@1 (%)", "mR@5 (%)",
                       "mR+@5 (%)", f"G-mIoU@1 (τ={threshold:g}, %)", f"G-mIoU@3 (τ={threshold:g}, %)"]
    markdown = ["| " + " | ".join(markdown_header) + " |",
                "| " + " | ".join(["---"] + ["---:"] * 8) + " |"]
    by_source_name = {row["source_name"]: row for row in rows}
    for source in source_rows[1:]:
        row = by_source_name[source[0]]
        cells = [row["source_name"]]
        for col, value in enumerate(row["values"]):
            label = f"{value:.2f}"
            if row["gains"] is not None:
                label += f' ({row["gains"][col]:+.2f})'
            cells.append(label)
        markdown.append("| " + " | ".join(cells) + " |")
    for row in rows:
        if row["block"] != 3:
            continue
        name = row["source_name"].replace("paper Table 4;", "论文 Table 4;")
        cells = [name] + [f"{value:.2f}" if value is not None else "" for value in row["values"]]
        markdown.append("| " + " | ".join(cells) + " |")
    (ROOT / ("TABLE2_OUR_RESULTS" + suffix + ".md")).write_text("\n".join(markdown) + "\n")
    export_path(".json").write_text(json.dumps({
        "source_csv": str(SOURCE), "threshold": threshold, "test_queries": 1036,
        "reference_layout": "/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval/2605.02623v1.pdf, Table 2 (PDF page 8)",
        "row_count": len(rows), "our_row_count": our_row_count, "paper_row_count": len(paper["rows"]),
        "block_counts": [sum(row["block"] == b for row in rows) for b in range(4)],
        "paper_table4": {**paper, "source_pdf_sha256": (hashlib.sha256(Path(paper["source_pdf"]).read_bytes()).hexdigest()
                                                        if Path(paper["source_pdf"]).is_file() else paper.get("source_pdf_sha256")),
                         "transcription_json": str(paper_path)},
        "highlight": "Column maximum in bold; second distinct maximum underlined, among available reported values; paper Rej-F1 retains its original threshold.",
        "gain_definition": f"S4_minus_Fusion minus the same-backbone GMR at threshold {threshold:g}, in percentage points; signed differences of the displayed two-decimal results.",
        "gains": [{"model": row["source_name"], "reference": row["reference"],
                   "delta_pp": {metric: f"{gain:+.2f}" for metric, gain in zip(METRICS, row["gains"])}}
                  for row in rows if row["gains"] is not None],
        "note": "14 of our experiment results, followed by 6 published Table 4 references. Paper Rej-F1 is reported at 0.4 in both tables, never relabeled as 0.6; missing paper metrics remain blank.",
    }, indent=2) + "\n")
    print(f"Exported {len(rows)} rows to {DEST}.{{pdf,png,svg,html,tex,csv,json}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.4, choices=(0.4, 0.5, 0.6))
    main(parser.parse_args().threshold)
