"""Live Val mAP dashboard for the Charades Multi + GEQ experiment.

Run without arguments to refresh the PNG every 60 seconds. Use --once to render
one snapshot and exit.
"""
import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paths import BACKBONES, METHODS, OUT,dump,sha
from oracle_presence import selected_val_reference


BACKBONE_NAMES = {
    "moment_detr": "Moment-DETR",
    "eatr": "EaTR",
    "qd_detr": "QD-DETR",
    "flashvtg": "FlashVTG",
}
METHOD_NAMES = {
    "gmr": "GMR",
    "multi": "GMR + Multi",
    "gmr_geq_frozen": "GMR + GEQ Frozen",
    "gmr_geq_full": "GMR + GEQ Full",
    "multi_geq_frozen": "GMR + Multi + GEQ Frozen",
    "multi_geq_full": "GMR + Multi + GEQ Full",
    "multi_positive_only": "EaTR Multi (positive-only)",
}
METHOD_COLORS = {
    "gmr": "#252525",
    "multi": "#e69f00",
    "gmr_geq_frozen": "#009e73",
    "gmr_geq_full": "#0072b2",
    "multi_geq_frozen": "#9b59b6",
    "multi_geq_full": "#d55e00",
    "multi_positive_only": "#dc2626",
}
METHOD_STYLES = {
    "gmr": "-",
    "multi": (0, (5, 1.5)),
    "gmr_geq_frozen": (0, (1.2, 1.2)),
    "gmr_geq_full": (0, (5, 1, 1, 1)),
    "multi_geq_frozen": (0, (3, 1, 1, 1)),
    "multi_geq_full": (0, (7, 1.5)),
    "multi_positive_only": (0, (2, 1)),
}
METHOD_MARKERS = {
    "gmr": "o",
    "multi": "s",
    "gmr_geq_frozen": "^",
    "gmr_geq_full": "D",
    "multi_geq_frozen": "P",
    "multi_geq_full": "X",
    "multi_positive_only": "v",
}


def method_name(backbone, method):
    if backbone == "flashvtg" and method == "multi":
        return "GMR + Multi-TQAux"
    if backbone == "flashvtg" and method == "multi_geq_frozen":
        return "GMR + Multi-TQAux + GEQ Frozen"
    if backbone == "flashvtg" and method == "multi_geq_full":
        return "GMR + Multi-TQAux + GEQ Full"
    return METHOD_NAMES[method]


def short_method_name(backbone, method):
    if backbone == "flashvtg" and method.startswith("multi"):
        return {
            "multi": "Multi-TQAux",
            "multi_geq_frozen": "Multi-TQAux+GEQ-F",
            "multi_geq_full": "Multi-TQAux+GEQ-Full",
        }[method]
    return {
        "gmr": "GMR",
        "multi": "GMR+Multi",
        "gmr_geq_frozen": "GMR+GEQ-F",
        "gmr_geq_full": "GMR+GEQ-Full",
        "multi_geq_frozen": "Multi+GEQ-F",
        "multi_geq_full": "Multi+GEQ-Full",
        "multi_positive_only": "Multi (pos-only)",
    }[method]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def read_history(path):
    rows = []
    try:
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # Ignore a partially written last line.
                epoch, map_value = row.get("epoch"), row.get("mAP")
                if epoch is None or map_value is None:
                    continue
                try:
                    epoch, map_value = int(epoch), float(map_value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(map_value):
                    rows.append({**row, "epoch": epoch, "mAP": map_value})
    except OSError:
        pass
    return sorted(rows, key=lambda row: row["epoch"])


def load_queue_active():
    snapshot = read_json(OUT / "logs" / "QUEUE.json") or {}
    active = set()
    for key in (snapshot.get("active") or {}):
        parts = key.strip("/").split("/")
        if len(parts) >= 2:
            active.add((parts[0], parts[1]))
    return active


def load_method(backbone, method, active):
    run_dir = OUT / "runs" / backbone / method
    history = read_history(run_dir / "history.jsonl")
    trained = read_json(run_dir / "TRAINED.json")
    status = read_json(run_dir / "status.json") or {}

    # The persisted Val history is authoritative; status may be a stale snapshot.
    selected_epoch = None
    if trained:
        selected_epoch = trained.get("best_epoch")
    elif history:
        selected_epoch = history[-1].get("best_epoch")
    if selected_epoch is None:
        selected_epoch = status.get("best_epoch")
    try:
        selected_epoch = int(selected_epoch) if selected_epoch is not None else None
    except (TypeError, ValueError):
        selected_epoch = None

    selected_row = next((row for row in history if row["epoch"] == selected_epoch), None)
    latest_row = history[-1] if history else None
    selected_map = selected_row.get("mAP") if selected_row else None
    latest_map = latest_row.get("mAP") if latest_row else None

    if trained:
        state = "complete"
    elif (backbone, method) in active:
        state = "running"
    elif history:
        state = "waiting"
    else:
        state = "pending"

    current_epoch = status.get("epoch")
    if current_epoch is None and latest_row:
        current_epoch = latest_row["epoch"]
    try:
        current_epoch = int(current_epoch) if current_epoch is not None else None
    except (TypeError, ValueError):
        current_epoch = None

    return {
        "history": history,
        "state": state,
        "current_epoch": current_epoch,
        "selected_epoch": selected_epoch,
        "selected_map": selected_map,
        "latest_map": latest_map,
        "selected_oracle": selected_val_reference(run_dir, OUT / "figures" / backbone / "val" / f"{method}_oracle_presence.json", history),
    }


def render(output):
    active = load_queue_active()
    has_positive_only = (OUT / "runs/eatr/multi_positive_only").is_dir()
    oracle_records={}
    fig = plt.figure(figsize=(20, 10), dpi=160)
    outer = fig.add_gridspec(2, 2, left=0.045, right=0.98, top=0.80, bottom=0.105,
                             hspace=0.34, wspace=0.17)

    for panel_index, backbone in enumerate(BACKBONES):
        row, column = divmod(panel_index, 2)
        panel = outer[row, column].subgridspec(1, 2, width_ratios=[1.4, 1.8], wspace=0.06)
        axis = fig.add_subplot(panel[0, 0])
        table = fig.add_subplot(panel[0, 1])
        methods = (*METHODS, "multi_positive_only") if backbone == "eatr" and has_positive_only else METHODS
        records = {method: load_method(backbone, method, active) for method in methods}
        oracle_records[backbone]={m:r['selected_oracle'] for m,r in records.items() if r['selected_oracle']}
        completed = sum(records[method]["state"] == "complete" for method in METHODS)
        running = sum(records[method]["state"] == "running" for method in methods)

        for method_index, method in enumerate(methods):
            record = records[method]
            history = record["history"]
            color = METHOD_COLORS[method]
            if history:
                epochs = [row["epoch"] for row in history]
                values = [row["mAP"] for row in history]
                axis.plot(
                    epochs, values, color=color, linewidth=2.5,
                    linestyle=METHOD_STYLES[method], marker=METHOD_MARKERS[method],
                    markevery=max(1, len(epochs) // 9), markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.45,
                    zorder=2 + method_index,
                )
                selected_epoch = record["selected_epoch"]
                selected_map = record["selected_map"]
                if selected_epoch is not None and selected_map is not None:
                    axis.scatter(
                        [selected_epoch], [selected_map], marker="*", s=150,
                        color=color, edgecolor="#171717", linewidth=0.7, zorder=12,
                    )

        if not any(records[method]["history"] for method in methods):
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 100)
            axis.text(0.5, 0.5, "Waiting for first Val epoch", transform=axis.transAxes,
                      ha="center", va="center", color="#64748b", fontsize=10)

        axis.set_title(
            f"{BACKBONE_NAMES[backbone]}  |  {completed}/6 trained  |  {running} running"
            + (" (incl. ablation)" if backbone == "eatr" and has_positive_only else ""),
            loc="left", fontsize=12, fontweight="bold", pad=6,
        )
        axis.set_ylabel("Val mAP (%)")
        axis.set_xlabel("Epoch")
        axis.grid(axis="both", alpha=0.24, linewidth=0.8)
        axis.tick_params(labelsize=10)
        axis.margins(x=0.04, y=0.16)
        axis.text(.02,.02,"Presence oracle: ungated mAP unchanged", transform=axis.transAxes,
                  fontsize=7.5,bbox=dict(facecolor="white",alpha=.9,edgecolor="none"))

        table.set_xlim(0, 1)
        table.set_ylim(0, 1)
        table.axis("off")
        table.text(0.01, 0.93, "Method", ha="left", va="center", fontweight="bold", fontsize=9)
        table.text(0.42, 0.93, "State", ha="left", va="center", fontweight="bold", fontsize=8.5)
        table.text(0.55, 0.93, "Last / picked", ha="left", va="center", fontweight="bold", fontsize=8.5)
        table.text(0.81, 0.93, "Oracle G1/G3", ha="left", va="center", fontweight="bold", fontsize=8)
        for index, method in enumerate(methods):
            record = records[method]
            latest = "—" if record["latest_map"] is None else f'{record["latest_map"]:.2f}'
            selected = "—" if record["selected_map"] is None else f'{record["selected_map"]:.2f}'
            state_name = {
                "complete": "done",
                "running": "run",
                "waiting": "waiting",
                "pending": "pending",
            }[record["state"]]
            y = 0.80 - index * 0.125
            table.text(0.01, y, short_method_name(backbone, method), ha="left", va="center",
                       fontsize=8.5, color=METHOD_COLORS[method], fontweight="bold")
            table.text(0.42, y, state_name, ha="left", va="center", fontsize=8,
                       color="#334155")
            table.text(0.55, y, f"{latest} / {selected}", ha="left", va="center",
                       fontsize=7.8, color="#172033", family="monospace")
            reference=record['selected_oracle']
            oracle=reference['oracle_presence'] if reference else None
            text=f"{oracle['gmiou_1']:.3f}/{oracle['gmiou_3']:.3f}" if oracle else "—"
            table.text(0.81,y,text,ha="left",va="center",fontsize=7.8,color="#172033",family="monospace")

    legend = [
        Line2D(
            [0], [0], color=METHOD_COLORS[method], lw=2.7,
            linestyle=METHOD_STYLES[method], marker=METHOD_MARKERS[method], markersize=6,
            label=METHOD_NAMES[method],
        )
        for method in ((*METHODS, "multi_positive_only") if has_positive_only else METHODS)
    ]
    fig.legend(handles=legend, loc="upper center", ncol=4 if has_positive_only else 3, frameon=False,
               bbox_to_anchor=(0.5, 0.91), fontsize=10, columnspacing=2.2,
               handlelength=3, labelspacing=1.0)
    fig.suptitle(
        "Charades Multi + GEQ | Val mAP and picked-checkpoint Oracle | seed 2023",
        y=0.965, fontsize=17, fontweight="bold",
    )
    fig.text(
        0.5, 0.025,
        "Lines: actual Val epochs, mAP (%)  |  ★: selected epoch (GEQ: maximum Val mean G-mIoU@1 over 0.4/0.5/0.6)  "
        "|  Test excluded  |  FlashVTG Multi = Multi-TQAux\n"
        "Oracle G1/G3: perfect presence with current Val-best candidates fixed (0–1); ungated mAP is unchanged.  "
        + ("EaTR positive-only: additional ablation  |  " if has_positive_only else "")
        +
        f"Updated {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        ha="center", fontsize=8.5, color="#475569",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + f".{os.getpid()}.tmp" + output.suffix)
    try:
        fig.savefig(temporary, dpi=160, format="png")
    finally:
        plt.close(fig)
    os.replace(temporary, output)
    dump(output.with_suffix('.oracle.json'),dict(type='mAP_dashboard_with_selected_val_oracles',png_sha256=sha(output),
         oracle_scope='Current Val-best only, fixed localization; ungated mAP unchanged.',references=oracle_records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=OUT / "reports" / "monitor" / "training_map_panel.png",
        help="PNG destination (replaced atomically on each refresh)",
    )
    parser.add_argument("--interval", type=float, default=60, help="refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="render one snapshot and exit")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")

    while True:
        started = time.monotonic()
        render(args.output)
        print(f"Updated {args.output}", flush=True)
        if args.once:
            return
        time.sleep(max(0.0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
