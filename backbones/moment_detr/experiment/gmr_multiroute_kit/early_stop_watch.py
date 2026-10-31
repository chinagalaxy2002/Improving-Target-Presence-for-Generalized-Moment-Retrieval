"""Stop active experiments after 20 epochs without a validation mAP gain."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time


VARIANTS = ("baseline", "shared", "sepffn", "instruction", "multi", "adaptive")


def validation_state(path: Path) -> tuple[int, float, int, float] | None:
    records = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                records.append((int(item["epoch"]), float(item["metrics"]["brief"]["MR-full-mAP"])))
    if not records:
        return None
    best_epoch, best_map = max(records, key=lambda item: item[1])
    latest_epoch, latest_map = records[-1]
    return best_epoch, best_map, latest_epoch, latest_map


def live_training_process(pid: int, variant: str) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, PermissionError):
        return False
    return "gmr_multiroute_kit/run_train.py" in command and f"--variant {variant}" in command


def append_event(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--patience-epochs", type=int, default=20)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    log_path = args.root / "logs" / "early_stop_watch.log"
    stopped: set[str] = set()
    append_event(log_path, {"event": "watch_started", "patience_epochs": args.patience_epochs})

    while True:
        any_live = False
        for variant in VARIANTS:
            pid_path = args.root / "logs" / f"{variant}.pid"
            if not pid_path.exists():
                continue
            pid = int(pid_path.read_text(encoding="utf-8").strip())
            if not live_training_process(pid, variant):
                continue
            any_live = True
            state = validation_state(args.root / variant / "val.log")
            if state is None or variant in stopped:
                continue
            best_epoch, best_map, latest_epoch, latest_map = state
            if latest_epoch - best_epoch >= args.patience_epochs:
                os.kill(pid, signal.SIGTERM)
                stopped.add(variant)
                append_event(log_path, {
                    "event": "early_stop",
                    "variant": variant,
                    "pid": pid,
                    "best_epoch": best_epoch,
                    "best_map": best_map,
                    "latest_epoch": latest_epoch,
                    "latest_map": latest_map,
                })
        if not any_live:
            append_event(log_path, {"event": "watch_finished"})
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
