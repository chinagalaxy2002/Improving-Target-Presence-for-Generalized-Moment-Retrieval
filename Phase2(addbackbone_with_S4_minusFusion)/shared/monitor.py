#!/usr/bin/env python3
"""Read-only job monitor; artifacts are written only inside this fine-tuning run."""
import datetime
import json
from pathlib import Path
import sqlite3
import subprocess
import time
from train import dump

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT.parent / 'table2_repro_20261022/shared/jobs.sqlite3'
NAMES = ('moment_detr','qd_detr','cg_detr','eatr')


def refresh():
    with sqlite3.connect(f'file:{DB}?mode=ro',uri=True) as connection:
        connection.row_factory = sqlite3.Row
        jobs = [dict(row) for row in connection.execute("SELECT id,model,gpu,status,pid,pid_tick,started,finished,exit_code FROM jobs WHERE id LIKE 's4-gmr-finetune6-%' ORDER BY submitted")]
    rows = []
    for name in NAMES:
        directory = ROOT/name/'runs/s2026_gmr_s4_e6'
        def read(filename):
            path = directory/filename
            return json.loads(path.read_text()) if path.exists() else {}
        preflight = read('PREFLIGHT.json'); configuration = read('CONFIG.json'); progress = read('progress.json')
        init = configuration.get('initialization', preflight.get('initialization', {}))
        job = next((j for j in reversed(jobs) if j['model'] == name), {})
        rows.append({'backbone':name,'scheduler':job,'progress':progress,'preflight_passed':bool(preflight.get('passed')),
                     'initialization':init,'run_directory':str(directory),'log':str(ROOT/name/'training.log')})
    data = {'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'backbones':rows}
    dump(ROOT/'reports/STATUS.json',data)
    text = ['# S4_minus_Fusion six-epoch fine-tuning','',f'Updated {data["time_utc"]}.', '',
            'Exactly six fine-tuning epochs, seed 2026. Each epoch is saved as checkpoints/epoch_NNN.ckpt. '
            'No best-checkpoint selection or early stopping is used. Epochs 3–6 each reload their own saved checkpoint and evaluate '
            'on all 1036 Standard test queries. Original/GMR four G-mIoU lines remain, with each S4 epoch added. '
            'Both single-epoch overlays and cumulative overlays are exported, with analysis/gmiou/tradeoff/rej_f1 in PNG/PDF/SVG. '
            'Original baselines and earlier runs are preserved. LoRA LR 8e-5; new-module LR 3e-5; all other weights frozen; no augmentation.', '',
            '| Backbone | GPU | PID | Scheduler | GPU preflight | Epoch / batch | Optimizer updates | Trainable / LoRA |',
            '|---|---:|---:|---|---|---|---:|---|']
    for row in rows:
        j=row['scheduler'];p=row['progress'];counts=row['initialization'].get('counts',{})
        text.append(f'| {row["backbone"]} | {j.get("gpu", "pending")} | {j.get("pid", "pending")} | {j.get("status", "pending")} | {"PASS" if row["preflight_passed"] else "pending"} | {p.get("epoch", "-")} / {p.get("batch", "-")} | {p.get("optimizer_updates", 0)} | {counts.get("trainable", "-")} / {counts.get("lora", "-")} |')
    text += ['', '## Initialization checkpoints', '']
    for row in rows:
        init=row['initialization']
        text.append(f'- {row["backbone"]}: `{init.get("source_checkpoint", "pending")}`; SHA256 `{init.get("sha256", "pending")}`. [Run directory]({row["backbone"]}/runs/s2026_gmr_s4_e6), [training log]({row["backbone"]}/training.log).')
    text += ['', '## Saved epoch artifacts', '',
             'Each run has checkpoints/epoch_001.ckpt through epoch_006.ckpt, with hash sidecars. '
             'Test predictions and metrics: evaluations/epoch_003/ through epoch_006/. '
             'Images: /home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/figures/<backbone>/finetune_epochs/. '
             'Each epoch has this_epoch/ (original four lines plus this S4 epoch), all_epochs_so_far/ (original four plus all available S4 epochs), '
             'and a JSON curve/figure manifest tying results to the exact saved epoch hash.', '',
             'GPU preflight checks gradients, unchanged frozen parameters, strict checkpoint reload, natural negative examples and evaluator compatibility. '
             'It uses a small validation subset only as an implementation check. The reported fine-tuning curves all use Standard test. '
             'There is no test-driven checkpoint or threshold selection. The latest.ckpt symlink is only a convenience pointer, not a selected best.', '',
             'Existing scheduler limit: at most two jobs per physical GPU. Attach monitoring: `tmux attach -t s4_gmr_finetune6_20261022`. '
             'Baseline sources, previous logs and superseded 40-epoch-schedule runs remain preserved.']
    (ROOT/'DEPLOYMENT_REPORT.md').write_text('\n'.join(text)+'\n')
    return jobs


if __name__ == '__main__':
    while True:
        try:
            jobs = refresh()
            if len(jobs)==4 and all(j['status'] not in ('QUEUED','RUNNING') for j in jobs): break
        except Exception as exc: print(type(exc).__name__,str(exc),flush=True)
        time.sleep(10)
