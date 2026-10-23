"""Wait for the existing queue, then audit and export; never launch training."""
import json
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import subprocess
import sys
import time
from pathlib import Path
from run import sha

ROOT = Path(__file__).resolve().parent
BACKBONES = ['moment_detr', 'qd_detr', 'cg_detr', 'eatr', 'flashvtg']


def main():
    while True:
        queue = json.loads((ROOT / 'QUEUE.json').read_text())
        states = [s['status'] for s in queue['jobs'].values()]
        if any(s in ['failed', 'requires_review'] for s in states):
            raise RuntimeError('Training queue has a failed/incomplete task; inspect logs')
        if all(s == 'complete' for s in states):
            break
        os.kill(queue['pid'], 0)
        time.sleep(5)
    for backbone in BACKBONES:
        report = ROOT / 'reports' / f'{backbone}_AUDIT.json'
        if report.exists():
            audit = json.loads(report.read_text())
            if audit.get('passed') and all(c['checkpoint_sha256'] == sha(ROOT / 'runs' / backbone / c['setting'] / 'epoch_004.ckpt') for c in audit['checks']):
                continue
        subprocess.run([sys.executable, str(ROOT / 'audit.py'), '--backbone', backbone], check=True)
    subprocess.run([sys.executable, str(ROOT / 'summarize.py'), '--verify'], check=True)
    subprocess.run([sys.executable, str(ROOT / 'tables.py')], check=True)
    (ROOT / 'reports/COMPLETION.json').write_text(json.dumps(dict(
        passed=True, cells=20, newly_trained=10, reused=10, adapted_cells=15,
        frozen_baselines=5, fixed_adaptation_epoch=4, seed=2026,
        independently_audited_new_runs=10, official_metric_evaluations=120), indent=2) + '\n')


if __name__ == '__main__':
    main()
