#!/usr/bin/env python3
"""Run the packaged training, evaluation and queue scripts from any directory."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
COMMANDS = {
    'train': 'd_gmr.engine.train',
    'test': 'd_gmr.scripts.evaluate_checkpoint',
    'launch': 'd_gmr.scripts.launch',
    'preflight': 'd_gmr.scripts.preflight_two_stage',
    'report': 'd_gmr.scripts.report',
}

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit('Usage: python run.py {train|test|launch|preflight|report} [arguments]')
    env = dict(os.environ)
    if env.get('DGMR_FLASHVTG_CHECKPOINT'):
        env.setdefault('GMR_FLASH_CHECKPOINT', env['DGMR_FLASHVTG_CHECKPOINT'])
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'code'), str(REPO), env.get('PYTHONPATH', '')])
    os.chdir(REPO)
    os.execve(sys.executable, [sys.executable, '-m', COMMANDS[sys.argv[1]], *sys.argv[2:]], env)
