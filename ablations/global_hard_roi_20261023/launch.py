"""Two GPU subprocess workers; no agent delegation; never overwrite a run."""
import concurrent.futures
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
lock = threading.Lock()
jobs = [(b, s) for s in ['hard_only', 'global_hard'] for b in
        ['moment_detr', 'qd_detr', 'cg_detr', 'eatr', 'flashvtg']]
state = {f'{b}/{s}': dict(status='pending') for b, s in jobs}


def save():
    tmp = ROOT / 'QUEUE.tmp'
    tmp.write_text(json.dumps(dict(pid=os.getpid(), updated=time.time(), jobs=state), indent=2) + '\n')
    tmp.replace(ROOT / 'QUEUE.json')


def worker(gpu):
    while True:
        with lock:
            if not jobs:
                return
            backbone, setting = jobs.pop(0)
            key = f'{backbone}/{setting}'
            directory = ROOT / 'runs' / backbone / setting
            if directory.exists():
                state[key] = dict(status='complete' if (directory / 'COMPLETE.json').exists() else 'requires_review')
                save()
                continue
            path = ROOT / 'logs' / f'{backbone}_{setting}.log'
            path.parent.mkdir(exist_ok=True)
            stream = path.open('w')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
            process = subprocess.Popen([sys.executable, str(ROOT / 'run.py'), '--backbone', backbone,
                                        '--setting', setting], env=env, stdout=stream, stderr=subprocess.STDOUT)
            state[key] = dict(status='running', pid=process.pid, gpu=gpu, log=str(path), started=time.time())
            save()
        code = process.wait()
        stream.close()
        with lock:
            state[key].update(status='complete' if code == 0 and (directory / 'COMPLETE.json').exists() else 'failed',
                              exit_code=code, ended=time.time())
            save()


def main():
    with (ROOT / 'launch.lock').open('w') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker, [0, 1]))
        assert all(s['status'] == 'complete' for s in state.values()), state


if __name__ == '__main__':
    main()
