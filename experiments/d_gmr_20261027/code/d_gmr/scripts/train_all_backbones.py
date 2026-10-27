"""Persistent multi-worker training queue for D-GMR."""
import argparse
import concurrent.futures
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from d_gmr.common import BACKBONES, REPO, RUNS, dump
from d_gmr.scripts.gpu_slots import slot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    args = parser.parse_args()
    for backbone in BACKBONES:
        assert not (RUNS/backbone).exists(), f'Will not overwrite {RUNS/backbone}'
    guard = (RUNS/'train_queue.lock').open('a+')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock = threading.Lock()
    jobs = deque(BACKBONES)
    state = {backbone: dict(status='pending') for backbone in BACKBONES}

    def publish():
        dump(RUNS/'TRAIN_QUEUE.json', dict(pid=os.getpid(), updated=time.time(), jobs=state,
                                           gpus=args.gpus, slots_per_gpu=2))

    def worker(location):
        gpu, index = location
        while True:
            with slot(gpu, index):
                with lock:
                    if not jobs:
                        return
                    backbone = jobs.popleft()
                    path = RUNS/'queue_logs'/f'train_{backbone}.log'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    log = path.open('w')
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1',
                                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
                    process = subprocess.Popen(
                        [sys.executable, '-m', 'd_gmr.engine.train', '--backbone', backbone],
                        cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    state[backbone] = dict(status='running', pid=process.pid, gpu=gpu, slot=index,
                                           started=time.time(), log=str(path))
                    publish()
                code = process.wait()
                log.close()
                with lock:
                    state[backbone].update(
                        status=('complete' if code == 0 and
                                (RUNS/backbone/'TRAIN_COMPLETE.json').exists() else 'failed'),
                        exit_code=code, ended=time.time())
                    publish()
    publish()
    locations = [(gpu, index) for index in range(2) for gpu in args.gpus]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2*len(args.gpus)) as pool:
        list(pool.map(worker, locations))
    passed = all(value['status'] == 'complete' for value in state.values())
    dump(RUNS/'TRAIN_QUEUE_FINISHED.json', dict(passed=passed, jobs=state), exclusive=True)
    assert passed, state


if __name__ == '__main__':
    main()
