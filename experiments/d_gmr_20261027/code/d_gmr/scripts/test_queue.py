"""Formal Test queue for D-GMR only for the immutable four-epoch Val selection."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

from d_gmr.common import BACKBONES, REPO, RUNS, dump
from d_gmr.scripts.gpu_slots import slot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    args = parser.parse_args()
    guard = (RUNS/'test_queue.lock').open('a+')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {backbone: dict(status='waiting_for_training') for backbone in BACKBONES}

    def publish():
        dump(RUNS/'TEST_QUEUE.json', dict(pid=os.getpid(), updated=time.time(), jobs=state))
    publish()
    while True:
        for backbone in BACKBONES:
            if state[backbone]['status'] in ('complete', 'failed', 'training_failed'):
                continue
            run = RUNS/backbone
            if not (run/'TRAIN_COMPLETE.json').exists():
                if (run/'TRAIN_FAILED.json').exists():
                    state[backbone] = dict(status='training_failed')
                    publish()
                continue
            best = json.loads((run/'best_epoch.json').read_text())
            assert best['selection_split'] == 'val' and best['epochs_considered'] == [1, 2, 3, 4]
            launched = False
            for gpu in args.gpus:
                for index in range(2):
                    with slot(gpu, index, blocking=False) as available:
                        if not available:
                            continue
                        log_path = RUNS/'queue_logs'/f'test_{backbone}.log'
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        with log_path.open('w') as log:
                            process = subprocess.Popen([
                                sys.executable, '-m', 'd_gmr.scripts.evaluate_checkpoint',
                                '--backbone', backbone, '--checkpoint', best['checkpoint_path'],
                                '--split', 'test', '--threshold', '0.4', '--formal-test'],
                                cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                                                   PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2',
                                                   MKL_NUM_THREADS='2'),
                                stdout=log, stderr=subprocess.STDOUT)
                            state[backbone] = dict(
                                status='running', pid=process.pid, gpu=gpu, slot=index,
                                selected_epoch=best['best_epoch'], selection_split='val',
                                started=time.time(), log=str(log_path))
                            publish()
                            code = process.wait()
                        state[backbone].update(
                            status=('complete' if code == 0 and
                                    (run/'TEST_COMPLETE.json').exists() else 'failed'),
                            exit_code=code, ended=time.time())
                        publish()
                        subprocess.run([sys.executable, '-m', 'd_gmr.scripts.report'],
                                       cwd=REPO, check=True)
                        launched = True
                    if launched:
                        break
                if launched:
                    break
        finished = RUNS/'TRAIN_QUEUE_FINISHED.json'
        if finished.exists():
            train = json.loads(finished.read_text())
            for backbone in BACKBONES:
                if (train['jobs'][backbone]['status'] == 'failed' and
                        state[backbone]['status'] == 'waiting_for_training'):
                    state[backbone] = dict(status='training_failed')
            publish()
        if all(value['status'] in ('complete', 'failed', 'training_failed')
               for value in state.values()):
            break
        time.sleep(10)
    subprocess.run([sys.executable, '-m', 'd_gmr.scripts.report'], cwd=REPO, check=True)
    passed = all(value['status'] == 'complete' for value in state.values())
    dump(RUNS/'TEST_QUEUE_FINISHED.json', dict(passed=passed, jobs=state), exclusive=True)
    assert passed, state


if __name__ == '__main__':
    main()
