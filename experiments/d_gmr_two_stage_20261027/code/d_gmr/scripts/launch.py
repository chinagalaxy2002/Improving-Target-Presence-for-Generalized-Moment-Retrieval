"""Detach persistent train and formal-Test queues for D-GMR."""
import argparse
import os
import subprocess
import sys

from d_gmr.common import REPO, RUNS, dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    args = parser.parse_args()
    gpu_args = ['--gpus', *map(str, args.gpus)]
    assert not (RUNS/'LAUNCH.json').exists(), 'Existing queues must never be launched twice'
    directory = RUNS/'queue_logs'
    directory.mkdir(parents=True, exist_ok=True)
    record = {}
    for target in ('train_all_backbones', 'test_queue'):
        with (directory/f'{target}.log').open('w') as stream:
            process = subprocess.Popen(
                [sys.executable, '-m', f'd_gmr.scripts.{target}', *gpu_args],
                cwd=REPO, env=dict(os.environ, PYTHONUNBUFFERED='1'),
                start_new_session=True, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT)
        record[target] = dict(pid=process.pid, log=str(directory/f'{target}.log'))
    dump(RUNS/'LAUNCH.json', dict(queues=record, gpus=args.gpus, slots_per_gpu=2), exclusive=True)
    print(record, flush=True)


if __name__ == '__main__':
    main()
