#!/usr/bin/env python3
"""Persistent, single-owner three-backbone training and paired-test queue.

Launch only after the expansion debug and regression gates have passed. On a
worker failure stop dispatching; restart after inspection, using legal last.pt.
The lock is inherited by workers, preventing a replacement supervisor from
starting while orphaned workers are still active.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('bm_detr', 'bam_detr', 'uvcom')


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--python', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.queue.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    configs = {name: root/'configs'/(name+'.json') for name in NAMES}
    if not all(p.is_file() for p in configs.values()):
        raise FileNotFoundError('Missing explicit resolved queue configurations')
    workers = {}
    done = set()
    failure = None
    jobs = [(name, kind) for name in NAMES for kind in ('gmr', 'gmr_test', 'geq_full', 'geq_full_test')]

    def complete(job):
        name, kind = job
        def digest(path):
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
            return h.hexdigest()
        if kind.endswith('_test'):
            directory = root/name/kind
            for split in ('test_standard','test_full'):
                part=directory/split
                marker=part/'complete.json'
                if not marker.is_file(): return False
                data=json.loads(marker.read_text())
                if any(not (part/file).is_file() or digest(part/file)!=data[key] for file,key in [('predictions.jsonl','predictions_sha256'),('metrics.json','metrics_sha256')]): return False
            return True
        directory=root/name/kind
        marker=directory/'complete.json'
        if not marker.is_file() or not (directory/'best.pt').is_file(): return False
        data=json.loads(marker.read_text())
        return data.get('stage')==kind and digest(directory/'best.pt')==data['best_sha256']

    def deps(job):
        name, kind = job
        return [] if kind == 'gmr' else [(name, 'gmr')] if kind in ('gmr_test','geq_full') else [(name,'geq_full')]

    def command(job, gpu):
        name, kind = job
        output = root/name/kind
        common = [args.python, '-m', 'soccergmr']
        if kind.endswith('_test'):
            stage = kind[:-5]
            return common + ['test','--config',str(configs[name]),'--output',str(output),'--checkpoint',str(root/name/stage/'best.pt'),'--device',f'cuda:{gpu}']
        cmd = common + ['train','--config',str(configs[name]),'--output',str(output),'--stage',kind,'--device',f'cuda:{gpu}']
        if kind == 'geq_full': cmd += ['--baseline',str(root/name/'gmr'/'best.pt')]
        if (output/'last.pt').is_file(): cmd += ['--resume']
        return cmd

    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT/'src')
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.chdir(ROOT)
    (root/'logs').mkdir(exist_ok=True)
    while True:
        for job, (process, gpu, log) in list(workers.items()):
            code = process.poll()
            if code is None: continue
            log.close()
            del workers[job]
            if code != 0 or not complete(job):
                failure = {'job':job,'exit_code':code,'log':str(root/'logs'/('_'.join(job)+'.log'))}
            else: done.add(job)
        # On supervisor restart, rerun each entry point so it validates source,
        # assets and completion hashes before satisfying downstream dependencies.
        if failure is None:
            for job in jobs:
                if job in done or job in workers or not all(x in done for x in deps(job)): continue
                counts = {gpu:sum(value[1] == gpu for value in workers.values()) for gpu in (0,1)}
                available = [gpu for gpu in (0,1) if counts[gpu] < 2]
                if not available: break
                gpu = min(available, key=lambda x:counts[x])
                log = (root/'logs'/('_'.join(job)+'.log')).open('a', buffering=1)
                cmd = command(job,gpu)
                log.write('\nDispatch: '+repr(cmd)+'\n')
                process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, pass_fds=(lock.fileno(),))
                workers[job] = (process,gpu,log)
        state = {'supervisor_pid':os.getpid(), 'completed':['/'.join(j) for j in sorted(done)],
                 'running':{'/'.join(j):{'pid':p.pid,'gpu':g,'log':str(root/'logs'/('_'.join(j)+'.log'))} for j,(p,g,_) in workers.items()},
                 'failure':failure, 'status':'failed' if failure else 'complete' if len(done)==len(jobs) else 'running'}
        atomic_json(root/'queue_state.json',state)
        if not workers and failure:
            return 1
        if not workers and len(done)==len(jobs):
            state['status']='reporting'
            atomic_json(root/'queue_state.json',state)
            with (root/'logs'/'report.log').open('a',buffering=1) as log:
                result=subprocess.run([args.python,str(ROOT/'scripts'/'report_expansion.py'),'--root',str(root)],stdout=log,stderr=subprocess.STDOUT,env=env,pass_fds=(lock.fileno(),))
            state['status']='complete' if result.returncode==0 else 'failed'
            if result.returncode:state['failure']={'job':'report','exit_code':result.returncode,'log':str(root/'logs'/'report.log')}
            atomic_json(root/'queue_state.json',state)
            return result.returncode
        time.sleep(5)


if __name__=='__main__':
    sys.exit(main())
