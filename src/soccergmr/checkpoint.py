"""Atomic, strict epoch-boundary checkpoints and exclusive run ownership."""
import contextlib
import fcntl
import os
import random
from pathlib import Path
import numpy as np
import torch
from .io import fingerprint,sha256

FORMAT=1

@contextlib.contextmanager
def run_lock(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.run.lock').open('w') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('Another worker owns this run directory') from error
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)

def rng_state(generator):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],loader=generator.get_state())

def restore_rng(state,generator):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:
        if len(state['cuda'])!=torch.cuda.device_count():raise ValueError('Resume with the same visible CUDA devices')
        torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['loader'])

def save(path,payload):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as handle:
        torch.save(dict(format=FORMAT,**payload),handle)
        handle.flush();os.fsync(handle.fileno())
    os.replace(temp,path)

def load(path,identity=None,stage=None):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint.get('format')!=FORMAT:raise ValueError('Unsupported checkpoint; explicit conversion required')
    if identity is not None and checkpoint['identity']!=identity:raise ValueError('Checkpoint/config/data/source identity mismatch')
    if stage is not None and checkpoint['stage']!=stage:raise ValueError('Wrong checkpoint stage')
    return checkpoint

def source_identity():
    root=Path(__file__).parent
    return fingerprint({str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*.py'))})

def asset_identity(datasets):
    paths=set()
    for dataset in datasets:
        paths.add(dataset.path)
        for row in dataset.data:paths.update(dataset.feature_paths(row))
    return fingerprint({str(p):sha256(p) for p in sorted(paths)})
