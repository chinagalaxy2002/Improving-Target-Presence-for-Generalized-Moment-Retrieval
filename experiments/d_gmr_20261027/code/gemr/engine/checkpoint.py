"""Complete epoch archives, exclusive atomic publish and strict load checks."""
import os
import ctypes
import errno
import random
import uuid
from pathlib import Path
import numpy as np
import torch


REQUIRED = {'epoch', 'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
            'scaler_state_dict', 'config', 'seed', 'validation_metrics', 'rng_state'}


def publish_exclusive(temp, path):
    """Atomic rename with RENAME_NOREPLACE on Linux; exclusive publish fallback."""
    libc = ctypes.CDLL(None, use_errno=True)
    if hasattr(libc, 'renameat2'):
        result = libc.renameat2(-100, os.fsencode(temp), -100, os.fsencode(path), 1)
        if result == 0:
            return
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(path)
        if error not in (errno.ENOSYS, errno.EINVAL):
            raise OSError(error, os.strerror(error), str(path))
    os.link(temp, path)
    temp.unlink()


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def load(path):
    path = Path(path)
    assert path.is_file() and path.stat().st_size > 0
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert REQUIRED <= state.keys(), REQUIRED-state.keys()
    assert state['seed'] == 2026 and 1 <= state['epoch'] <= 4
    assert state['model_state_dict'] and state['optimizer_state_dict']['param_groups']
    return state


def save(path, model, optimizer, scheduler, epoch, config, validation_metrics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Epoch archives are never overwritten: {path}')
    state = dict(epoch=epoch, model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                 optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                 scaler_state_dict=None, config=config, seed=2026, validation_metrics=validation_metrics,
                 rng_state=rng_state(), checkpoint_phase=('epoch_complete_before_validation' if validation_metrics is None else 'epoch_complete_with_validation'))
    temp = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('xb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    reloaded = load(temp)
    model.load_state_dict(reloaded['model_state_dict'], strict=True)
    # Atomically rename without replacing ANY existing epoch checkpoint.
    publish_exclusive(temp, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, epoch=epoch, strict_reload=True)


def resume(path, model, optimizer=None, scheduler=None):
    state = load(path)
    model.load_state_dict(state['model_state_dict'], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler_state_dict'])
    restore_rng(state['rng_state'])
    return state
