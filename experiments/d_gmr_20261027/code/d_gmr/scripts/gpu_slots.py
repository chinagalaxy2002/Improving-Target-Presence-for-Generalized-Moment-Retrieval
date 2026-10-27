"""Two cross-process slots per GPU, shared by D-GMR Train and Test."""
import fcntl
import os
from contextlib import contextmanager

from d_gmr.common import RUNS


@contextmanager
def slot(gpu, index, blocking=True):
    directory = RUNS/'locks'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/f'gpu_{gpu}_slot_{index}.lock').open('a+') as stream:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(stream, flags)
        except BlockingIOError:
            yield False
            return
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()))
        stream.flush()
        try:
            yield True
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
