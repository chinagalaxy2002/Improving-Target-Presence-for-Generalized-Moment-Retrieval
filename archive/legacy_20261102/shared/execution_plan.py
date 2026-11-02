"""Per-job queue overrides, applied by newly launched runner processes only."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
HERE = Path(os.environ.get('GMR_RUNTIME_ROOT', CODE_ROOT))


def plan():
    path = HERE / 'execution_plan.json'
    if not path.exists():
        path = CODE_ROOT / 'execution_plan.json'
    return json.loads(path.read_text()) if path.exists() else {'revision': None, 'jobs': {}}


def job_profile(backbone, variant):
    definition = plan()
    profile = dict(definition['jobs'].get(f'{backbone}/{variant}', {}))
    profile.setdefault('microbatch', 2)
    profile.setdefault('gradient_accumulation', 4)
    profile['revision'] = definition['revision'] if profile.get('action') else None
    return profile


def assert_not_running(backbone, variant):
    """Protect the current job and all other actively executing queue jobs."""
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = proc.joinpath('cmdline').read_bytes().split(b'\0')
            args = [arg.decode(errors='replace') for arg in args if arg]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if not {str(HERE / 'run.py'), str(CODE_ROOT / 'run.py')}.intersection(args):
            continue
        try:
            same = args[args.index('--backbone') + 1] == backbone and args[args.index('--variant') + 1] == variant
        except (ValueError, IndexError):
            continue
        if same:
            raise RuntimeError(f'Refusing to reset active job {backbone}/{variant}, PID={proc.name}')


def prepare_restart(backbone, variant):
    """Archive old state once, so retries resume only the new batch32 run."""
    profile = job_profile(backbone, variant)
    if profile.get('action') != 'restart':
        return profile
    run = HERE / 'runs' / backbone / variant
    marker = run / 'execution_plan.json'
    if marker.exists() and json.loads(marker.read_text()).get('revision') == profile['revision']:
        return profile
    assert_not_running(backbone, variant)
    archive = HERE / 'archived_runs' / profile['revision'] / backbone / variant
    if run.exists():
        if archive.exists():
            raise RuntimeError(f'Archive already exists; refusing to overwrite: {archive}')
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run), str(archive))
    run.mkdir(parents=True, exist_ok=True)
    payload = {
        **profile,
        'backbone': backbone,
        'variant': variant,
        'fresh_start': True,
        'archived_previous_run': str(archive) if archive.exists() else None,
    }
    marker.write_text(json.dumps(payload, indent=2) + '\n')
    (run / 'status.json').write_text(json.dumps({
        'status': 'PENDING_FRESH_TRAINING',
        'microbatch': profile['microbatch'],
        'gradient_accumulation': profile['gradient_accumulation'],
        'effective_batch': profile['microbatch'] * profile['gradient_accumulation'],
        'revision': profile['revision'],
    }, indent=2) + '\n')
    return profile


if __name__ == '__main__':
    for key, value in plan()['jobs'].items():
        if value.get('action') == 'restart':
            prepare_restart(*key.split('/'))
            print('PREPARED', key, value)
