#!/usr/bin/env python3
"""Verify the archive against its SHA256 manifest without accessing source experiments."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


if __name__=='__main__':
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    failures=[]
    for item in manifest['files']:
        p=ROOT/item['path']
        if not p.is_file() or p.stat().st_size!=item['bytes'] or digest(p)!=item['sha256']:
            failures.append(item['path'])
    result={'passed':not failures,'files_verified':len(manifest['files']),'failures':failures}
    print(json.dumps(result,indent=2))
    raise SystemExit(1 if failures else 0)
