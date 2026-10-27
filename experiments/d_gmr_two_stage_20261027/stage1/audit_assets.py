"""Read-only audit of every split, external feature, and original checkpoint."""
import json
from pathlib import Path
import numpy as np
from common import ROOT, OLD, ASSETS, BACKBONES, rows, source, sha, dump

def main():
    labels={s:rows(ROOT/f'data/{s}.jsonl') for s in ('train','val','test')}
    seen=set();videos=set();queries=set();missing=[];bad=[]
    for s,rs in labels.items():
        original=rows(ASSETS.parent/f'data/label/Standard/{s}.jsonl')
        assert rs==original, f'{s} differs from native Standard labels'
        assert len(rs)==dict(train=4138,val=465,test=1036)[s]
        ids={r['qid'] for r in rs};assert len(ids)==len(rs) and not ids&seen;seen|=ids
        for r in rs:
            assert r['duration']>0
            assert all(0<=a<b<=r['duration'] for a,b in r['relevant_windows'])
            videos.add(Path(r['vid']).stem);queries.add(r['qid'])
    lengths={};feature_records=[]
    for vid in sorted(videos):
        counts=[]
        for kind,width in (('slowfast',2304),('clip',512)):
            p=ASSETS/kind/f'{vid}.npz'
            if not p.exists():missing.append(str(p));continue
            with np.load(p) as z:
                x=z['features'];counts.append(len(x))
                if x.ndim!=2 or not 0<len(x)<=75 or x.shape[-1]!=width or not np.isfinite(x).all():bad.append(str(p))
            feature_records.append(dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns))
        if counts:lengths[vid]=min(counts)
    for qid in sorted(queries,key=str):
        p=ASSETS/'clip_text'/f'qid{qid}.npz'
        if not p.exists():missing.append(str(p));continue
        with np.load(p) as z:
            x=z['last_hidden_state']
            if x.ndim!=2 or not len(x)>0 or x.shape[-1]!=512 or not np.isfinite(x).all():bad.append(str(p))
        feature_records.append(dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns))
    checks={}
    for b in BACKBONES:
        p=source(b);expected=json.loads((OLD/f'results/{b}/gmr/CONFIG.json').read_text())['initialization']['sha256']
        checksum=sha(p);assert checksum==expected
        checks[b]=dict(path=str(p),sha256=checksum,verified_against_existing_E0=True)
    report=dict(passed=not missing and not bad,splits={s:dict(total=len(rs),positive=sum(bool(r['relevant_windows']) for r in rs),max_gt=max(len(r['relevant_windows']) for r in rs),sha256=sha(ROOT/f'data/{s}.jsonl')) for s,rs in labels.items()},
                missing=missing,invalid_features=bad,checkpoints=checks,video_lengths=lengths,features=feature_records,
                encoder_policy='CLIP/SlowFast feature files are outside computation graph; all native network parameters require gradients')
    dump(ROOT/'reports/ASSET_AUDIT.json',report)
    assert report['passed'],report
    print('ALL_ASSETS_VERIFIED',len(videos),len(queries))

if __name__=='__main__':main()
