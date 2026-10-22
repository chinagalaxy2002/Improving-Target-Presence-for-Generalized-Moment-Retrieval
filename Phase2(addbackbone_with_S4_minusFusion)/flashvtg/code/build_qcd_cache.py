"""Freeze WorldCup Train full clip x unique-query cosine statistics and scores."""
import argparse
import json
from pathlib import Path

from environment import ROOT, ASSETS, read_rows, read_json, write_json
from qcd import masks, sha, worldcup
import numpy as np
import torch



def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--clip-model', type=Path, required=True, help='Original OpenAI CLIP ViT-B/32 TorchScript checkpoint matching the cached Lighthouse features')
    parser.add_argument('--output', type=Path, default=ROOT/'cache/qcd_a3.npz')
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Frozen cache exists; refusing overwrite')
    rows = [r for r in read_rows(ROOT/'data/train.jsonl') if worldcup(r)]
    protocol = read_json(ROOT/'configs/s4_protocol.json')
    max_l, clip_len = protocol['max_v_l'], protocol['clip_length']
    vids = sorted({r['vid'] for r in rows})
    queries = sorted({r['query'] for r in rows})
    query_index = {q: i for i, q in enumerate(queries)}
    paths = {}
    bank, offsets = [], [0]
    for vid in vids:
        stem = Path(vid).stem if Path(vid).suffix in ('.mp4','.mkv','.webm','.avi','.mov','.m4v') else vid
        arrays = []
        for directory in ('slowfast','clip'):
            path = ASSETS/directory/f'{stem}.npz'
            with np.load(path) as data:
                arrays.append(data['features'][:max_l].astype(np.float32))
            paths[str(path.relative_to(ASSETS))] = sha(path)
        length = min(map(len, arrays))
        visual = arrays[1][:length]
        if visual.shape[1] != 512 or length == 0 or not np.isfinite(visual).all():
            raise ValueError(f'Invalid visual feature: {vid}')
        norms = np.linalg.norm(visual, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise ValueError('Zero CLIP visual norm')
        bank.append(visual/norms)
        offsets.append(offsets[-1]+length)
    offsets = np.asarray(offsets, dtype=np.int64)
    visual = np.concatenate(bank)
    # Lighthouse caches ln_final token states BEFORE text_projection.
    # Use EOT = last non-padding token, then the original CLIP projection.
    model = torch.jit.load(str(args.clip_model), map_location='cpu').eval()
    projection = model.state_dict()['text_projection'].float().numpy()
    if projection.shape != (512,512):
        raise ValueError('Expected original CLIP ViT-B/32 text projection')
    del model
    texts = []
    for query in queries:
        row = next(r for r in rows if r['query'] == query)
        path = ASSETS/'clip_text'/f"qid{row['qid']}.npz"
        with np.load(path) as data:
            mask = data['attention_mask'].astype(bool)
            tokens = data['last_hidden_state'].astype(np.float32)
        n = int(mask.sum())
        if n < 2 or not mask[:n].all() or mask[n:].any() or tokens.shape != (77,512):
            raise ValueError(f'Invalid text cache: {path}')
        text = tokens[n-1] @ projection
        if not np.isfinite(text).all() or np.linalg.norm(text) <= 0:
            raise ValueError('Invalid projected text embedding')
        texts.append(text/np.linalg.norm(text))
        paths[str(path.relative_to(ASSETS))] = sha(path)
    scores = np.asarray(texts, dtype=np.float32) @ visual.T
    gt = np.zeros(scores.shape, dtype=bool)
    vid_index = {v: i for i,v in enumerate(vids)}
    for row in rows:
        j = vid_index[row['vid']]
        mask,_ = masks(row['relevant_windows'], int(offsets[j+1]-offsets[j]), clip_len, 0)
        gt[query_index[row['query']], offsets[j]:offsets[j+1]] |= mask
    positive, non = scores[gt], scores[~gt]
    if not len(positive) or not len(non):
        raise ValueError('Empty similarity statistics')
    lo,hi = float(np.percentile(non,10)), float(np.percentile(positive,60))
    if lo > hi:
        raise ValueError('Inverted QCD threshold interval')
    np.savez(output, clip=visual, scores=scores, offsets=offsets)
    coverage=[]
    for row in rows:
        if not row['relevant_windows']:
            continue
        q = query_index[row['query']]
        coverage.extend([bool(((scores[q,offsets[j]:offsets[j+1]] >= lo) & (scores[q,offsets[j]:offsets[j+1]] <= hi)).any())
                         for j,v in enumerate(vids) if v != row['vid']])
    metadata=dict(train_sha256=sha(ROOT/'data/train.jsonl'),cache_sha256=sha(output),
        clip_model=str(args.clip_model),clip_model_sha256=sha(args.clip_model),
        feature_sha256=paths,vids=vids,queries=queries,clip_len=clip_len,max_v_l=max_l,
        alpha=10,beta=60,theta_min=lo,theta_max=hi,
        statistics={name:dict(count=len(values),mean=float(values.mean()),std=float(values.std(ddof=0)))
                    for name,values in [('gt',positive),('non',non)]},
        train_worldcup_rows=len(rows),positive_rows=sum(bool(r['relevant_windows']) for r in rows),
        single_rows=sum(len(r['relevant_windows'])==1 for r in rows),multi_rows=sum(len(r['relevant_windows'])>1 for r in rows),
        donor_pair_nonempty_fraction=float(np.mean(coverage)),
        text_method='cached ln_final EOT @ original ViT-B/32 text_projection, then L2',
        statistics_policy='all WorldCup Train clips x unique exact query strings; GT union for annotated same-query same-video pairs; all other pairs non-GT')
    write_json(output.with_suffix('.json'),metadata)
    print(json.dumps({k:v for k,v in metadata.items() if k not in ('feature_sha256','queries','vids')},indent=2),flush=True)


if __name__=='__main__':
    main()
