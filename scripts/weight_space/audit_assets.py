#!/usr/bin/env python3
"""Hash Train/Val labels and only their CLIP, SlowFast and text feature assets."""
import argparse,json
from pathlib import Path
from core import read_jsonl,sha256,write_json
p=argparse.ArgumentParser();p.add_argument('--legacy-code',required=True);p.add_argument('--features',required=True);p.add_argument('--output',required=True);a=p.parse_args()
code=Path(a.legacy_code);root=Path(a.features);assets=set();labels={};rows={}
for split in ('train','val'):
 path=code/'data'/f'{split}.jsonl';data=read_jsonl(path);labels[split]={'path':str(path.resolve()),'sha256':sha256(path),'queries':len(data)};rows[split]=data
 for row in data:
  stem=Path(row['vid']).stem
  for kind in ('slowfast','clip'):
   candidates=[root/kind/f'{stem}{ext}' for ext in ('.npz','.npy')];matches=[x for x in candidates if x.is_file()]
   if not matches:raise FileNotFoundError(candidates)
   assets.add(matches[0])
  assets.add(root/'clip_text'/f"qid{row['qid']}.npz")
assert {r['vid'] for r in rows['train']}.isdisjoint(r['vid'] for r in rows['val'])
files={str(x.resolve()):sha256(x) for x in sorted(assets)}
write_json(a.output,{'labels':labels,'feature_root':str(root.resolve()),'feature_count':len(files),'features_sha256':files,'train_val_videos_disjoint':True,'test_used':False})
print('hashed features',len(files))
