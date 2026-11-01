import os,json,hashlib
from pathlib import Path
KIT=Path(__file__).resolve().parent
PROJECT=Path(os.getenv('CHARADES_PROJECT','/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval'))
OUT=Path(os.getenv('CHARADES_OUTPUT',str(PROJECT/'charades/multiroute_geq_seed2023_20261101')))
OLD=PROJECT/'charades/d_gmr_two_stage/runtime'
SPLITS=PROJECT/'charades/d_gmr_two_stage/dataset/charades/splits/strict_seed2026'
FEATURES=PROJECT/'charades/d_gmr_two_stage/dataset/charades/features/lighthouse_bn45'
BACKBONES=('moment_detr','eatr','qd_detr','flashvtg')
METHODS=('gmr','multi','gmr_geq_frozen','gmr_geq_full','multi_geq_frozen','multi_geq_full')
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 with t.open('w') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False,default=str);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def jsonl(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp')
 with t.open('w') as f:
  for x in v:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
  f.flush();os.fsync(f.fileno())
 os.replace(t,p)
