"""Portable two-model environment. Frozen-prediction reporting needs no external assets."""
from pathlib import Path
import hashlib,json,os,random,sys
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'baseline_source'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key,value in {'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','MKL_THREADING_LAYER':'GNU','MPLCONFIGDIR':str(ROOT/'tmp/matplotlib')}.items():os.environ.setdefault(key,value)
LOCAL=json.loads((ROOT/'local_paths.json').read_text()) if (ROOT/'local_paths.json').exists() else {}
ASSETS=Path(os.getenv('GMR_ASSETS_DIR') or LOCAL.get('assets_dir') or '/path/to/Soccergmr')
E0_CHECKPOINTS={2026:Path(os.getenv('GMR_E0_CHECKPOINT') or LOCAL.get('baseline_checkpoint') or '/path/to/E0_s2026/best.ckpt')}
S4_CHECKPOINT=Path(os.getenv('GMR_S4F_CHECKPOINT') or LOCAL.get('s4_fusion_checkpoint') or '/path/to/S4_noFusion/best.ckpt')
for d in ['tmp','cache','reports','figures']:(ROOT/d).mkdir(parents=True,exist_ok=True)
import numpy as np
import torch

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read_json(path):return json.loads(Path(path).read_text())
def read_rows(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
def write_json(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def write_rows(path,rows):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));tmp.replace(p)
def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def options(job,device):
 from types import SimpleNamespace
 import nncore
 values=read_json(ROOT/'configs'/('baseline_protocol.json' if job.get('variant')=='E0' else 's4_protocol.json'))
 values.update(seed=2026,device=torch.device(device),config=str(ROOT/'baseline_source/configs/flash_vtg_gmr/model.py'),train_path=str(ROOT/'data/train.jsonl'),eval_path=str(ROOT/'data/val.jsonl'),eval_split_name='val',v_feat_dirs=[str(ASSETS/'slowfast'),str(ASSETS/'clip')],t_feat_dir=str(ASSETS/'clip_text'),n_epoch=job['epochs'],lr=job['lr'],wd=job['wd'],lr_drop=job['lr_drop'],grad_clip=job['grad_clip'],use_plurality_gating=False)
 opt=SimpleNamespace(**values);opt.cfg=nncore.Config.from_file(opt.config);return opt
def make_dataset(opt,split,smoke=False):
 from training.flash_vtg_gmr.dataset import StartEndDataset
 assert split in ('train','val','test')
 path=ROOT/'data'/f'{split}.jsonl'
 data=StartEndDataset(dset_name=opt.dset_name,data_path=str(path),v_feat_dirs=opt.v_feat_dirs,q_feat_dir=opt.t_feat_dir,q_feat_type=opt.q_feat_type,max_q_l=opt.max_q_l,max_v_l=opt.max_v_l,ctx_mode=opt.ctx_mode,normalize_v=True,normalize_t=True,load_labels=split=='train',clip_len=opt.clip_length,max_windows=opt.max_windows,span_loss_type=opt.span_loss_type,mr_only=True,keep_empty_gt=True)
 assert len(data)==len(read_rows(path)), 'Dataset silently filtered rows'
 return data
