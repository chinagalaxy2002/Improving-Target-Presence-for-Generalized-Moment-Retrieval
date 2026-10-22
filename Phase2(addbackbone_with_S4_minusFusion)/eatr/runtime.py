"""Strict native/GMR initialization, stable snapshotting, and fresh PEFT optimization."""
import hashlib,json,os,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import torch
from models.build_model import build_model
from models.module import MLP
from model import S4EaTR
DEFAULT_SOURCE=Path('/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/eatr/gmr/best.ckpt')

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def write_json(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def validate_baseline(state,kind=None):
 assert 'model' in state and 'config' in state,'Require trained Phase 1 model/config; smoke checkpoints are not baselines'
 assert state.get('epoch',-1)>=0,'Untrained smoke checkpoints cannot verify GMR initialization'
 gmr=bool(state['config'].get('use_exist_head',False))
 assert gmr==any(k.startswith('exist_head.') for k in state['model']),'GMR head/config mismatch'
 if kind is not None:assert gmr==(kind=='gmr'),'Checkpoint provenance does not match architecture'
 assert state['config']['hidden_dim']==256 and state['config']['v_feat_dim']==2818
 assert state['config']['feature_order']==['slowfast2304','clip512','tef2']
 return gmr

def snapshot(source=DEFAULT_SOURCE,provenance='gmr'):
 assert provenance in ('original','gmr')
 source=Path(source).resolve();dest=ROOT/'snapshots';dest.mkdir(exist_ok=True)
 temp=dest/f'.copy-{os.getpid()}.tmp'
 for attempt in range(5):
  with source.open('rb') as src,temp.open('wb') as target:
   before=os.fstat(src.fileno());h=hashlib.sha256()
   for block in iter(lambda:src.read(1024*1024),b''):h.update(block);target.write(block)
   after=os.fstat(src.fileno())
  if (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns) and digest(source)==h.hexdigest():break
  temp.unlink(missing_ok=True)
 else:raise RuntimeError('Live checkpoint changed during snapshot')
 state=torch.load(temp,map_location='cpu',weights_only=False)
 try:gmr=validate_baseline(state,provenance)
 except Exception:
  temp.unlink(missing_ok=True);raise
 final=dest/f'{provenance}-{h.hexdigest()}.ckpt'
 if final.exists():temp.unlink()
 else:temp.replace(final);final.chmod(0o444)
 manifest={'source_checkpoint':str(source),'snapshot':str(final),'sha256':digest(final),'checkpoint_kind':provenance,
           'verified_gmr_baseline':gmr,'epoch':state['epoch'],'best_val_map':state.get('best'),
           'optimizer_policy':'fresh optimizer and scheduler; never restore baseline states'}
 write_json(dest/f'{h.hexdigest()}.json',manifest);write_json(ROOT/'initialization.json',manifest)
 return final

def build(device='cpu',checkpoint=None,config=None):
 config=config or json.loads((ROOT/'config.json').read_text())
 checkpoint=Path(checkpoint or json.loads((ROOT/'initialization.json').read_text())['snapshot'])
 checkpoint_sha=digest(checkpoint)
 manifest_path=ROOT/'snapshots'/f'{checkpoint_sha}.json'
 assert manifest_path.exists(),'Use --source-checkpoint/prepare_initial_snapshot.py to create an audited immutable snapshot first'
 manifest=json.loads(manifest_path.read_text())
 assert Path(manifest['snapshot']).resolve()==checkpoint.resolve() and manifest['sha256']==checkpoint_sha
 state=torch.load(checkpoint,map_location='cpu',weights_only=False);gmr=validate_baseline(state,manifest['checkpoint_kind'])
 assert digest(checkpoint)==checkpoint_sha,'Snapshot changed during initialization'
 opt=SimpleNamespace(**state['config']);opt.device=torch.device('cpu')
 base,_=build_model(opt)
 # Exact Phase 1 GMR factory: baseline MLP existence head on final decoder max pool.
 # Preserve this frozen checkpoint head, then replace its output with the independent S4 global head.
 if gmr:base.exist_head=MLP(opt.hidden_dim,opt.hidden_dim,1,2)
 base.load_state_dict(state['model'],strict=True)
 before={n:t.clone() for n,t in base.state_dict().items()}
 model=S4EaTR(base,config);actual=model.base.state_dict()
 for name,tensor in before.items():
  mapped=name
  if name.startswith('transformer.encoder.layers.') and ('.linear1.' in name or '.linear2.' in name):
   stem,leaf=name.rsplit('.',1);mapped=f'{stem}.base_linear.{leaf}'
  assert torch.equal(tensor,actual[mapped]),name
 assert all(torch.count_nonzero(p)==0 for n,p in model.named_parameters() if n.endswith('lora_B'))
 counts={'total':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
         'lora':sum(p.numel() for n,p in model.named_parameters() if '.lora_' in n),'frozen':sum(p.numel() for p in model.parameters() if not p.requires_grad)}
 return model.to(device),{'strict_base_load':True,'preserved_tensors':len(before),'exact_shared_weights':True,'counts':counts,'checkpoint':str(checkpoint),'sha256':digest(checkpoint),'verified_gmr_baseline':gmr,'initialization':manifest}

def optimizer_for(model,config):
 groups=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and '.lora_' not in n],'lr':config['lr']},
         {'params':[p for n,p in model.named_parameters() if p.requires_grad and '.lora_' in n],'lr':config['lora_lr']}]
 optimizer=torch.optim.AdamW(groups,weight_decay=config['wd'],foreach=False)
 return optimizer,torch.optim.lr_scheduler.StepLR(optimizer,config['lr_drop'],gamma=.5)
