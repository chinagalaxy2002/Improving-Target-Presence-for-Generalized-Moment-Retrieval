import sys,json,torch,math
from pathlib import Path
root=Path('/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/charades/multiroute_geq_seed2023_20261101');sys.path.insert(0,str(root/'code'))
from native_factory import TargetAdapter
from torch.utils.data import DataLoader
torch.set_num_threads(2);ad=TargetAdapter('flashvtg','cpu','charades',seed=2023)
state=torch.load(root/'runs/flashvtg/gmr/last.ckpt',map_location='cpu',weights_only=False)
g=torch.Generator();g.set_state(state['sampler_epoch_start']);loader=DataLoader(list(range(27788)),batch_size=8,shuffle=True,generator=g)
rows=[json.loads(l) for l in (root/'data/train.jsonl').read_text().splitlines()];lengths=json.loads((root/'data/audit.json').read_text())['video_lengths']
for step,ids in enumerate(loader):
 if step<state['cursor']:continue
 if step>state['cursor']+100:break
 batch=[rows[i] for i in ids.tolist()];pos=[r for r in batch if r['relevant_windows']]
 if not pos:continue
 n=max(lengths[r['vid']] for r in batch);pyramid=[torch.zeros(1,math.ceil(n/s),256) for s in (1,2,4,8)];point=ad.base.generator(pyramid)
 targets=[ad.criterion.loss.get_target_single(point,torch.tensor(r['relevant_windows'])*.5,torch.ones(1,1,dtype=torch.long)) for r in pos]
 cls=torch.stack([x[0] for x in targets]);reg=torch.stack([x[1] for x in targets]);msk=cls.unsqueeze(-1).expand(-1,-1,2).bool()
 if not msk.any():
  src=torch.ones_like(reg,requires_grad=True);out={};ad.criterion.loss.loss_reg({'out_coord':src},out,cls,reg)
  if torch.isfinite(out['loss_reg']):
   assert out['loss_reg'].item()==0;out['loss_reg'].backward();assert torch.isfinite(src.grad).all() and not src.grad.any()
  evidence=dict(epoch=state['epoch'],step_zero_based=step,positive_samples=[dict(evaluation_id=r['evaluation_id'],vid=r['vid'],gt=r['relevant_windows']) for r in pos],regression_points=int(msk.sum()),loss=float(out['loss_reg']),finite=bool(torch.isfinite(out['loss_reg'])))
  (root/'reports/flash_zero_reg_reproduction.json').write_text(json.dumps(evidence,indent=2)+'\n');print(evidence);break
else:print('No matching zero-point batch')
