"""Real process termination/restart proof for each of the 24 model configurations."""
import sys,copy,torch
from paths import *
from engine_utils import seed,save,rng,restore,state_hash
from adapter import Adapter
from geq_model import RouteGEQ
from runner import loss
b,m=sys.argv[1:3];torch.set_num_threads(4);seed();ad=Adapter(b,m.startswith('multi'));model=ad.base
if '_geq_' in m:
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(2023);model=RouteGEQ(model,b,m.rsplit('_',1)[1]).cuda()
groups=[dict(params=[p for p in ad.base.parameters() if p.requires_grad],lr=1e-5 if '_geq_' in m else 3e-5)]
if '_geq_' in m:groups.append(dict(params=list(model.geq.parameters()),lr=3e-5))
opt=torch.optim.AdamW([g for g in groups if len(list(g['params']))],weight_decay=1e-4)
d=OUT/'reports/debug24'/b/m;statefile=d/'interrupt_state.pt';report=json.loads((d/'preflight.json').read_text());assert report['passed']
resuming=statefile.exists()
if resuming:
 saved=torch.load(statefile,map_location='cpu',weights_only=False);model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer']);restore(saved['rng']);assert state_hash(model)==saved['updated_hash']
 initial=saved['initial'];initial_hash=saved['initial_hash']
else:initial=copy.deepcopy(model.state_dict());initial_hash=state_hash(model)
ds=ad.dataset('train');ids=[next(i for i,r in enumerate(ds.data) if r['relevant_windows']),next(i for i,r in enumerate(ds.data) if not r['relevant_windows'])]
model.train();meta,inputs,targets=ad.collate([ds[i] for i in ids]);opt.zero_grad(set_to_none=True);value=loss(ad,model,model(**inputs),targets,meta,1);assert torch.isfinite(value);value.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.1);opt.step()
assert state_hash(model)!=initial_hash
if not resuming:
 save(statefile,dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng(),initial=initial,initial_hash=initial_hash,updated_hash=state_hash(model)));print('INTERRUPT_AFTER_REAL_OPTIMIZER_STEP',b,m,flush=True);sys.exit(75)
model.load_state_dict(initial,strict=True);assert state_hash(model)==initial_hash
report.update(real_process_exit_and_restart=True,optimizer_reloaded_and_stepped=True,post_resume_loss=float(value),diagnostic_updates_restored=True)
dump(d/'preflight.json',report);statefile.unlink();print('RESUME_DEBUG_PASS',b,m,flush=True)
