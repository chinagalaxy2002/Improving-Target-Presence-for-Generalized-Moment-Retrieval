"""Fresh-process optimization check; diagnostic updates never touch formal checkpoints."""
import sys,torch
from paths import *
from engine_utils import seed,state_hash
from adapter import Adapter
from geq_model import RouteGEQ
from runner import preflight,evaluate
b,m=sys.argv[1:3];torch.set_num_threads(4);seed();ad=Adapter(b,m.startswith('multi'));model=ad.base
if '_geq_' in m:
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(2023);model=RouteGEQ(model,b,m.rsplit('_',1)[1]).cuda()
groups=[dict(params=[p for p in ad.base.parameters() if p.requires_grad],lr=1e-5 if '_geq_' in m else 3e-5)]
if '_geq_' in m:groups.append(dict(params=list(model.geq.parameters()),lr=3e-5))
optimizer=torch.optim.AdamW([g for g in groups if len(list(g['params']))],weight_decay=1e-4)
d=OUT/'reports/debug24'/b/m;d.mkdir(parents=True,exist_ok=True);before=state_hash(model);preflight(ad,model,optimizer,m,d);assert state_hash(model)==before
p=d/'preflight.json';record=json.loads(p.read_text());record.update(backbone=b,method=m,physical_batch=ad.bs,amp=False,peak_memory_bytes=torch.cuda.max_memory_allocated(),initial_state_sha256=before,parent='Diagnostic scratch parent; formal tasks repeat against actual Val-best',formal_weights_modified=False)
dump(p,record);print('DEBUG_PASS',b,m,flush=True)
