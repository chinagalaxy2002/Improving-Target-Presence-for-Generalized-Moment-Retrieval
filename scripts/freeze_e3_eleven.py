import json,subprocess
from pathlib import Path
from soccergmr import checkpoint as ck
from soccergmr.io import dump_json,sha256
from soccergmr.config import load_config
from soccergmr.data import SoccerDataset
from soccergmr.backbones.registry import SPECS,get_spec
R=Path(__file__).resolve().parents[1];RUN=R/'runs/e3_eleven/formal';(RUN/'logs').mkdir(exist_ok=True,parents=True)
source=ck.source_identity();accept={}
for n in SPECS:
 p=R/f'runs/e3_eleven/acceptance/{n}/complete.json';x=json.loads(p.read_text());assert x['passed'] and x['source']==source,n;accept[n]=str(p.relative_to(R))
assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Commit final implementation before freezing'
files=[p for root in ['src','configs','scripts','tests'] for p in sorted((R/root).rglob('*')) if p.is_file() and p.suffix in ['.py','.json','.yaml','.yml']]
files += [R/'E3_ELEVEN_PROTOCOL.md',R/'BACKBONE_IMPORT.json',RUN/'moment_detr/REUSE.json']+[R/p for p in accept.values()]
c=load_config(R/'configs/experiments/moment_detr.json');spec=get_spec('moment_detr');training=ck.asset_identity([SoccerDataset(c,s,spec) for s in ['train','val']]);testing=ck.asset_identity([SoccerDataset(c,s,spec) for s in ['test_standard','test_full']])
result=dict(source=source,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files={str(p.relative_to(R)):sha256(p) for p in files},train_val_assets=training,test_assets=testing,acceptance=accept,max_compute_per_gpu=3,moment_retrained=False,native_architectures=list(SPECS),new_training_stages=20,new_test_evaluations=40,reused_moment_tests=4)
dump_json(RUN/'FREEZE.json',result);print('FROZEN',result['commit'],source)
