"""CPU-only canonical initialization audit; exports digests, never tensor values."""
import hashlib,torch
from paths import *
from geq_model import GlobalEventQueryHead

def summaries(state):
 return {k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()) for k,v in state.items()}

def audit():
 torch.set_num_threads(2);result=dict(seed=2023,backbones={},all_native_inits_ready=True,all_four_geq_inits_ready=True)
 for b in BACKBONES:
  info={};native=OUT/'runs'/b/'native_init.pt'
  if not native.exists():result['all_native_inits_ready']=False;info['native']='PENDING'
  else:
   saved=torch.load(native,map_location='cpu',weights_only=False);assert saved['seed']==2023
   digest=hashlib.sha256()
   for k,v in saved['model'].items():digest.update(k.encode());digest.update(v.contiguous().numpy().tobytes())
   assert digest.hexdigest()==saved['sha256']
   factory=json.loads((OUT/'reports'/f'native_factory_equivalence_{b}.json').read_text());assert factory['passed'] and factory['all_native_tensors_bitwise_equal']
   info['native']=dict(snapshot_sha256=sha(native),state_sha256=saved['sha256'],tensors=summaries(saved['model']))
  geq=OUT/'runs'/b/'geq_init.pt'
  if not geq.exists():result['all_four_geq_inits_ready']=False;info['geq']='PENDING'
  else:
   saved=torch.load(geq,map_location='cpu',weights_only=False);assert saved['seed']==2023
   depth=len({int(k.split('.')[1]) for k in saved['model'] if k.startswith('video_reads.')})
   with torch.random.fork_rng(devices=[]):torch.manual_seed(2023);expected=GlobalEventQueryHead(depth).state_dict()
   assert expected.keys()==saved['model'].keys() and all(torch.equal(v,saved['model'][k]) for k,v in expected.items())
   info['geq']=dict(snapshot_sha256=sha(geq),encoder_depth=depth,bitwise_matches_fresh_seed2023=True,tensors=summaries(saved['model']))
  result['backbones'][b]=info
 dump(OUT/'reports/initialization_audit.json',result);return result
if __name__=='__main__':
 result=audit();print(json.dumps({k:v for k,v in result.items() if k!='backbones'}))
