import sys,torch
from paths import *
from native_factory import TargetAdapter
b=sys.argv[1];torch.set_num_threads(2)
ad=TargetAdapter(b,'cpu','charades',seed=2023)
old=torch.load(OUT/'runs'/b/'native_init.pt',map_location='cpu',weights_only=False)['model'];new=ad.base.state_dict()
assert set(old)==set(new)
assert all(torch.equal(v,new[k]) for k,v in old.items()),[k for k,v in old.items() if not torch.equal(v,new[k])]
dump(OUT/'reports'/f'native_factory_equivalence_{b}.json',dict(passed=True,all_native_tensors_bitwise_equal=True,tensors=len(old),tensor_summaries={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=__import__('hashlib').sha256(v.contiguous().numpy().tobytes()).hexdigest()) for k,v in old.items()}))
print('NATIVE_FACTORY_EQUIVALENT',b)
