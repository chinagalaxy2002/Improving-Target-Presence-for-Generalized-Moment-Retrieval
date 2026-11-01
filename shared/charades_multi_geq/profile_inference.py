"""Val-only CUDA latency measurement; never selects or updates a checkpoint."""
import copy,time
import numpy as np
import torch
from paths import *
from adapter import Adapter
from geq_model import RouteGEQ

def profile(ad,model,directory):
 marker=json.loads((directory/'TRAINED.json').read_text());assert sha(directory/'best.ckpt')==marker['checkpoint_sha256']
 state=torch.load(directory/'best.ckpt',map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True);model.eval()
 data=ad.dataset('val');batch=1 if ad.name=='flashvtg' else ad.bs
 metas,inputs,targets=ad.collate([data[i] for i in range(batch)])
 native=None
 if isinstance(model,RouteGEQ):
  native_adapter=Adapter(ad.name,directory.name.startswith('multi'))
  native_adapter._valid=inputs['src_vid_mask'].bool()
  native=native_adapter.base;native.load_state_dict(model.base.state_dict(),strict=True);native.eval()
 def time_forward(module):
  with torch.no_grad():
   for _ in range(25):module(**inputs)
   torch.cuda.synchronize();durations=[]
   for _ in range(100):
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record();module(**inputs);end.record();end.synchronize();durations.append(start.elapsed_time(end))
  return dict(mean_ms_per_batch=float(np.mean(durations)),median_ms_per_batch=float(np.median(durations)),p95_ms_per_batch=float(np.percentile(durations,95)),mean_ms_per_query=float(np.mean(durations))/batch)
 total=time_forward(model);baseline=time_forward(native) if native is not None else total
 result=dict(checkpoint_sha256=marker['checkpoint_sha256'],source_split='val',batch_size=batch,warmup_iterations=25,measured_iterations=100,total=total,matched_native_trunk=baseline,incremental_geq_ms_per_query=total['mean_ms_per_query']-baseline['mean_ms_per_query'] if native is not None else None,parameters=sum(p.numel() for p in model.parameters()),geq_parameters=sum(p.numel() for p in model.geq.parameters()) if isinstance(model,RouteGEQ) else 0,gpu=torch.cuda.get_device_name(),scope='CUDA forward compute only; excludes feature I/O, proposal export and metrics. Native comparator has matching trained trunk tensors and no GEQ hooks. Timing noise can make a difference negative.',time=time.time())
 dump(directory/'PROFILE.json',result);return result
