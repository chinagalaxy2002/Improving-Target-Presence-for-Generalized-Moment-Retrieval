"""Create the queue readiness marker only after all required diagnostics pass."""
from paths import *
import time

def main():
 evidence={}
 assert json.loads((OUT/'data/audit.json').read_text())['passed']
 assert json.loads((OUT/'reports/flash_zero_point_guard.json').read_text())['passed']
 for b in BACKBONES:
  assert json.loads((OUT/'reports'/f'resume_audit_{b}.json').read_text())['passed']
  for m in METHODS:
   p=OUT/'reports/debug24'/b/m/'preflight.json';r=json.loads(p.read_text())
   assert r['passed'] and r['real_process_exit_and_restart'] and r['diagnostic_updates_restored'] and r['optimizer_reloaded_and_stepped']
   assert r['checks']['initial_state_restored'] and r['checks']['eval_auxiliary_forward_calls']==0 and r['checks']['native_export_and_metrics']['rows']==2
   assert r['physical_batch']==dict(moment_detr=8,eatr=32,qd_detr=32,flashvtg=8)[b]
   if m.endswith('frozen'):assert r['checks']['frozen_parameters_buffers_unchanged']
   if '_geq_' in m:assert r['checks']['geq_parameters_changed']
   evidence[b+'/'+m]=dict(passed=True,path=str(p.relative_to(OUT)),sha256=sha(p))
 dump(OUT/'reports/DEBUG24_READY.json',dict(passed=True,time=time.time(),configurations=len(evidence),real_process_restart_proofs=len(evidence),reports=evidence,flash_zero_point_guard_sha256=sha(OUT/'reports/flash_zero_point_guard.json'),scope='Diagnostic scratch parents; formal GEQ checks repeat at actual parent best. Diagnostic updates never enter formal training.'))
 print('ALL_24_PREFLIGHTS_SEALED')
if __name__=='__main__':main()
