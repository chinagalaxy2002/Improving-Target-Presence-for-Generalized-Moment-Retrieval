import argparse,json,csv
from pathlib import Path
import numpy as np
from core import read_jsonl,write_json,sha256
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();r=Path(a.root)
f=json.loads((r/'ffn_validation/val.json').read_text());dev=json.loads((r/'ffn_validation/development.json').read_text());selection=json.loads((r/'ffn_validation/selection.json').read_text());native=json.loads((r/'ffn_native/native_invariance.json').read_text());assert native['passed'] and native['rows']==465
causal=read_jsonl(r/'ffn_layer1/causal_scores.jsonl');errors={}
for n,j in [('ffn_a0.5',1),('ffn_a1',2)]:
 predictions={str(x['qid']):x for x in read_jsonl(r/f'ffn_forwards/{n}/train/predictions.jsonl')};err=max(abs(predictions[str(x['qid'])]['pred_exist_score']-x['scores'][j]) for x in causal);assert err<1e-5;errors[n]=err
lines=['# Expanded FlashVTG E1 experiments','', 'The head sweep and matched controls are in [the head report](report/REPORT.md). All 588 virtual hypotheses and failures are retained. The development-selected head failed on reused Val. No Test data was used.','', 'The FFN search also tested layers 0 and 1, 32 selected output columns, at suppression strengths 0.5 and 1.0. Layer 1 passed the development activation-intervention gate, so two FFN checkpoints were materialized. Layer 0 failed and was not edited.','',f"FFN development selection: `{selection['selected']}`; selection was written before evaluating Val. Actual checkpoint forwards agree with causal activation intervention within {max(errors.values()):.2g} probability. All native tensors on 465 Val queries are exactly unchanged for both FFN checkpoints and baseline.",'','| Model | Dev rescued | Dev FP/FN | Val rescued | Val FP/FN | AUROC | G1 @ .4/.5/.6 | mAP | Parameters | L2 |','|---|---:|---|---:|---|---:|---|---:|---:|---:|']
for n,x in f.items():
 d=dev[n];g='/'.join(str(x['metrics'][t]['G-mIoU@1']) for t in ('0.4','0.5','0.6'));lines.append(f"| {n} | {d['fn_loc_rescued']} | {d['confusion']['FP']}/{d['confusion']['FN']} | {x['fn_loc_rescued']} | {x['confusion']['FP']}/{x['confusion']['FN']} | {x['auroc']:.5f} | {g} | {x['metrics']['0.5']['mAP']} | {x['changed_parameters']} | {x['delta_l2']:.5f} |")
lines+=['','These small development changes are exploratory. Val has been reused across rounds and is not a fresh confirmatory set. Checkpoints are research candidates, not evidence of established general improvement. Official candidate exports are fixed to original E1; raw native outputs are independently verified.','', 'The two FFN checkpoint/delta/manifests are under `ffn_edits/`; qids and full G-mIoU@1/@3/confusion/AUROC results are in `ffn_validation/`. Every rejected head candidate and causal test is retained. No backward, optimizer, gradients or base-weight changes were used.']
(r/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
table=[]
for split,data in [('development',dev),('val',f)]:
 for name,x in data.items():
  row=dict(split=split,name=name,AUROC=x['auroc'],**x['confusion'],rescue=x['fn_loc_rescued'],rescue_rate=x['fn_loc_rescue_rate'],hard_null_false_accept_rate=x['hard_null_false_accept_rate'],changed_parameters=x['changed_parameters'],changed_fraction=x['changed_fraction'],delta_l2=x['delta_l2'])
  for t in ('0.4','0.5','0.6'):
   for k in (1,3):row[f'G{k}@{t}']=x['metrics'][t][f'G-mIoU@{k}']
  row['mAP']=x['metrics']['0.5']['mAP'];table.append(row)
with (r/'FFN_COMPARISON.csv').open('x') as handle:
 writer=csv.DictWriter(handle,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
paths=['sweep.json','report/audit.json','geometry_audit.json','ffn_validation/development.json','ffn_validation/selection.json','ffn_validation/val.json','ffn_native/native_invariance.json','ffn_layer0/causal.json','ffn_layer1/causal.json']
write_json(r/'FINAL_AUDIT.json',dict(source_sha256=json.loads((r/'sweep.json').read_text())['source_sha256'],test_used=False,gradient_used=False,ffn_intervention_vs_weight_error=errors,native_tensor_exact=True,head_virtual_candidates=588,ffn_selected=selection['selected'],inputs_sha256={p:sha256(r/p) for p in paths},code_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}))
print('finalized',selection['selected'],errors)
