import argparse,json,csv
from pathlib import Path
from core import write_json,sha256
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();r=Path(a.root);o=r/'report';o.mkdir(exist_ok=False)
s=json.loads((r/'sweep.json').read_text());dev=json.loads((r/'validation/development.json').read_text());val=json.loads((r/'validation/val.json').read_text());control=json.loads((r/'control_validation/val.json').read_text());native=json.loads((r/'native_check/native_invariance.json').read_text());assert native['passed'] and not native.get('limited',False)
ffn={str(i):json.loads((r/f'ffn_layer{i}/causal.json').read_text()) for i in (0,1)}
rows=[]
for split,data in [('train_development',dev),('val',val),('val_control',control)]:
 for name,x in data.items():
  z=dict(split=split,name=name,AUROC=x['auroc'],**x['confusion'],rescue=x['fn_loc_rescued'],rescue_rate=x['fn_loc_rescue_rate'],hard_null_false_accept_rate=x['hard_null_false_accept_rate'],changed_parameters=x['changed_parameters'],delta_l2=x['delta_l2'])
  for t in ('.4','.5','.6'):
   for k in (1,3):z[f'G{k}@{t}']=x['metrics']['0'+t][f'G-mIoU@{k}']
  z['mAP']=x['metrics']['0.5']['mAP'];rows.append(z)
with (o/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (o/'all_588_candidates.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(s['candidates'][0]));w.writeheader();w.writerows(s['candidates'])
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(9,6))
for family in sorted({x['family'] for x in s['candidates']}):
 xs=[x for x in s['candidates'] if x['family']==family];ax.scatter([x['FP']-s['baseline']['FP'] for x in xs],[x['rescued'] for x in xs],s=15,alpha=.6,label=family)
ax.axvline(0,color='black');ax.set(xlabel='Development FP change',ylabel='Development FN-Loc rescues',title='588 gradient-free analytic interventions');ax.legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(o/'sweep.png',dpi=160)
lines=['# FlashVTG E1 expanded exploration','',f"588 virtual analytic interventions; {sum(x['eligible'] for x in s['candidates'])} meet development constraints. Eight different families materialized as checkpoints. Discovery creates all directions; development selects; Val reports only. Prior-round Val had already been viewed, so this is reused Val, not a fresh confirmatory holdout.",'','Methods: null-variation PCA removal, A/D mean protection, covariance-conditioned directions, sparse coordinate edits and boundary-local prototypes. Linear solves are closed-form data-derived edits; no backward or optimizer. These are inspired by diagnostic reasoning in https://rikiyat.github.io/blog/weight-space-programming/, not a replication of its LLM experiments.','',f"Frozen development selection: `{s['selected']}`. It rescues 10 development FN-Loc queries but newly rejects 16 previously accepted positive queries. Val does not confirm the benefit: FP 9→13, FN 156→156, AUROC 0.69968→0.67966, G-mIoU@1 at 0.5 51.94→50.54. Retain as failed-generalization evidence; do not deploy or reselect using Val.",'','| Candidate | Dev rescue | Dev FP/FN | Val rescue | Val FP/FN | Val G1 @ .5 |','|---|---:|---|---:|---|---:|']
for name,x in dev.items():
 y=val[name];lines.append(f"| {name} | {x['fn_loc_rescued']} | {x['confusion']['FP']}/{x['confusion']['FN']} | {y['fn_loc_rescued']} | {y['confusion']['FP']}/{y['confusion']['FN']} | {y['metrics']['0.5']['G-mIoU@1']} |")
lines+=['','All 588 direction vectors/strengths/centers and failed simulated trials remain in `sweep.json` and `directions.npz`; actual checkpoint edits, differences and manifests are in `edits/` and `controls/`. Complete qid changes are in validation JSON. Native outputs are tensor-exact on all 465 Val queries for all nine materialized head models. Official metrics freeze the original candidate export; mAP stays 28.15.','', 'FFN interventions:']
for layer,c in ffn.items():
 for x in c['results']:lines.append(f"- Layer {layer}, 32 columns, strength {x['strength']}: rescue {x['rescued']}, FP {x['fp']}, FN {x['fn']}; causal pass {x['pass_causal_gate']}.")
(o/'REPORT.md').write_text('\n'.join(lines)+'\n')
paths=['sweep.json','directions.npz','validation/development.json','validation/selection.json','validation/val.json','control_validation/val.json','native_check/native_invariance.json','ffn_layer0/causal.json','ffn_layer1/causal.json']
write_json(o/'audit.json',dict(test_used=False,gradient_used=False,source_sha256=s['source_sha256'],selected=s['selected'],virtual_candidates=len(s['candidates']),materialized_head_models=len(val),native_exact=True,ffn_passed={k:v['passed'] for k,v in ffn.items()},inputs_sha256={p:sha256(r/p) for p in paths},outputs_sha256={p.name:sha256(p) for p in o.iterdir() if p.is_file()}))
