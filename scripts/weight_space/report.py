#!/usr/bin/env python3
"""Render a complete, non-selective table and audit from frozen results."""
import argparse,csv,json,platform
from pathlib import Path
import numpy as np
from core import read_jsonl,sha256,write_json,label
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
r=Path(a.root);o=Path(a.output);o.mkdir(parents=True,exist_ok=False)
analysis=json.loads((r/'analysis/analysis.json').read_text());dev=json.loads((r/'validation/development.json').read_text());val=json.loads((r/'validation/val.json').read_text());selection=json.loads((r/'validation/selection.json').read_text())
assert sha256(r/'validation/development.json')==selection['development_sha256']
check=json.loads((r/'native_check/native_invariance.json').read_text());assert check['passed'] and not check.get('limited',False)
headcheck=json.loads((r/'head_math/head_math.json').read_text());bridgecheck=json.loads((r/'bridge_check/verification.json').read_text());analyticcheck=json.loads((r/'bridge_check/analytic_verification.json').read_text());assert headcheck['passed'] and bridgecheck['passed'] and analyticcheck['passed']
causal=json.loads((r/'causal_ffn/causal.json').read_text());assert set(dev)==set(val)==set(check['candidates'])
assert check['rows']==len(read_jsonl(r/'val/rows.jsonl')) and all(n==check['rows'] for n in check['candidates'].values())
norm_matching={}
for name,x in dev.items():
 if x['kind']=='direction':
  random_name=name.replace('direction_','random_',1);assert random_name in dev
  random_norm=dev[random_name]['delta_l2'];difference=abs(x['delta_l2']-random_norm)
  assert difference<1e-7,(name,difference)
  norm_matching[name]={'direction_l2':x['delta_l2'],'random_l2':random_norm,'absolute_difference':difference}
cols=['name','decision','split','auroc','TP','FN','FP','TN','fn_loc_count','fn_loc_rescued','fn_loc_rescue_rate','hard_null_count','hard_null_false_accept_rate','g1_0.4','g3_0.4','g1_0.5','g3_0.5','g1_0.6','g3_0.6','map','changed_parameters','changed_fraction','delta_l2','native_exact']
table=[]
for name in dev:
 for split,results in [('train_development',dev),('val_independent',val)]:
  x=results[name];row={'name':name,'decision':dev[name]['development_decision'],'split':split,'auroc':x['auroc'],**x['confusion'],**{k:x[k] for k in ('fn_loc_count','fn_loc_rescued','fn_loc_rescue_rate','hard_null_count','hard_null_false_accept_rate','changed_parameters','changed_fraction','delta_l2')},'map':x['metrics']['0.5']['mAP'],'native_exact':bool(check['candidates'][name]==check['rows'])}
  for t in ('0.4','0.5','0.6'):
   for k in (1,3):row[f'g{k}_{t}']=x['metrics'][t][f'G-mIoU@{k}']
  table.append(row)
with (o/'comparison.csv').open('x',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader();writer.writerows(table)
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,3,figsize=(15,8))
for j,(split,title) in enumerate([('train','Train, all videos'),('val','Val, disjoint videos')]):
 rows=read_jsonl(r/split/'rows.jsonl');groups={g:[x for x in rows if label(x,analysis['tau'],analysis['hard_null_threshold'])==g] for g in 'ABCD'}
 for g,items in groups.items():
  for ax,key in ((axes[j,0],'probability'),(axes[j,1],'logit')):
   if items:ax.hist([x[key] for x in items],bins=30,alpha=.45,label=f'{g} n={len(items)}',density=True)
 axes[j,0].axvspan(.4,.6,color='gray',alpha=.12);axes[j,0].set_xlabel('GEQ probability');axes[j,0].set_title(title);axes[j,0].legend(fontsize=7);axes[j,1].set_xlabel('GEQ logit');axes[j,1].set_title(title)
 for g,items in groups.items():
  near=[x for x in items if .4<=x['probability']<=.6];axes[j,2].scatter([x['probability'] for x in near],[x['candidate_score'] for x in near],s=10,alpha=.55,label=g)
 axes[j,2].axhline(analysis['hard_null_threshold'],color='black',linestyle='--');axes[j,2].set(xlabel='GEQ probability, 0.4–0.6',ylabel='Top-1 candidate score',title=title);axes[j,2].legend()
fig.tight_layout();fig.savefig(o/'activation_diagnostics.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,6))
for name,x in dev.items():
 if name=='unedited':continue
 marker={'direction':'o','bias_only':'s','random':'^','ffn':'D'}[x['kind']];ax.scatter(x['confusion']['FP']-dev['unedited']['confusion']['FP'],x['fn_loc_rescued'],marker=marker,label=name)
ax.axvline(0,color='black',linewidth=.8);ax.set(xlabel='Change in false accepts on Train development',ylabel='FN-Loc rescues on Train development',title='All candidate edits; selection excludes increased false accepts')
ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(o/'development_tradeoffs.png',dpi=160);plt.close(fig)
lines=['# FlashVTG E1 weight-space repair','',f"Selection: **{selection['selected'] or 'none'}**. Rule fixed before Val; all candidate results are retained.",'','Historical E1 is a fixed epoch-4 checkpoint; it is not the current branch’s ten-patience trained checkpoint. Candidate scores in the official metric input are frozen from the unedited historical export because its native post-processing scales candidate scores by the existence score. The native model tensors were independently re-run and checked exactly on every Val query.','',f"Hard null: top candidate score >= {analysis['hard_null_threshold']:.4f} (75th percentile among Train discovery null queries). Classification uses score > {analysis['tau']}. AUROC is 0–1; official G-mIoU and mAP are 0–100.",'','| Candidate | Development decision | Dev rescue | Dev FP | Val rescue | Val FP | Val G1 @ .4/.5/.6 | Val mAP | Changed parameters | L2 |','|---|---|---:|---:|---:|---:|---|---:|---:|---:|']
for name in dev:
 x=dev[name];y=val[name];g='/'.join(f"{y['metrics'][t]['G-mIoU@1']:.2f}" for t in ('0.4','0.5','0.6'))
 lines.append(f"| {name} | {x['development_decision']} | {x['fn_loc_rescued']}/{x['fn_loc_count']} | {x['confusion']['FP']} | {y['fn_loc_rescued']}/{y['fn_loc_count']} | {y['confusion']['FP']} | {g} | {y['metrics']['0.5']['mAP']:.2f} | {y['changed_parameters']} | {y['delta_l2']:.4g} |")
lines += ['','The CSV has the complete G-mIoU@1/@3 values for each threshold, all confusion counts, AUROC, hard-null rates and edit sizes. Per-query lists and all rejection reasons are in `validation/development.json` and `validation/val.json`. Controls are retained for comparison and are not eligible for selection.','',f"FFN activation intervention passed: **{causal['passed']}**. See `causal_ffn/causal.json` for every tested strength; no FFN edit may be produced without a passing intervention.",'','The earlier first-pass results are retained in the parent directory. Their random control matched the weight norm only; this run matches the full weight-and-bias edit norm. The direction candidates and the development selection rule were fixed before seeing Val, and did not change when correcting the random control.']
(o/'REPORT.md').write_text('\n'.join(lines)+'\n')
paths=['analysis/analysis.json','analysis/direction.npz','validation/development.json','validation/selection.json','validation/val.json','native_check/native_invariance.json','head_math/head_math.json','bridge_check/verification.json','bridge_check/analytic_verification.json','causal_ffn/causal.json','train/audit.json','val/audit.json','edits/index.json','feature_audit.json']
if (r/'repro_audit.json').exists():
 assert json.loads((r/'repro_audit.json').read_text())['passed'];paths.append('repro_audit.json')
script_dir=Path(__file__).resolve().parent
train_audit=json.loads((r/'train/audit.json').read_text());checkpoint=Path(train_audit['checkpoint']);assert sha256(checkpoint)==analysis['checkpoint_sha256']
legacy_code=Path(train_audit['label_path']).parent.parent;project=legacy_code.parents[1]
baseline=project/'stage_0_gmr/checkpoints/flashvtg/gmr/best.ckpt';assert sha256(baseline)==train_audit['baseline_sha256']
source_files=list(legacy_code.glob('*.py'))+list((project/'dependencies/stage1_reference/official_eval').rglob('*.py'))+list((project/'dependencies/stage1_reference/baseline_source/models/flash_vtg_gmr').rglob('*.py'))+list((project/'dependencies/stage1_reference/baseline_source/training/flash_vtg_gmr').rglob('*.py'))
write_json(o/'audit.json',{'backbone':'FlashVTG','setting':'historical E1, epoch 4','checkpoint_sha256':analysis['checkpoint_sha256'],'baseline_sha256':train_audit['baseline_sha256'],'train_val_video_disjoint':True,'discovery_development_video_disjoint':True,'test_used':False,'gradient_based_edit':False,'candidate_export':'Frozen original E1 post-NMS candidate predictions; edited existence score only','native_tensor_exact':check['passed'],'head_math_passed':headcheck['passed'],'bridge_passed':bridgecheck['passed'],'random_control_norm_matching':norm_matching,'candidate_selection':selection,'causal_ffn_passed':causal['passed'],'python':platform.python_version(),'inputs_sha256':{p:sha256(r/p) for p in paths},'code_sha256_at_report_time':{p.name:sha256(p) for p in script_dir.glob('*.py')},'legacy_source_sha256_at_report_time':{str(p):sha256(p) for p in source_files},'code_provenance_note':'Some scripts gained fail-closed FFN support after the head experiment started; these are hashes at reporting, not execution-time snapshots.','outputs_sha256':{p.name:sha256(p) for p in o.iterdir() if p.is_file()}})
