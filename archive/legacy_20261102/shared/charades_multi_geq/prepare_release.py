"""Prepare only Charades-owned release paths; exclude weights/features/credentials."""
import argparse,gzip,hashlib,io,json,re,shutil,time,csv
from pathlib import Path
from paths import *
WORKTREE=Path('/home/guoxiangyu/charades-multi-geq-worktree')
ALLOWED_SOURCE={'.py','.json','.md','.sh','.txt'}
FORBIDDEN={'.pt','.pth','.ckpt','.npz','.npy','.mp4','.avi','.sqlite3','.lock','.pyc'}
LABELS={'gmr':'GMR','multi':'GMR + Multi','gmr_geq_frozen':'GMR + GEQ Frozen','gmr_geq_full':'GMR + GEQ Full','multi_geq_frozen':'GMR + Multi + GEQ Frozen','multi_geq_full':'GMR + Multi + GEQ Full'}

def copy_file(source,dest):
 dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name(dest.name+'.copying');shutil.copyfile(source,tmp);tmp.replace(dest)

def prepare(worktree=WORKTREE):
 manifest_path=OUT/'source_manifest.json';source_manifest=json.loads(manifest_path.read_text());source_manifest['files']={str(p.relative_to(KIT)):sha(p) for p in sorted(KIT.rglob('*')) if p.is_file() and p.suffix in ALLOWED_SOURCE and '__pycache__' not in p.parts};source_manifest['initialization_snapshots']={str(p.relative_to(OUT)):sha(p) for b in BACKBONES for name in ('native_init.pt','geq_init.pt') for p in [OUT/'runs'/b/name] if p.exists()};dump(manifest_path,source_manifest)
 source=worktree/'shared/charades_multi_geq';source.mkdir(parents=True,exist_ok=True)
 for old in source.rglob('*'):
  if old.is_file() and '__pycache__' not in old.parts and not (KIT/old.relative_to(source)).exists() and old.name not in ('run.py','dispatch.py'):
   archive=OUT/'reports/unreferenced_release_sources'/old.relative_to(source);archive.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,archive);old.unlink()
 for p in KIT.rglob('*'):
  if p.is_file() and p.suffix in ALLOWED_SOURCE and '__pycache__' not in p.parts:copy_file(p,source/p.relative_to(KIT))
 # Keep exactly one implementation behind historic entry point names.
 (source/'run.py').write_text('from runner import main\nif __name__ == "__main__": main()\n')
 (source/'dispatch.py').write_text('from queue_manager import main\nif __name__ == "__main__": main()\n')
 # This is a reusable monitor, not a public pointer to the author's local thread.
 monitor=source/'monitor_wakeup.py'
 if monitor.exists():monitor.write_text(monitor.read_text().replace("THREAD='01a19315-f496-7fc1-a79e-751074928b45'","THREAD=os.environ.get('CHARADES_CODEX_THREAD','')"))
 result=worktree/'results/charades_multi_geq';result.mkdir(parents=True,exist_ok=True)
 for name in ('protocol.json','source_manifest.json'):copy_file(OUT/name,result/name)
 if (OUT/'reports/all_metrics.csv').exists():copy_file(OUT/'reports/all_metrics.csv',result/'all_metrics.csv')
 summary=['# Charades summary heatmaps','','Colors compare actual values within a backbone row only. O denotes the fixed-candidate perfect-presence reference; it does not select a checkpoint or threshold.','']
 for p in sorted((OUT/'reports').glob('heatmap_*')):
  if p.is_file() and p.suffix in ('.png','.pdf','.svg','.csv','.json'):
   copy_file(p,result/'summary'/p.name)
   if p.suffix=='.png':summary += [f'## {p.stem}','','!['+p.stem+']('+p.name+')','',f'[PDF]({p.stem}.pdf) · [SVG]({p.stem}.svg) · [CSV]({p.stem}.csv) · [Offline snapshot]({p.stem}.snapshot.json)','']
 (result/'summary').mkdir(exist_ok=True);(result/'summary/README.md').write_text('\n'.join(summary)+'\n')
 for p in (OUT/'figures').rglob('*'):
  if p.is_file() and p.suffix in ('.png','.pdf','.svg','.json','.csv','.md'):copy_file(p,result/p.relative_to(OUT/'figures'))
 audit_names=('DEBUG24_READY.json','export_protocol_audit.json','initialization_audit.json','flash_zero_point_guard.json','independent_data_audit.json','independent_metric_contract.json','completion_audit.json','flash_zero_reg_fix_20261101.json','moment_frozen_parent_integrity_20261101.json','moment_frozen_cache_fix_20261101.json','validation_cadence_regression_20261101.json','validation_cadence_restart_20261101.json','auxiliary_matching_batch_transfer_20261101.json','eatr_matching_gpu_verification_20261101.json','eatr_matching_activation_20261101.json','eatr_original_continuation_regression_20261101.json','eatr_restore_all_queries_20261101.json','FIGURE_REVIEW.json')
 for name in audit_names:
  p=OUT/'reports'/name
  if p.exists():copy_file(p,result/'audit'/name)
 if (OUT/'reports/final_artifact_audit.json').exists():copy_file(OUT/'reports/final_artifact_audit.json',result/'audit/final_artifact_audit.json')
 for pattern in ('native_factory_equivalence_*.json','resume_audit_*.json'):
  for p in (OUT/'reports').glob(pattern):copy_file(p,result/'audit'/p.name)
 for p in (OUT/'reports/debug24').glob('*/*/preflight.json'):copy_file(p,result/'audit/debug24'/p.relative_to(OUT/'reports/debug24'))
 for name in ('audit.json','exclusions.json'):copy_file(OUT/'data'/name,result/'audit/data'/name)
 # Derived label views are small reproducibility inputs, never feature arrays.
 for split in ('train','val','test_id','test_ood'):copy_file(OUT/'data'/f'{split}.jsonl',result/'data'/f'{split}.jsonl')
 trained=0;tested=0;table=[]
 for b in BACKBONES:
  for m in METHODS:
   d=OUT/'runs'/b/m;complete=(d/'TRAINED.json').exists();trained+=complete
   for name in ('config.json','TRAINED.json','preflight.json','history.jsonl','best_val_metrics.json','PROFILE.json','validation_schedule.json','USER_STOP_REQUEST.json','USER_CONTINUE_REQUEST.json','PRIOR_TRAINED.json','RETRAIN_PROTOCOL.json','runtime_optimizations.json'):
    if (d/name).exists():copy_file(d/name,result/'run_metadata'/b/m/name)
   val=json.loads((d/'best_val_metrics.json').read_text()) if (d/'best_val_metrics.json').exists() else None
   label=LABELS[m].replace('Multi','Multi-TQAux') if b=='flashvtg' else LABELS[m]
   table.append(f"| {b} | {label} | {'Val selection finished' if complete else 'In progress' if val else 'Pending'} | {val['mAP']:.2f} | {val['selection_mean']:.4f} |" if val else f'| {b} | {label} | Pending | — | — |')
   for split in ('test_id','test_ood'):
    sd=d/split
    if not (sd/'COMPLETE.json').exists():continue
    marker=json.loads((sd/'COMPLETE.json').read_text());assert marker['prediction_sha256']==sha(sd/'predictions.jsonl') and marker['metrics_sha256']==sha(sd/'metrics.json')
    tested+=1
    for name in ('metrics.json','COMPLETE.json'):copy_file(sd/name,result/'run_metadata'/b/m/split/name)
    dest=result/'run_metadata'/b/m/split/'predictions.jsonl.gz';dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('wb') as f:
     with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as out:out.write((sd/'predictions.jsonl').read_bytes())
 finished=trained==24 and tested==48
 state='Training and formal evaluation are complete; final acceptance is documented in the audit.' if finished else 'This release remains incomplete until the full audit and remote verification pass.'
 readme=['# Charades GMR + Multi + GEQ (seed 2023)','',f'**Actual completion: {trained}/24 training tasks; {tested}/48 formal test evaluations.** '+state,'', 'Test-ID is the primary result. Test-OOD is supplementary and keeps all 7,440 rows, including 2,170 duplicate extra records. Tests never select checkpoints or thresholds. All local positive examples have one GT: mR+ and multi-GT coverage are N/A. Flash Multi uses the **Multi-TQAux structural adaptation**, not the same native query decoder as the other backbones.','', '[Reproduction instructions](../../shared/charades_multi_geq/README.md) · [Protocol](protocol.json) · [Completion audit](audit/completion_audit.json) · [All metrics CSV](all_metrics.csv) · [Summary heatmaps](summary/README.md) · [Existence-score distributions](score_distributions/README.md)','', 'Stage A selects ungated positive-Val mAP. Stage B selects the Val G-mIoU@1 mean at presence thresholds0.4/0.5/0.6. Stage A validation cadence was prospectively changed to every5 epochs by the user; historical validations are retained. Moment Multi was ended early by user instruction; original EaTR Multi received an explicitly authorized continuation. These exceptions and their checkpoint/history hashes are recorded in run_metadata.','']
 if (result/'all_metrics.csv').exists():
  records=list(csv.DictReader((result/'all_metrics.csv').open()))
  for split in ('test_id','test_ood'):
   readme += [f'## {split} — fixed Val-selected checkpoints','','G1/G3 are the three-threshold means on a0–1 scale; mAP and AUROC are percentages. ΔG1 uses the same-backbone GMR at the same fixed thresholds. Every row is retained, including decreases. Full metrics, timing and memory are in the CSV.','', '| Backbone | Method | mAP % | G1 mean | G3 mean | AUROC % | ΔG1 vs GMR |','|---|---|---:|---:|---:|---:|---:|']
   baseline={r['backbone']:float(r['gmiou_1_mean']) for r in records if r['split']==split and r['method']=='gmr'}
   for r in records:
    if r['split']!=split:continue
    label=LABELS[r['method']].replace('Multi','Multi-TQAux') if r['backbone']=='flashvtg' else LABELS[r['method']]
    readme += [f"| {r['backbone']} | {label} | {float(r['mAP_percent']):.2f} | {float(r['gmiou_1_mean']):.4f} | {float(r['gmiou_3_mean']):.4f} | {float(r['AUROC_percent']):.2f} | {float(r['gmiou_1_mean'])-baseline[r['backbone']]:+.4f} |"]
   readme += ['']
 readme += ['## Val-selected checkpoints','','G-mIoU@1 is the mean at presence thresholds0.4/0.5/0.6, on a0–1 scale. mAP is ungated positive-query localization mAP in percent.','', '| Backbone | Method | Status | Val mAP % | Val G-mIoU@1 mean |','|---|---|---|---:|---:|']+table
 extra=OUT/'runs/eatr/multi_positive_only'
 if (extra/'DUAL_VARIANT_AUTHORIZATION.json').exists():
  for name in ('config.json','TRAINED.json','preflight.json','history.jsonl','best_val_metrics.json','validation_schedule.json','RETRAIN_PROTOCOL.json','runtime_optimizations.json','DUAL_VARIANT_AUTHORIZATION.json'):
   if (extra/name).exists():copy_file(extra/name,result/'run_metadata/eatr/multi_positive_only'/name)
  val=json.loads((extra/'best_val_metrics.json').read_text()) if (extra/'best_val_metrics.json').exists() else None
  readme += ['', '## Additional EaTR positive-only ablation','', 'This independently resumed Stage A run is additional to the main24 training tasks and48 formal Test evaluations. It has separate checkpoints and Val history. Its training must finish before final acceptance.','']
  if val:readme += [f"Val-selected mAP: {val['mAP']:.2f}%; training {'finished' if (extra/'TRAINED.json').exists() else 'in progress'}.",'']
 for b in BACKBONES:
  readme += ['',f'## {b}','',f'[All figures, CSV files and offline snapshots]({b}/README.md)','']
  if (result/b/'test_id/frozen.png').exists():readme += [f'![{b} Test-ID Frozen]({b}/test_id/frozen.png)','']
  elif (result/b/'val/selection.png').exists():readme += [f'![{b} Val selection]({b}/val/selection.png)','']
 (result/'README.md').write_text('\n'.join(readme)+'\n')
 marker_begin='<!-- CHARADES_MULTI_GEQ_BEGIN -->';marker_end='<!-- CHARADES_MULTI_GEQ_END -->';root_readme=worktree/'README.md';text=root_readme.read_text();text=re.sub(re.escape(marker_begin)+r'.*?'+re.escape(marker_end)+r'\n*','',text,flags=re.S)
 section=[marker_begin,'# Charades and SoccerGMR experiments','', '## Charades: target-domain seed2023 training','',f'Charades progress: **{trained}/24 training tasks, {tested}/48 Test evaluations**. [Full protocol, status and results](results/charades_multi_geq/README.md). GMR + Multi is the explicit name of the Multi baseline; Flash uses Multi-TQAux.','']
 for b in BACKBONES:
  image=result/b/'test_id/frozen.png'
  if not image.exists():image=result/b/'val/selection.png'
  if image.exists():section += [f'![Charades {b}]({image.relative_to(worktree).as_posix()})','']
 section += ['## SoccerGMR: previously published experiments','', 'The existing sections below concern SoccerGMR and are separate from the Charades experiment above.',marker_end,'']
 root_readme.write_text('\n'.join(section)+'\n'+text)
 owned=[source,result]+[worktree/'backbones'/b/'charades_multi_geq.sh' for b in BACKBONES]+[worktree/'backbones/flashvtg/README.md',root_readme]
 files=[]
 for p in owned:
  files.extend(x for x in p.rglob('*') if x.is_file() and '__pycache__' not in x.parts) if p.is_dir() else files.append(p) if p.is_file() else None
 secret=re.compile(rb'(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)')
 manifest={}
 for p in files:
  assert p.suffix not in FORBIDDEN and 'token2git' not in p.name.lower(),str(p)
  payload=p.read_bytes();inspect=gzip.decompress(payload) if p.name.endswith('.jsonl.gz') else payload
  assert not secret.search(inspect),f'Credential pattern in {p.relative_to(worktree)}'
  manifest[p.relative_to(worktree).as_posix()]=hashlib.sha256(payload).hexdigest()
 dump(OUT/'reports/release_manifest.json',dict(time=time.time(),trained=trained,tested=tested,files=manifest))
 print(json.dumps(dict(files=len(manifest),trained=trained,tested=tested,root=str(result))))
 return manifest

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worktree',default=str(WORKTREE));a=p.parse_args();prepare(Path(a.worktree))
