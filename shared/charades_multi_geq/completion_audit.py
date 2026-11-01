"""Evidence audit. Missing or unverified requirements always keep completion false."""
import argparse,csv,math,subprocess,time
from pathlib import Path
import numpy as np
from paths import *
from metrics import measure

def near(a,b):
 assert math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b))<1e-11,(a,b)

def audit(checkpoints=False,recompute=False):
 checks={};pending=[];errors=[];trained={};tested={};train_sha=sha(OUT/'data/train.jsonl')
 def run(key,fn):
  try:checks[key]=dict(passed=True,evidence=fn())
  except Exception as e:checks[key]=dict(passed=False,error=f'{type(e).__name__}: {e}');errors.append(key)
 def data_check():
  expected={'train':(27788,11199),'val':(2047,1205),'test_id':(7440,3720),'test_ood':(7440,3720)};evidence={};saved=json.loads((OUT/'data/audit.json').read_text())
  assert saved['passed'] and len(saved['excluded_negative_rows'])==21
  for split,(n,pos) in expected.items():
   p=OUT/'data'/f'{split}.jsonl';values=rows(p);assert len(values)==n and sum(bool(r['relevant_windows']) for r in values)==pos
   assert len({r['evaluation_id'] for r in values})==n and all(r['qid']==r['evaluation_id'] and r['source_line_number']==r['source_row_index']+1 for r in values)
   assert all(len(r['relevant_windows'])<=1 and 'original_qid' in r for r in values)
   assert sha(p)==saved[split]['sha256'];assert sha(SPLITS/f'{split}.jsonl')==saved[split]['source_sha256']
   evidence[split]=dict(rows=n,positive=pos,sha256=sha(p))
  return evidence
 run('data_views_and_original_integrity',data_check)
 def export_check():
  from export_protocol_audit import audit as verify_export
  report=verify_export();assert report['passed'] and not report['extra_ranking_modules'];return dict(report_sha256=sha(OUT/'reports/export_protocol_audit.json'),scope=report['scope'])
 run('native_grid_and_fixed_temporal_export_policy',export_check)
 def initialization_check():
  from initialization_audit import audit as check_initialization
  report=check_initialization();assert report['all_native_inits_ready']
  if not report['all_four_geq_inits_ready']:pending.append('canonical_GEQ_initialization_not_yet_created_for_all_backbones')
  return dict(seed=report['seed'],all_native_inits_ready=report['all_native_inits_ready'],all_four_geq_inits_ready=report['all_four_geq_inits_ready'],report_sha256=sha(OUT/'reports/initialization_audit.json'))
 run('canonical_initialization_seed_and_tensor_digests',initialization_check)
 for b in BACKBONES:
  for m in METHODS:
   d=OUT/'runs'/b/m;key=f'{b}/{m}';marker=d/'TRAINED.json'
   if not marker.exists():pending.append(key+'/training');continue
   def training_check(d=d,b=b,m=m):
    mark=json.loads((d/'TRAINED.json').read_text());cfg=json.loads((d/'config.json').read_text());history=rows(d/'history.jsonl');val=json.loads((d/'best_val_metrics.json').read_text())
    assert cfg['seed']==mark['seed']==2023 and cfg['train_sha256']==train_sha and cfg['amp'] is False
    assert cfg['microbatch']==dict(moment_detr=8,eatr=32,qd_detr=32,flashvtg=8)[b] and cfg['accumulation']==1
    assert sha(d/'best.ckpt')==mark['checkpoint_sha256']
    assert [r['epoch'] for r in history]==list(range(1,len(history)+1))
    stage_b='_geq_' in m;best=-math.inf;best_epoch=0;stale=0;best_above=False
    baseline=json.loads((OUT/'runs'/b/'gmr/best_val_metrics.json').read_text())['selection_values'] if stage_b else None
    for h in history:
     assert stale<(6 if stage_b else 20),'Training continued after stopping rule'
     if stage_b:near(h['selection'],float(np.mean(h['val_values'])))
     else:near(h['selection'],h['mAP'])
     above=bool(stage_b) and all(x>y for x,y in zip(h['val_values'],baseline))
     improvement=h['selection']>best;tie=h['selection']==best and above and not best_above
     stale=0 if improvement else stale+1
     if improvement or tie:best=h['selection'];best_epoch=h['epoch'];best_above=above
     assert h['best_epoch']==best_epoch and h['stale']==stale;near(h['best'],best)
    assert len(history)<=(20 if stage_b else 200)
    if mark.get('stopped_by_user'):
     assert (b,m)==('moment_detr','multi'),'User stop authorization is scoped to Moment-DETR Multi'
     request=json.loads((d/'USER_STOP_REQUEST.json').read_text())
     assert sha(d/'USER_STOP_REQUEST.json')==mark['stop_request_sha256']
     assert request['authorized_by']=='user' and request['instruction'] and (request['backbone'],request['method'])==(b,m)
     assert request['checkpoint_sha256']==mark['checkpoint_sha256'] and request['best_epoch']==best_epoch
     near(request['score'],best);assert request['selection']==mark['selection']=='Val positive mAP'
     assert mark['last_completed_epoch']==history[-1]['epoch'] and request['time']<=mark['stop_time']
    else:
     assert stale>=(6 if stage_b else 20) or len(history)==(20 if stage_b else 200)
    assert best_epoch==mark['best_epoch'];near(best,mark['score']);near(best,val['selection_mean'] if stage_b else val['mAP'])
    predictions=rows(d/'best_val_predictions.jsonl');gt=rows(OUT/'data/val.jsonl');assert [p['qid'] for p in predictions]==[g['evaluation_id'] for g in gt]
    if stage_b:
     parent=OUT/'runs'/b/('multi' if m.startswith('multi') else 'gmr');assert cfg['parent_sha256']==sha(parent/'best.ckpt')
    if recompute:
     measured=measure(predictions,gt);near(measured['mAP'],val['mAP']);near(measured['selection_mean'],val['selection_mean'])
    if checkpoints:
     import torch
     torch.set_num_threads(2)
     state=torch.load(d/'best.ckpt',map_location='cpu',weights_only=False)
     assert state['seed']==2023 and state['train_sha256']==train_sha and state['best_epoch']==best_epoch
     assert all(torch.isfinite(v).all() for v in state['model'].values() if v.is_floating_point())
     assert all(k in state for k in ('optimizer','rng','sampler_epoch_start'))
     assert all(g['weight_decay']==1e-4 for g in state['optimizer']['param_groups'])
     lrs=[g['lr'] for g in state['optimizer']['param_groups']]
     assert lrs==([3e-5] if not stage_b or m.endswith('frozen') else [1e-5,3e-5]),lrs
     if m.endswith('frozen'):
      parent_state=torch.load(parent/'best.ckpt',map_location='cpu',weights_only=False)['model'];base={k[5:]:v for k,v in state['model'].items() if k.startswith('base.')};assert base.keys()==parent_state.keys();assert all(torch.equal(v,parent_state[k]) for k,v in base.items())
    trained[b+'/'+m]=True
    return dict(best_epoch=best_epoch,last_epoch=len(history),score=best,best_checkpoint_sha256=mark['checkpoint_sha256'])
   run(key+'/training_selection',training_check)
 for b in BACKBONES:
  for m in METHODS:
   for split in ('test_id','test_ood'):
    d=OUT/'runs'/b/m;dest=d/split;key=f'{b}/{m}/{split}'
    if not (dest/'COMPLETE.json').exists():pending.append(key+'/evaluation');continue
    def test_check(d=d,dest=dest,b=b,m=m,split=split):
     assert b+'/'+m in trained
     done=json.loads((dest/'COMPLETE.json').read_text());mark=json.loads((d/'TRAINED.json').read_text())
     assert done['checkpoint_sha256']==mark['checkpoint_sha256']==sha(d/'best.ckpt')
     assert done['prediction_sha256']==sha(dest/'predictions.jsonl') and done['metrics_sha256']==sha(dest/'metrics.json')
     pred=rows(dest/'predictions.jsonl');gt=rows(OUT/'data'/f'{split}.jsonl');v=json.loads((dest/'metrics.json').read_text())
     assert done['rows']==len(pred)==len(gt)==7440
     assert [p['qid'] for p in pred]==[g['evaluation_id'] for g in gt]
     assert len({p['qid'] for p in pred})==7440
     assert all(p['original_qid']==g['original_qid'] and p['vid']==g['vid'] and p['source_row_index']==g['source_row_index'] for p,g in zip(pred,gt))
     assert v['mR_plus']==v['multi_gt_coverage']=='N/A' and v['all_reject_gmiou']==.5
     assert len(v['grid'])==101
     for i,r in enumerate(v['grid']):
      near(r['threshold'],i/100)
      assert all(math.isfinite(r[k]) and 0<=r[k]<=1 for k in ('gmiou_1','gmiou_3','rejection_f1','positive_retention','negative_rejection'))
     near(v['grid'][-1]['gmiou_1'],.5);near(v['grid'][-1]['positive_retention'],0)
     if recompute:
      measured=measure(pred,gt)
      for k in ('mAP','mR1','mR5','AUROC','selection_mean'):near(measured[k],v[k])
      for a,c in zip(measured['grid'],v['grid']):
       for k in a:near(a[k],c[k])
     if m.endswith('frozen'):
      parent=OUT/'runs'/b/('multi' if m.startswith('multi') else 'gmr')/split
      parent_pred=rows(parent/'predictions.jsonl');pv=json.loads((parent/'metrics.json').read_text())
      assert [p['pred_relevant_windows'] for p in pred]==[p['pred_relevant_windows'] for p in parent_pred]
      assert v['mAP']==pv['mAP'] and v['frozen_candidates_identical']
     tested[b+'/'+m+'/'+split]=True
     return dict(rows=len(pred),checkpoint_sha256=done['checkpoint_sha256'],metrics_sha256=done['metrics_sha256'])
    run(key+'/identity_metrics_and_frozen_invariance',test_check)
 for b in BACKBONES:
  for m in METHODS:
   p=OUT/'runs'/b/m/'PROFILE.json';key=f'{b}/{m}/inference_profile'
   if not p.exists():pending.append(key);continue
   def profile_check(p=p,b=b,m=m):
    v=json.loads(p.read_text());assert v['checkpoint_sha256']==json.loads((p.parent/'TRAINED.json').read_text())['checkpoint_sha256'];assert v['source_split']=='val' and v['measured_iterations']==100 and v['warmup_iterations']==25
    assert v['batch_size']==(1 if b=='flashvtg' else 8 if b=='moment_detr' else 32)
    assert math.isfinite(v['total']['mean_ms_per_query']) and v['total']['mean_ms_per_query']>0
    if '_geq_' in m:assert math.isfinite(v['incremental_geq_ms_per_query']) and v['geq_parameters']>0
    return dict(sha256=sha(p),batch_size=v['batch_size'])
   run(key,profile_check)
 for b in BACKBONES:
  for split in ('test_id','test_ood'):
   dest=OUT/'figures'/b/split
   for comparison in ('frozen','full','multi_comparison'):
    key=f'figures/{b}/{split}/{comparison}'
    required=[dest/(comparison+'.'+ext) for ext in ('png','pdf','svg','csv','snapshot.json')]
    if not all(p.exists() for p in required):pending.append(key);continue
    def figure_check(dest=dest,comparison=comparison,b=b,split=split):
     snap=json.loads((dest/(comparison+'.snapshot.json')).read_text());assert snap['backbone']==b and snap['split']==split
     expected={'frozen':['gmr','gmr_geq_frozen','multi','multi_geq_frozen'],'full':['gmr','gmr_geq_full','multi','multi_geq_full'],'multi_comparison':['multi','multi_geq_frozen','multi_geq_full']}[comparison];assert snap['methods']==expected
     for m in expected:
      assert b+'/'+m+'/'+split in tested
      actual=json.loads((OUT/'runs'/b/m/split/'COMPLETE.json').read_text());assert snap['provenance'][m]==actual
     with (dest/(comparison+'.csv')).open() as f:csv_rows=list(csv.DictReader(f))
     assert len(csv_rows)==101*len(expected)
     for row in csv_rows:
      grid=snap['series'][row['method']]['grid'];index=round(float(row['threshold'])*100)
      for k,v in grid[index].items():near(row[k],v)
     return dict(snapshot_sha256=sha(dest/(comparison+'.snapshot.json')))
    run(key,figure_check)
   for artifact in ('summary.csv','heatmap.png','heatmap.pdf','heatmap.svg','val_selection.png','val_selection.pdf','val_selection.svg'):
    if not (dest/artifact).exists():pending.append(f'figures/{b}/{split}/{artifact}')
  readme=OUT/'figures'/b/'README.md'
  if not readme.exists():pending.append(f'figures/{b}/README.md')
  else:
   import re
   for target in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',readme.read_text()):
    if not (readme.parent/target).is_file():errors.append(f'Broken README image: {readme}: {target}')
 review_path=OUT/'reports/FIGURE_REVIEW.json'
 if not review_path.exists():pending.append('manual_visual_figure_review_against_current_image_hashes')
 else:
  def review_check():
   review=json.loads(review_path.read_text());assert review['approved'] is True
   expected=[p for p in (OUT/'figures').rglob('*') if p.suffix in ('.png','.pdf','.svg')]
   expected.extend(p for p in (OUT/'reports').glob('heatmap_*') if p.suffix in ('.png','.pdf','.svg'))
   for image in expected:assert review['files'].get(str(image.relative_to(OUT)))==sha(image),'Visual review stale or missing: '+str(image)
   return dict(approved_artifacts=len(expected),review_sha256=sha(review_path))
  run('visual_review_current_hashes',review_check)
 if not (OUT/'logs/PUBLISHED.json').exists():pending.append('verified_final_GitHub_publication')
 if not checkpoints:pending.append('checkpoint_tensor_optimizer_and_parent_checks_not_run')
 if not recompute:pending.append('independent_metrics_recalculation_not_run')
 return dict(time=time.time(),complete=not pending and not errors and len(trained)==24 and len(tested)==48,training_verified=len(trained),tests_verified=len(tested),checks=checks,pending=pending,errors=errors)

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoints',action='store_true');p.add_argument('--recompute',action='store_true');p.add_argument('--require-complete',action='store_true');a=p.parse_args()
 report=audit(a.checkpoints,a.recompute);dump(OUT/'reports/completion_audit.json',report);print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
 if report['errors'] or a.require_complete and not report['complete']:raise SystemExit(1)
if __name__=='__main__':main()
