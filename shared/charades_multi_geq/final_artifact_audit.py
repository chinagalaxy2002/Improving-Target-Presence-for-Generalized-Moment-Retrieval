"""Final checks for the requested Oracle, distributions and offline artifacts."""
import csv,json,subprocess,sys,tempfile,time
import numpy as np
from paths import *
from oracle_presence import compute_oracle_presence
from plot_existence_score_distributions import build_snapshot


def main():
 evidence={};oracle_count=0
 for split in ('test_id','test_ood'):
  gt=rows(OUT/'data'/f'{split}.jsonl')
  for b in BACKBONES:
   for m in METHODS:
    d=OUT/'runs'/b/m/split;done=json.loads((d/'COMPLETE.json').read_text());assert sha(d/'predictions.jsonl')==done['prediction_sha256']
    actual=compute_oracle_presence(rows(d/'predictions.jsonl'),gt)
    cached=json.loads((OUT/'figures'/b/split/f'{m}_oracle_presence.json').read_text())
    assert cached['source']['prediction_sha256']==done['prediction_sha256']
    metrics=json.loads((d/'metrics.json').read_text())
    for k in (1,3):
     assert abs(actual[f'gmiou_{k}']-cached[f'gmiou_{k}'])<1e-12
     assert max(r[f'gmiou_{k}'] for r in metrics['grid'])<=actual[f'gmiou_{k}']+1e-12
    oracle_count+=1
 records=list(csv.DictReader((OUT/'reports/all_metrics.csv').open()));assert len(records)==48
 for r in records:
  m=json.loads((OUT/'runs'/r['backbone']/r['method']/r['split']/'metrics.json').read_text())
  for col,key in [('mAP_percent','mAP'),('mR1_percent','mR1'),('mR5_percent','mR5'),('AUROC_percent','AUROC'),('gmiou_1_mean','selection_mean')]:assert abs(float(r[col])-m[key])<1e-12
  for k in (1,3):assert abs(float(r[f'gmiou_{k}_mean'])-sum(m['grid'][i][f'gmiou_{k}'] for i in [40,50,60])/3)<1e-12
 for split in ('test_id','test_ood'):
  for family in ('gmr','multi'):
   f=OUT/'figures/score_distributions'/f'{split}_{family}_existence_score_distributions.snapshot.json'
   s=json.loads(f.read_text());assert s==build_snapshot(split,family)
   assert len(s['sources'])==12 and all(not p['pending'] for p in s['panels'].values())
   for p in s['panels'].values():
    for m in p['methods'].values():
     for v in m.values():assert v['n']==sum(v['counts'])==3720 and abs(np.dot(v['density'],np.diff(s['bin_edges']))-1)<1e-12
   evidence[str(f.relative_to(OUT))]=sha(f)
 vectors=list((OUT/'figures').rglob('*.svg'))+list((OUT/'reports').glob('heatmap_*.svg'))
 for svg in vectors:
  assert 'oracle' in svg.read_text().lower(),svg
  pdf=svg.with_suffix('.pdf');assert pdf.exists()
  result=subprocess.run(['pdftotext',str(pdf),'-'],capture_output=True,text=True,check=True)
  assert 'oracle' in result.stdout.lower(),pdf
  evidence[str(svg.relative_to(OUT))]=sha(svg);evidence[str(pdf.relative_to(OUT))]=sha(pdf)
 samples=['figures/eatr/test_id/full','figures/flashvtg/test_ood/frozen','figures/eatr/val/selection',
  'figures/eatr/test_ood/heatmap','reports/heatmap_test_id_gmiou_1_mean',
  'figures/qd_detr/test_ood/full_oracle_upper_bound','figures/qd_detr/test_ood/full_relative_to_gmr',
  'figures/score_distributions/test_ood_multi_existence_score_distributions']
 offline={}
 with tempfile.TemporaryDirectory(prefix='charades-final-replot-') as temporary:
  for i,stem in enumerate(samples):
   target=Path(temporary)/str(i)
   subprocess.run([sys.executable,str(KIT/'plot_report.py'),'--snapshot',str(OUT/(stem+'.snapshot.json')),'--output',str(target)],check=True,stdout=subprocess.DEVNULL)
   for suffix in ('png','pdf','svg'):assert sha(target.with_suffix('.'+suffix))==sha(OUT/(stem+'.'+suffix)),(stem,suffix)
   offline[stem]=['png','pdf','svg']
 report=dict(passed=True,time=time.time(),formal_test_oracles_recomputed=oracle_count,metrics_csv_rows_verified=48,
  score_distribution_snapshots_verified=4,vector_pairs_with_oracle=len(vectors),offline_replots_identical=offline,
  files=evidence,test_predictions_or_metrics_modified=False,test_used_for_selection=False)
 dump(OUT/'reports/final_artifact_audit.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('files','offline_replots_identical')},indent=2))


if __name__=='__main__':main()
