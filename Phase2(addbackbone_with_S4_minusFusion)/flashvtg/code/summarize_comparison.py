"""Recompute a predeclared 0:0.01:1 Test diagnostic grid from frozen predictions.
No model/threshold selection. All G scores checked against the official scorer.
"""
import copy,csv,gzip,json
from pathlib import Path
import numpy as np
from environment import ROOT,read_rows,read_json,write_json,digest
from evaluation import Evaluation
from eval.metrics import compute_G_mIoU
ORDER=['E0','S4Fusion']
SHAS={'E0':'15f3054881d562801182efbc1b96b1f26de4c58d72466a8337022027b6485484','S4Fusion':'2e2ff2a728012bad5965e51b13b4ee4452b9ade09df25a571e1ba34a72dd4301'}
def rows(p):
 with gzip.open(p,'rt') as f:return [json.loads(l) for l in f if l.strip()]
def csvout(name,data):
 with (ROOT/'reports'/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
def main():
 gt=read_rows(ROOT/'data/test.jsonl');assert len(gt)==1036
 assert digest(ROOT/'data/test.jsonl')=='d8c47dee986b3ea87ba3e008ac2f5f34b0ab000337631e324a6ecffd1dea910c'
 reports={};grid=[];subsets=[];source_sha={}
 for name in ORDER:
  folder=ROOT/'results'/name;complete=read_json(folder/'TEST_COMPLETE.json');assert complete['checkpoint_sha256']==SHAS[name]
  source_sha[name]={n:digest(folder/n) for n in ['test_predictions.jsonl.gz','test_predictions_pre_nms.jsonl.gz','TEST_COMPLETE.json']}
  ev=Evaluation(rows(folder/'test_predictions_pre_nms.jsonl.gz'),rows(folder/'test_predictions.jsonl.gz'),gt)
  measured=ev.metrics(.5);cached=read_json(folder/'test_metrics.json')['fixed_05']
  assert all(measured[k]==cached[k] for k in measured if k in cached),(name,[(k,measured[k],cached[k]) for k in measured if k in cached and measured[k]!=cached[k]])
  assert ev.verify();summary=ev.metrics(.5)
  masks={'single':ev.count==1,'multi':ev.count>=2,'worldcup2022':np.array([r['dataset_source']=='worldcup2022' for r in gt]),'sportsmoments':np.array([r['dataset_source']=='sportsmoments' for r in gt])}
  subs={k:ev.metrics(.5,mask=v) for k,v in masks.items()}
  for k,m in subs.items():subsets.append(dict(model=name,subset=k,**m))
  summary.update(Rej_F1=100*2*summary['TN']/(2*summary['TN']+summary['FN']+summary['FP']),Multi_G3=subs['multi']['G-mIoU@3'],Multi_positive_recall=subs['multi']['positive_recall'])
  reports[name]=dict(fixed_05=summary,subsets_tau05=subs,checkpoint_sha256=SHAS[name])
  for threshold in np.round(np.arange(0,1.001,.01),2):
   metric=ev.metrics(float(threshold));metric.update(Rej_F1=100*2*metric['TN']/(2*metric['TN']+metric['FN']+metric['FP']),Multi_G3=ev.metrics(float(threshold),mask=masks['multi'])['G-mIoU@3'])
   gated=copy.deepcopy(ev.post)
   for row,score in zip(gated,ev.score):
    if score<threshold:row['pred_relevant_windows']=[]
   official=compute_G_mIoU(gated,gt,(1,3,5));assert all(metric[k]==official[k] for k in official)
   grid.append(dict(model=name,**metric))
  print(name,'verified 101 thresholds',flush=True)
 csvout('threshold_grid_test.csv',grid)
 csvout('summary_test_tau05.csv',[dict(model=n,**reports[n]['fixed_05']) for n in ORDER])
 csvout('fixed_thresholds_test.csv',[r for r in grid if r['threshold'] in [.4,.5,.6]])
 csvout('subsets_test_tau05.csv',subsets)
 keys=['G-mIoU@1','G-mIoU@3','G-mIoU@5','mAP','raw_mAP','N_acc','positive_recall','Multi_G3','Multi_positive_recall','AUROC']
 deltas=[]
 for before,after in [('E0','S4Fusion')]:
  deltas.append(dict(comparison=before+' -> '+after,before=before,after=after,**{k:reports[after]['fixed_05'][k]-reports[before]['fixed_05'][k] for k in keys}))
 csvout('paired_deltas_test.csv',deltas)
 write_json(ROOT/'reports/figure_data.json',dict(reports=reports,deltas=deltas,model_order=ORDER))
 write_json(ROOT/'reports/COMPARISON_VERIFICATION.json',dict(passed=True,rows=1036,positive=544,negative=492,models=ORDER,thresholds=101,threshold_range=[0,1],threshold_step=.01,official_G_verified_all_202_model_threshold_pairs=True,mAP_verified_against_official=True,checkpoint_selection='Val only',Test_optimal_threshold_selected=False,source_sha256=source_sha,method_names={'E0':'FlashVTG-GMR baseline','S4Fusion':'S4-Fusion (S4_noFusion)'},causal_component_interaction_claimed=False))
if __name__=='__main__':main()
