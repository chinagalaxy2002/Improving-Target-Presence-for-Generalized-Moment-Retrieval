#!/usr/bin/env python3
"""Full Table2-style summary and five Fig7-style baseline/GMR/S4 comparisons."""
import csv,datetime,hashlib,json,pathlib,shutil,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator,MaxNLocator

HERE=pathlib.Path(__file__).resolve().parent
BASE=HERE.parent
PROJECT=BASE.parent
ARCHIVE=PROJECT/'Phase2(addbackbone_with_S4_minusFusion)'
OUT=HERE/'complete_comparison'
DATA=OUT/'data'
SHARED=ARCHIVE/'shared'
sys.path.insert(0,str(SHARED))
from official_eval.eval_main import evaluate_gmr
from official_eval.metrics import get_existence_score,_compute_set_iou_score,prepare_submission_for_gmiou,compute_G_mIoU,compute_gmr_cls
from build_summary_and_plots import curves_for,read_rows

MODELS=['moment_detr','qd_detr','cg_detr','eatr','flashvtg']
NAMES={'moment_detr':'Moment-DETR','qd_detr':'QD-DETR','cg_detr':'CG-DETR','eatr':'EaTR','flashvtg':'FlashVTG'}
KEYS=['AUROC','Rej-F1@0.4','mAP','mR@1','mR@5','mR+@5','G-mIoU@1','G-mIoU@3']
LABELS={'original':'Baseline','gmr':'Baseline + GMR','s4':'Baseline + GMR + S4_minus_Fusion (epoch 4)'}
STYLES={'original':dict(color='#009E73',linestyle='-.',marker='^'),'gmr':dict(color='#64748B',linestyle='--',marker='o'),'s4':dict(color='#0072B2',linestyle='-',marker='s')}
GRID=np.round(np.linspace(0,1,101),2)

def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def dump(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def load_baselines(model):
 variants=['gmr'] if model=='flashvtg' else ['original','gmr']
 result={}
 for variant in variants:
  p=ARCHIVE/'shared/plot_baselines'/f'{model}_{variant}.json';data=json.loads(p.read_text())
  source=pathlib.Path(data['source_predictions']);assert source.is_file() and sha(source)==data['source_sha256'],str(source)
  assert data['checks']['query_coverage']==1036 and len(data['grid'])==101
  assert data['grid'][40]['G-mIoU@1']==data['tau_0.4']['G-mIoU@1']
  result[variant]=data
 return result

def measure_s4(model,gt):
 directory=ARCHIVE/model;index=json.loads((directory/'ARCHIVE_INDEX.json').read_text())
 checkpoint=directory/index['checkpoint'];snapshot=directory/index['baseline_snapshot'];predpath=directory/index['test_predictions'];metricpath=directory/index['test_metrics']
 assert index['epoch']==4 and index['queries']==1036
 assert sha(checkpoint)==index['checkpoint_sha256'] and sha(snapshot)==index['baseline_sha256']
 if model=='flashvtg':
  baseline_sha=json.loads((BASE/model/'gmr/SOURCE_PROVENANCE.json').read_text())['source_checkpoint']['sha256']
 else:baseline_sha=sha(BASE/model/'gmr'/('best.pt' if model=='cg_detr' else 'best.ckpt'))
 assert baseline_sha==index['baseline_sha256'],'S4 initialization does not match displayed GMR baseline'
 pred=read_rows(predpath);by_id={str(p['qid']):p for p in pred};assert len(pred)==len(by_id)==1036 and set(by_id)=={str(g['qid']) for g in gt}
 pred=[{**by_id[str(g['qid'])],'qid':g['qid']} for g in gt]
 metrics=evaluate_gmr(pred,gt,cls_thresholds=(.4,.5,.6,.8),gmiou_cls_threshold=.4,map_num_workers=0,verbose=False)
 recorded=json.loads(metricpath.read_text());assert metrics['brief']==recorded['brief'],model
 scores=np.array([get_existence_score(p)[0] for p in pred]);positive=np.array([bool(g['relevant_windows']) for g in gt]);assert np.isfinite(scores).all() and np.all((scores>=0)&(scores<=1))
 clean,_=prepare_submission_for_gmiou(pred,cls_threshold=-1,max_pred_windows=10)
 local=np.array([[_compute_set_iou_score([w[:2] for w in p['pred_relevant_windows'][:k]],g['relevant_windows']) for k in (1,3)] for p,g in zip(clean,gt)])
 grid=curves_for(scores,local,positive,GRID);empirical=curves_for(scores,local,positive,np.unique(np.r_[0.,scores,1.]))
 archived=json.loads((directory/'epoch_004/figures/S4_TEST_CURVES.json').read_text());assert grid==archived['grid'],model
 for threshold in (0.,.4,.5,.6,.8,1.):
  row=grid[int(round(threshold*100))];gated,_=prepare_submission_for_gmiou(pred,cls_threshold=threshold,max_pred_windows=10)
  og=compute_G_mIoU(gated,gt,k_list=(1,3));oc=compute_gmr_cls(pred,gt,thresholds=(threshold,))['per_threshold'][str(threshold)]
  assert all(row[k]==og[k] for k in ('G-mIoU@1','G-mIoU@3')) and row['Rej-F1']==oc['Rej-F1']
 data={'backbone':model,'variant':'s4','epoch':4,'name':NAMES[model]+'-GMR + S4_minus_Fusion (epoch 4)','source_predictions':str(predpath),'source_sha256':sha(predpath),'source_metrics':str(metricpath),'source_metrics_sha256':sha(metricpath),'checkpoint':str(checkpoint),'checkpoint_sha256':index['checkpoint_sha256'],'initialization_baseline_sha256':baseline_sha,'tau_0.4':{k:metrics['brief'][k] for k in KEYS},'grid':grid,'empirical_tradeoff':empirical,'checks':{'query_coverage':1036,'positive':544,'negative':492,'recomputed_metrics_match_archive':True,'all_101_thresholds_match_archive':True,'initialization_matches_displayed_gmr':True},'archive_index':index,'flashvtg_reference_no_qcd':model=='flashvtg'}
 dump(DATA/f'{model}_s4_epoch4.json',data);dump(DATA/f'{model}_s4_epoch4_metrics_tau_0.4.json',metrics)
 print('VERIFIED_S4',model,json.dumps(data['tau_0.4']),flush=True)
 return data

def configure_style():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,'axes.titleweight':'bold','axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':11,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

def axis(ax,title,xlabel,ylabel,xmax=1,ymax=None):
 ax.set_title(title,loc='left',pad=10);ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.set_xlim(0,xmax)
 ax.xaxis.set_major_locator(MultipleLocator(.2 if xmax==1 else 20));ax.grid(axis='y',color='#DFE7F1',linewidth=.7);ax.set_axisbelow(True)
 if ymax is not None:ax.set_ylim(0,ymax)

def line(ax,x,y,key,markers=True):
 style=STYLES[key].copy()
 if not markers:style.pop('marker')
 return ax.plot(x,y,**style,linewidth=2 if key=='s4' else 1.8,markersize=3.8,markevery=10 if markers else None,label=LABELS[key])[0]

def render(model,series):
 fig,axes=plt.subplots(2,3,figsize=(15,8.9));fig.subplots_adjust(left=.065,right=.986,bottom=.095,top=.865,hspace=.36,wspace=.26)
 ax1,ax3,delta,tradeoff,f1,fixed=axes.flat
 axis(ax1,'(a) Threshold sensitivity','Presence threshold','G-mIoU@1 (%)',ymax=65)
 axis(ax3,'(b) Top-3 retrieval','Presence threshold','G-mIoU@3 (%)',ymax=65)
 axis(delta,'(c) Relative to GMR baseline','Presence threshold',r'$\Delta$ G-mIoU@1 (pp)')
 axis(tradeoff,'(d) Rejection / positive retention','Positive recall (%)','N-acc (%)',xmax=100,ymax=105)
 axis(f1,'(e) Negative-class F1','Presence threshold','Rejection F1 (%)',ymax=100)
 fixed.set_title('(f) Fixed operating points',loc='left',pad=10);fixed.set_xlabel('Presence threshold');fixed.set_ylabel('G-mIoU@1 (%)');fixed.set_ylim(0,70);fixed.grid(axis='y',color='#DFE7F1',linewidth=.7);fixed.set_axisbelow(True)
 baseline=np.array([r['G-mIoU@1'] for r in series['gmr']['grid']]);deltas=[]
 for key,data in series.items():
  rows=data['grid'];x=np.array([r['threshold'] for r in rows]);g1=np.array([r['G-mIoU@1'] for r in rows]);g3=np.array([r['G-mIoU@3'] for r in rows])
  line(ax1,x,g1,key);line(ax3,x,g3,key);line(delta,x,g1-baseline,key);deltas.extend((g1-baseline).tolist())
  empirical=data['empirical_tradeoff'];line(tradeoff,[r['positive_recall'] for r in empirical],[r['N_acc'] for r in empirical],key,markers=False)
  # Mark the same three fixed operating points without selecting any threshold.
  for threshold in (.4,.5,.6):
   point=rows[int(round(threshold*100))];tradeoff.scatter(point['positive_recall'],point['N_acc'],color=STYLES[key]['color'],marker=STYLES[key]['marker'],s=29,zorder=5)
  line(f1,x,[r['Rej-F1'] for r in rows],key)
 for ax in (ax1,ax3,delta,f1):ax.axvline(.4,color='#94A3B8',linestyle=':',linewidth=.7,zorder=0)
 delta.axhline(0,color='#94A3B8',linewidth=.7,zorder=0)
 low=min(deltas);high=max(deltas);span=max(5,high-low);delta.set_ylim(low-.10*span,high+.14*span);delta.yaxis.set_major_locator(MaxNLocator(6))
 positions=np.arange(3);width=.22 if len(series)==3 else .28
 for number,(key,data) in enumerate(series.items()):
  values=[data['grid'][int(t*100)]['G-mIoU@1'] for t in (.4,.5,.6)]
  offset=(number-(len(series)-1)/2)*width;bars=fixed.bar(positions+offset,values,width=width*.94,color=STYLES[key]['color'],zorder=3)
  fixed.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=3,rotation=90,fontsize=8.5)
 fixed.set_xticks(positions,['0.4','0.5','0.6'])
 handles=[Line2D([0],[0],**STYLES[key],linewidth=2,markersize=4.5,label=LABELS[key]) for key in series]
 fig.suptitle(NAMES[model]+' | Soccer-GMR Standard test',fontsize=14,fontweight='bold',y=.991)
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.951),ncol=len(handles),frameon=False,columnspacing=1.6,handlelength=2.6)
 footer='TEST | frozen baseline checkpoints + S4 epoch 4 | all 101 thresholds shown; no Test-based tuning'
 if model=='flashvtg':footer+=' | no adapter-free baseline'
 fig.text(.5,.016,footer,ha='center',va='bottom',fontsize=10)
 paths=[]
 for suffix in ('png','pdf','svg'):
  path=OUT/f'{model}_fig7_test_threshold_sweep.{suffix}';fig.savefig(path,dpi=220,bbox_inches='tight',pad_inches=.1);paths.append(str(path))
 plt.close(fig)
 return paths

def table(series):
 header=['Model','AUROC (%)','Rej-F1 (τ=0.4, %)','mAP (%)','mR@1 (%)','mR@5 (%)','mR+@5 (%)','G-mIoU@1 (τ=0.4, %)','G-mIoU@3 (τ=0.4, %)']
 lines=['| '+' | '.join(header)+' |','| '+' | '.join(['---']+['---:']*8)+' |'];rows=[]
 for model in MODELS:
  for key,data in series[model].items():
   name=NAMES[model] if key=='original' else NAMES[model]+'-GMR' if key=='gmr' else NAMES[model]+'-GMR + S4_minus_Fusion (epoch 4)'
   if model=='flashvtg' and key=='gmr':name+=' (E0)'
   row={'Model':name,**data['tau_0.4']};rows.append(row)
   lines.append('| '+name+' | '+' | '.join(f'{row[k]:.2f}' for k in KEYS)+' |')
 path=BASE/'TABLE2_OUR_RESULTS.md';before=OUT/'TABLE2_BEFORE_S4.md'
 if not before.exists():shutil.copy2(path,before)
 assert len(rows)==14 and len(lines)==16
 temp=path.with_suffix('.md.tmp');temp.write_text('\n'.join(lines)+'\n');temp.replace(path)
 with (OUT/'TABLE2_OUR_RESULTS.csv').open('w',newline='') as stream:
  w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 from build_table2_pdf import main as render_table2
 render_table2()
 return rows

def main():
 OUT.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True);configure_style()
 gt=read_rows(ARCHIVE/'shared/labels/test.jsonl');labelsha=sha(ARCHIVE/'shared/labels/test.jsonl');assert labelsha=='d8c47dee986b3ea87ba3e008ac2f5f34b0ab000337631e324a6ecffd1dea910c'
 series={};files=[]
 for model in MODELS:
  series[model]=load_baselines(model);series[model]['s4']=measure_s4(model,gt)
  assert set(series[model])==({'gmr','s4'} if model=='flashvtg' else {'original','gmr','s4'})
  files+=render(model,series[model]);print('PLOTTED_FULL',model,flush=True)
 rows=table(series)
 provenance={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'table':str(BASE/'TABLE2_OUR_RESULTS.md'),'table_rows':14,'table_only_markdown':True,'table_threshold':.4,'standard_test_sha256':labelsha,'test_queries':1036,'series_by_backbone':{m:list(v) for m,v in series.items()},'delta_reference':'GMR baseline at the identical threshold','fixed_operating_points':[.4,.5,.6],'source_archive':str(ARCHIVE),'source_archive_manifest_sha256':sha(ARCHIVE/'MANIFEST.json'),'source_archive_not_modified':True,'s4_epoch':4,'s4_no_qcd':True,'flashvtg_s4_kind':'historical verified no-QCD epoch4 reference','original_flashvtg_included':False,'no_mope_or_other_epochs':True,'outputs':files,'generator_sha256':sha(__file__),'baseline_helper_sha256':sha(HERE/'build_summary_and_plots.py'),'models':[{k:v for k,v in d.items() if k not in ['grid','empirical_tradeoff']} for s in series.values() for d in s.values()]}
 dump(OUT/'SOURCE_MANIFEST.json',provenance)
 captions=['# Complete backbone comparisons','', 'S4_minus_Fusion changes the rejection/retention operating point relative to the frozen backbone and GMR variants. Each figure contains only Baseline, Baseline+GMR and Baseline+GMR+S4_minus_Fusion(epoch4); FlashVTG omits the unavailable adapter-free Baseline. No MoPE or other S4 epochs are included.','', 'Panels follow the supplied Figure7 reference: (a) threshold versus G-mIoU@1; (b) threshold versus G-mIoU@3; (c) difference in G-mIoU@1 relative to the GMR baseline at the same threshold; (d) negative-query rejection rate(N-acc) versus positive-query retention(Positive recall); (e) threshold versus rejection F1; (f) G-mIoU@1 at the fixed thresholds0.4/0.5/0.6. All curves use the original1036-query Standard test split; the summary table uses only0.4.','', 'S4 checkpoints are fixed epoch4. Four are from the six-epoch cross-backbone experiment, while FlashVTG is the historical real epoch4 without QCD. The source archive and its originals remain intact. Numerical metrics and101-point curves were recomputed from saved predictions and matched the archived results; checkpoint hashes match the archive and initialization weights match the displayed GMR model.','']
 for model in MODELS:captions.append(f'- [{NAMES[model]}]({model}_fig7_test_threshold_sweep.png): [PDF]({model}_fig7_test_threshold_sweep.pdf), [SVG]({model}_fig7_test_threshold_sweep.svg).')
 (OUT/'README.md').write_text('\n'.join(captions)+'\n')
 print('COMPLETE',len(rows),'table rows;',len(files),'figure exports',flush=True)

if __name__=='__main__':main()
