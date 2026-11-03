"""Fixed-checkpoint E3 vs explicit baselines; 101-point existence sweep."""
import csv,gzip,json,shutil
from pathlib import Path
import numpy as np
from soccergmr.io import dump_json,read_jsonl,sha256
from soccergmr.evaluation import evaluate
from soccergmr.evaluation.official.metrics import _clean_pred_windows,_compute_set_iou_score,prepare_submission_for_gmiou,compute_G_mIoU
R=Path(__file__).resolve().parents[1];OUT=R/'results/E3_baseline_comparison';OUT.mkdir(parents=True,exist_ok=True)
V1=Path('/home/guoxiangyu/SoccerGMR-GEQ-clipseg/runs/clipseg/formal');V2=Path('/home/guoxiangyu/SoccerGMR-GEQ-clip-v2/runs/clip_v2');JOINT=R/'runs/clip_joint'
METHODS=[('gmr_baseline','GMR baseline',V1/'A/gmr',V1/'A/gmr_test'),('original_geq','Original GMR + GEQ',V1/'A/geq_full',V1/'A/geq_full_test'),('e3_noaux','E3 GEQ (no aux)',V2/'E3/geq_full',V2/'E3/geq_full_test'),('e3_aux','E3 GEQ + aux',JOINT/'E3/geq_full',JOINT/'E3/geq_full_test')]
cfg=json.loads((R/'configs/experiments/E3.json').read_text());curves=[];summary=[];data={};audits=[]
for split,count in [('test_standard',1036),('test_full',2970)]:
 gtpath=Path(cfg['data']['splits'][split]);gt=read_jsonl(gtpath);assert len(gt)==count;data[split]={}
 for key,label,train,test in METHODS:
  complete=json.loads((train/'complete.json').read_text());checkpoint_hash=sha256(train/'best.pt');assert checkpoint_hash==complete['best_sha256'] and not complete['test_used_for_selection']
  part=test/split;marker=json.loads((part/'complete.json').read_text());metrics=json.loads((part/'metrics.json').read_text());pred=read_jsonl(part/'predictions.jsonl')
  assert marker['rows']==count and metrics['provenance']['checkpoint_sha256']==checkpoint_hash
  assert metrics['provenance']['label_sha256']==sha256(gtpath)
  assert all(sha256(part/file)==marker[k] for file,k in [('predictions.jsonl','predictions_sha256'),('metrics.json','metrics_sha256')])
  verified=evaluate(pred,gt)
  for k in ['brief','gmiou','mAP_detail']:assert verified[k]==metrics[k]
  byid={p['qid']:p for p in pred};assert len(byid)==len(pred)==count and set(byid)=={g['qid'] for g in gt};pred=[byid[g['qid']] for g in gt]
  scores=np.array([p['pred_exist_score'] for p in pred]);assert np.isfinite(scores).all()
  accepted=np.array([_compute_set_iou_score([w[:2] for w in _clean_pred_windows(p['pred_relevant_windows'],max_pred_windows=10)[:1]],g['relevant_windows']) for p,g in zip(pred,gt)])
  rejected=np.array([_compute_set_iou_score([],g['relevant_windows']) for g in gt]);thresholds=[i/100 for i in range(101)]
  ys=[round(float(100*np.where(scores>t,accepted,rejected).mean()),2) for t in thresholds]
  for t in [0.,.4,.5,.6,1.]:
   gated,_=prepare_submission_for_gmiou(pred,cls_threshold=t,max_pred_windows=10);official=compute_G_mIoU(gated,gt,k_list=(1,))['G-mIoU@1'];assert abs(official-ys[int(round(t*100))])<1e-9
  for t in [.4,.5,.6]:assert ys[int(t*100)]==metrics['gmiou'][str(t)]['G-mIoU@1']
  row=dict(split=split,method=key,label=label,mAP=metrics['brief']['mAP'],G1_04=ys[40],G1_05=ys[50],G1_06=ys[60],G1_mean=float(np.mean([ys[40],ys[50],ys[60]])),checkpoint_sha256=checkpoint_hash,best_epoch=complete['best_epoch'])
  summary.append(row);data[split][key]=dict(**row,thresholds=thresholds,G1=ys)
  curves.extend(dict(split=split,method=key,threshold=t,G_mIoU_at_1=y) for t,y in zip(thresholds,ys))
  target=OUT/'predictions'/split/key;target.mkdir(parents=True,exist_ok=True)
  with gzip.GzipFile(filename=str(target/'predictions.jsonl.gz'),mode='wb',mtime=0) as dst:dst.write((part/'predictions.jsonl').read_bytes())
  shutil.copy2(part/'metrics.json',target/'metrics.json');shutil.copy2(part/'complete.json',target/'complete.json')
  audits.append(dict(split=split,method=key,rows=count,checkpoint_sha256=checkpoint_hash,predictions_sha256=marker['predictions_sha256'],compressed_sha256=sha256(target/'predictions.jsonl.gz'),metrics_recomputed=True,official_sweep_checks=[0,.4,.5,.6,1],label_sha256=sha256(gtpath)))
  print('VERIFIED',split,key,row['mAP'],row['G1_mean'],flush=True)
 for row in summary:
  if row['split']==split:
   base=data[split]['gmr_baseline'];row['delta_map_vs_gmr']=row['mAP']-base['mAP'];row['delta_G1_vs_gmr']=row['G1_mean']-base['G1_mean']
for file,rows in [('METRICS.csv',summary),('CURVES.csv',curves)]:
 with (OUT/file).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
dump_json(OUT/'PLOT_DATA.json',data);dump_json(OUT/'AUDIT.json',dict(passed=True,checkpoints_fixed_before_test=True,test_used_for_selection=False,rows=audits,threshold_decision='pred_exist_score > threshold',threshold_grid='0:.01:1',note='E3 requested evaluation after user paused the queue. All six new GEQ Val choices had already finalized.'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
colors=['#333333','#0072B2','#009E73','#D55E00'];styles=['-','--',':','-.'];fig,axes=plt.subplots(1,2,figsize=(13,4.8),sharey=True)
for ax,split,title in zip(axes,['test_standard','test_full'],['Standard Test (1,036 queries)','Full-Decodable Test (2,970 / 2,986)']):
 for (key,label,_,_),color,style in zip(METHODS,colors,styles):
  row=data[split][key];ax.plot(row['thresholds'],row['G1'],color=color,linestyle=style,lw=2,label=f"{label} | mAP {row['mAP']:.2f}")
  ax.scatter([.4,.5,.6],[row['G1'][40],row['G1'][50],row['G1'][60]],color=color,s=18,zorder=4)
 ax.set_title(title,fontweight='bold');ax.set_xlabel('Existence threshold');ax.set_xlim(0,1);ax.set_ylim(0,100);ax.grid(alpha=.25);ax.legend(fontsize=8.5,loc='upper right');ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('G-mIoU@1 (0–100)');fig.suptitle('Moment-DETR E3 GEQ + aux vs. baselines | fixed-checkpoint Test',fontsize=14,fontweight='bold');fig.tight_layout(rect=[0,.06,1,.93]);fig.text(.5,.025,'mAP uses ungated positive-query candidates. Threshold sweep is descriptive; no Test-based selection.',ha='center',fontsize=9)
for ext in ['png','svg','pdf']:fig.savefig(OUT/('G1_THRESHOLDS.'+ext),dpi=180)
plt.close(fig)
lines=['# E3 GEQ＋aux 与 baseline 固定 Test 对比','','“baseline”同时提供三个明确口径：原始 GMR baseline、原始 GMR→GEQ 控制，以及 E3 同一 GMR 起点的无辅助损失 GEQ。E3 GEQ＋aux 从已完成的 E3 GMR best 开始，继承并更新辅助头，新增 0.1 × (Balanced BCE + 0.2 × Dice) 辅助监督。','','| Split | Method | mAP | G1@0.4 | G1@0.5 | G1@0.6 | G1 三点均值 |','|---|---|---:|---:|---:|---:|---:|']
for row in summary:lines.append(f"| {row['split']} | {row['label']} | {row['mAP']:.2f} | {row['G1_04']:.2f} | {row['G1_05']:.2f} | {row['G1_06']:.2f} | {row['G1_mean']:.2f} |")
lines+=['','![存在性阈值曲线](G1_THRESHOLDS.png)','','曲线覆盖0到1、步长0.01，共101个点；保留规则是分数严格大于阈值。mAP 不经存在性门控，因此不随该阈值改变。0.4/0.5/0.6 是预先固定的三点报告，不使用 Test 调整模型或阈值。每种方法在两个 Test 使用同一个已由 Val 选定的 checkpoint。','', 'GMR baseline 与 E3 的 GMR 训练方式不同；E3 no-aux 与 E3 aux 共用完全相同的 GMR 初始化，因而是更直接的微调目标对照。单 seed=2023，实际早停时长不同。Standard 与 Full Test 存在重叠；Full 仅覆盖既有可解码子集2970/2986。','', '[指标CSV](METRICS.csv) · [全部101点曲线CSV](CURVES.csv) · [PDF](G1_THRESHOLDS.pdf) · [SVG](G1_THRESHOLDS.svg) · [重算及哈希审计](AUDIT.json)']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
