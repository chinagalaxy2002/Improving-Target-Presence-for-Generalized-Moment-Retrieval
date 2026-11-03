"""Read-only 11-family Val progress; Moment is explicitly marked reused."""
import json,fcntl
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];RUN=R/'runs/e3_eleven/formal';OUT=RUN/'monitor';OUT.mkdir(parents=True,exist_ok=True)
NAMES=['moment_detr','eatr','qd_detr','cg_detr','sim_detr','bm_detr','bam_detr','uvcom','tr_detr','taskweave','ld_detr']
def main():
 with (OUT/'.lock').open('a+') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  q=json.loads((RUN/'queue_state.json').read_text());fig,axes=plt.subplots(11,2,figsize=(16,29),gridspec_kw={'width_ratios':[1.5,1]})
  for i,n in enumerate(NAMES):
   ax,tab=axes[i];tab.axis('off');ax.set_title(n+(' [existing E3 reused]' if n=='moment_detr' else ''),loc='left',fontweight='bold',fontsize=12);ax.set_xlabel('Epoch within stage');ax.set_ylabel('Standard Val mAP (%)');ax.grid(alpha=.25);vals=[]
   for x,h in zip([0,.32,.56],['Stage','State','Last / picked mAP']):tab.text(x,.95,h,fontsize=10,fontweight='bold')
   for j,(stage,color,marker) in enumerate([('gmr','#292929','o'),('geq_full','#D55E00','D')]):
    folder=RUN/n/stage;p=folder/'history.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines() if s] if p.exists() else [];state='reused' if n=='moment_detr' else 'done' if (folder/'complete.json').exists() else 'running' if n+'/'+stage in q['running'] else 'waiting';y=.73-j*.28
    tab.text(0,y,'GMR + E3' if j==0 else 'GEQ + aux',color=color,fontsize=10);tab.text(.32,y,state,fontsize=10)
    if rows:
     values=[x['metrics']['brief']['mAP'] for x in rows];vals+=values;b=max(rows,key=lambda x:x['selection_score']);ax.plot([x['epoch'] for x in rows],values,color=color,linestyle='-' if j==0 else '--',marker=marker,markersize=3);ax.scatter([b['epoch']],[b['metrics']['brief']['mAP']],marker='*',s=90,color=color)
     tab.text(.56,y,f"{values[-1]:.2f} / {b['metrics']['brief']['mAP']:.2f}",fontsize=10);tab.text(.56,y-.08,f"picked epoch {b['epoch']}",fontsize=8,color='#64748b')
   ax.set_ylim(0,max(1,max(vals,default=0)*1.16))
   diag=RUN/n/'diagnostics/clip_val.json'
   if diag.exists():
    x=json.loads(diag.read_text());tab.text(0,.1,f"Val clip AUPRC {x['clip_auprc']:.3f} | GT/background gap {x['mean_per_video_gt_background_gap']:.3f}",fontsize=9)
  fig.suptitle('11 backbones | E3 Balanced BCE + Dice → GEQ + aux',fontweight='bold',fontsize=17);fig.tight_layout(rect=[0,.02,1,.985]);fig.text(.5,.01,'Seed2023 · max200 · GMR Val mAP / patience20 · GEQ Val mean G1(.4/.5/.6) / patience10 · '+datetime.now().isoformat(timespec='seconds'),ha='center',fontsize=9)
  for ext in ['png','svg','pdf']:
   t=OUT/('training_val_panel.tmp.'+ext);fig.savefig(t,dpi=140);t.replace(OUT/('training_val_panel.'+ext))
  plt.close(fig)
if __name__=='__main__':main()
