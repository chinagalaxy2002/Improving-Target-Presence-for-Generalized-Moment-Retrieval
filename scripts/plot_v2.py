"""Six-row live Val monitor; no model or training mutations."""
import json,os,fcntl
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];RUN=R/'runs/clip_v2';OUT=RUN/'monitor';OUT.mkdir(exist_ok=True,parents=True)
def main():
 with (OUT/'.lock').open('a+') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  state=json.loads((RUN/'queue_state.json').read_text());fig,axes=plt.subplots(6,2,figsize=(16,18),gridspec_kw={'width_ratios':[1.5,1]})
  names=['Balanced BCE','Balanced Focal','Balanced BCE + Dice','Hard-Negative Ranking','Soft Boundary','Native Saliency BCE']
  for i,title in enumerate(names,1):
   n=f'E{i}';ax,tab=axes[i-1];tab.axis('off');ax.set_title(n+': '+title,loc='left',fontsize=12,fontweight='bold');ax.set_xlabel('Epoch within stage');ax.set_ylabel('Standard Val mAP (%)');ax.grid(alpha=.25);allvals=[]
   for x,h in zip([0,.33,.55],['Stage','State','Last / picked mAP']):tab.text(x,.95,h,fontweight='bold',fontsize=10)
   for j,(stage,color,marker) in enumerate([('gmr','#292929','o'),('geq_full','#0072B2','D')]):
    d=RUN/n/stage;p=d/'history.jsonl';hist=[json.loads(s) for s in p.read_text().splitlines() if s] if p.exists() else [];y=.73-j*.28
    status='done' if (d/'complete.json').exists() else 'running' if n+'/'+stage in state['running'] else 'waiting'
    tab.text(0,y,'GMR' if stage=='gmr' else 'GMR + GEQ',color=color,fontsize=10);tab.text(.33,y,status,fontsize=10)
    if hist:
     allvals.extend(h['metrics']['brief']['mAP'] for h in hist);best=max(hist,key=lambda h:h['selection_score']);ax.plot([h['epoch'] for h in hist],[h['metrics']['brief']['mAP'] for h in hist],color=color,marker=marker,markersize=3,linestyle='-' if j==0 else '--');ax.scatter([best['epoch']],[best['metrics']['brief']['mAP']],marker='*',s=110,color=color,zorder=5)
     tab.text(.55,y,f"{hist[-1]['metrics']['brief']['mAP']:.2f} / {best['metrics']['brief']['mAP']:.2f}",fontsize=10);tab.text(.55,y-.07,f"picked epoch {best['epoch']}",fontsize=8,color='#64748b')
   ax.set_ylim(0,max(1,max(allvals,default=0)*1.16))
   diag=RUN/n/'diagnostics/clip_val.json'
   if diag.exists():
    v=json.loads(diag.read_text());tab.text(0,.1,f"Val clip AUPRC {v['clip_auprc']:.3f}  |  GT/background gap {v['mean_per_video_gt_background_gap']:.3f}",fontsize=9)
  fig.suptitle('Moment-DETR Clip V2 | GMR-only supervision → unchanged GEQ',fontsize=17,fontweight='bold');fig.tight_layout(rect=[0,.025,1,.975]);fig.text(.5,.012,'GMR selected by Val mAP; GEQ by Val mean G1 (.4/.5/.6). Test excluded.  '+datetime.now().isoformat(timespec='seconds'),ha='center',fontsize=9)
  for ext in ['png','svg','pdf']:
   temp=OUT/('training_val_panel.tmp.'+ext);fig.savefig(temp,dpi=150);temp.replace(OUT/('training_val_panel.'+ext))
  plt.close(fig)
if __name__=='__main__':main()
