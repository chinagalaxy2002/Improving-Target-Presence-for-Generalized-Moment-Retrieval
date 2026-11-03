"""Selected GMR Val diagnostics, independent of model selection."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from soccergmr import checkpoint as ck
from soccergmr.config import load_config
from soccergmr.clip import wrap_gmr,clip_auxiliary_loss,clip_enabled
from soccergmr.geq import GEQFull
from soccergmr.backbones.registry import build,get_spec
from soccergmr.data import SoccerDataset,collate,move
from soccergmr.io import dump_json,sha256

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
 c=load_config(a.config);saved=ck.load(root/'geq_full/best.pt');assert saved['source']==ck.source_identity() and saved['config']['identity']==c['identity']
 base,_=build(c,'cuda:0');model=GEQFull(base,get_spec(c['backbone']),clip_head=clip_enabled(c)).to('cuda:0');model.load_state_dict(saved['model'],strict=True);model.eval()
 ds=SoccerDataset(c,'val',get_spec(c['backbone']));truth=[];scores=[];gaps=[];losses=[];visual=[]
 for metas,x,y in DataLoader(ds,batch_size=c['protocol']['eval_batch_size'],num_workers=2,collate_fn=collate):
  x,y=move(x,'cuda:0'),move(y,'cuda:0');o=model(**x);z=o['saliency_scores'] if c['clip']['method']=='native' else o['clip_logits'];prob=z.sigmoid();valid=x['src_vid_mask'].bool();hard=y['saliency_all_labels'].bool()
  loss,_=clip_auxiliary_loss(z,y,valid,c['clip']);losses.append((float(loss),len(metas)))
  truth.extend(hard[valid].cpu().tolist());scores.extend(prob[valid].cpu().tolist())
  for i,m in enumerate(metas):
   pos=hard[i]&valid[i];neg=~hard[i]&valid[i]
   if pos.any() and neg.any():gaps.append(float(prob[i][pos].mean()-prob[i][neg].mean()))
   if len(visual)<4 and pos.any():visual.append((m,prob[i][valid[i]].cpu().numpy(),hard[i][valid[i]].cpu().numpy()))
 t=np.array(truth,dtype=bool);s=np.array(scores);thresholds={}
 for v in [.1,.3,.5]:
  pred=s>=v;tp=(pred&t).sum();thresholds[str(v)]=dict(precision=float(tp/max(1,pred.sum())),recall=float(tp/max(1,t.sum())),false_positive_rate=float((pred&~t).sum()/max(1,(~t).sum())))
 result=dict(split='Standard Val',checkpoint_sha256=sha256(root/'geq_full/best.pt'),source=saved['source'],method=c['clip']['method'],positive_clips=int(t.sum()),negative_clips=int((~t).sum()),clip_auprc=float(average_precision_score(t,s)),mean_gt_probability=float(s[t].mean()),mean_background_probability=float(s[~t].mean()),mean_per_video_gt_background_gap=float(np.mean(gaps)),thresholds=thresholds,aux_loss=sum(v*n for v,n in losses)/sum(n for v,n in losses),selection_used=False)
 dump_json(root/'diagnostics/clip_val.json',result)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(len(visual),1,figsize=(10,2.5*len(visual)),squeeze=False)
 for ax,(m,prob,gt) in zip(axes[:,0],visual):
  ax.imshow(np.stack([gt,prob]),aspect='auto',vmin=0,vmax=1,extent=[0,m['duration'],2,0],cmap='viridis');ax.set_yticks([.5,1.5],['GT','Probability']);ax.set_title(str(m.get('query',m['qid']))[:110],fontsize=9);ax.set_xlabel('Time (seconds)')
 fig.tight_layout();fig.savefig(root/'diagnostics/clip_heatmaps.png',dpi=160);fig.savefig(root/'diagnostics/clip_heatmaps.pdf');plt.close(fig)
if __name__=='__main__':main()
