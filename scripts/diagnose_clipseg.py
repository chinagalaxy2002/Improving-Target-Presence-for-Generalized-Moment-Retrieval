"""Fixed-checkpoint Val-only clip diagnostics and same-video query heatmaps."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from soccergmr import checkpoint as ck
from soccergmr.backbones.registry import build,get_spec
from soccergmr.clip import clip_enabled,wrap_gmr,masked_clip_loss
from soccergmr.config import load_config
from soccergmr.data import SoccerDataset,collate,move
from soccergmr.geq import GEQFull
from soccergmr.io import dump_json,sha256


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    cfg=load_config(args.config);saved=ck.load(args.checkpoint);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    complete=json.loads((Path(args.checkpoint).parent/'complete.json').read_text())
    assert complete['best_sha256']==sha256(args.checkpoint) and saved['source']==ck.source_identity()
    assert saved['config']['identity']==cfg['identity'] and complete['identity']==saved['identity']
    base,_=build(cfg,'cuda:0');spec=get_spec(cfg['backbone'])
    model=GEQFull(base,spec,clip_enabled(cfg)) if saved['stage']=='geq_full' else wrap_gmr(base,spec,cfg)
    model=model.to('cuda:0');model.load_state_dict(saved['model'],strict=True);model.eval()
    if not hasattr(model,'clip_head') or model.clip_head is None:
        dump_json(out/'metrics.json',dict(enabled=False,reason='A has no clip head'));return
    data=SoccerDataset(cfg,'val',spec);totals=dict(bce=0.,videos=0,tp=0,fp=0,fn=0,positive_probability=0.,negative_probability=0.,positive_clips=0,negative_clips=0)
    for metas,x,y in DataLoader(data,batch_size=cfg['protocol']['eval_batch_size'],collate_fn=collate,num_workers=2):
        x,y=move(x,'cuda:0'),move(y,'cuda:0');z=model(**x)['clip_logits'];prob=z.sigmoid();valid=x['src_vid_mask'].bool();truth=y['saliency_all_labels'].bool();pred=prob>=.5
        totals['bce']+=float(masked_clip_loss(z,y['saliency_all_labels'],valid))*len(metas);totals['videos']+=len(metas)
        for name,mask in [('tp',pred&truth&valid),('fp',pred&~truth&valid),('fn',~pred&truth&valid)]:totals[name]+=int(mask.sum())
        for prefix,mask in [('positive',truth&valid),('negative',~truth&valid)]:
            totals[prefix+'_probability']+=float(prob[mask].sum());totals[prefix+'_clips']+=int(mask.sum())
    result=dict(split='Standard Val',checkpoint_sha256=sha256(args.checkpoint),source=saved['source'],masked_bce=totals['bce']/totals['videos'],precision=totals['tp']/max(1,totals['tp']+totals['fp']),recall=totals['tp']/max(1,totals['tp']+totals['fn']),totals=totals)
    groups=defaultdict(list)
    for i,r in enumerate(data.data):groups[r['vid']].append(i)
    pairs=[]
    for vid,indices in sorted(groups.items()):
        positives=[i for i in indices if data.data[i]['relevant_windows']];nulls=[i for i in indices if not data.data[i]['relevant_windows']]
        if positives and nulls:pairs.append((positives[0],nulls[0]))
        if len(pairs)==3:break
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(max(1,len(pairs)),1,figsize=(11,3.1*max(1,len(pairs))),squeeze=False)
    swap=[];arrays={}
    for n,(i,j) in enumerate(pairs):
        metas,x,y=collate([data[i],data[j]]);x,y=move(x,'cuda:0'),move(y,'cuda:0');prob=model(**x)['clip_logits'].sigmoid().cpu().numpy();truth=y['saliency_all_labels'].cpu().numpy()
        assert torch.equal(x['src_vid'][0],x['src_vid'][1])
        matrix=np.stack([truth[0],prob[0],truth[1],prob[1]])
        ax=axes[n,0];ax.imshow(matrix,vmin=0,vmax=1,aspect='auto',extent=(0,metas[0]['duration'],4,0),cmap='viridis');ax.set_yticks([.5,1.5,2.5,3.5],['GT positive','P positive','GT null','P null']);ax.set_xlabel('Time (seconds)')
        q1=str(metas[0].get('query',''))[:100];q2=str(metas[1].get('query',''))[:100]
        ax.set_title(f"{metas[0]['vid']} | {q1}\nNull query: {q2}",fontsize=9)
        swap.append(dict(vid=metas[0]['vid'],qids=[metas[0]['qid'],metas[1]['qid']],mean_absolute_probability_difference=float(np.abs(prob[0]-prob[1]).mean()),positive_mean=float(prob[0].mean()),null_mean=float(prob[1].mean()),positive_gt_mean=float(prob[0][truth[0]>0].mean())))
        arrays['pair'+str(n)]=matrix
    if not pairs:axes[0,0].text(.1,.5,'No positive/null query pair shares a Val video')
    fig.tight_layout();fig.savefig(out/'query_heatmaps.png',dpi=160);fig.savefig(out/'query_heatmaps.pdf');plt.close(fig)
    np.savez_compressed(out/'query_heatmaps.npz',**arrays);result['same_video_query_pairs']=swap
    result['interpretation']='Nonzero differences demonstrate query dependence; semantic correctness requires GT alignment and quantitative metrics.'
    dump_json(out/'metrics.json',result)


if __name__=='__main__':main()
