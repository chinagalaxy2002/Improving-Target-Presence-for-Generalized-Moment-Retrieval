#!/usr/bin/env python3
"""Live Standard-Val-only monitor for the expansion queue."""
import argparse
import fcntl
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from soccergmr.io import read_jsonl
from soccergmr.evaluation.official.metrics import _clean_pred_windows, _compute_set_iou_score

NAMES={'bm_detr':'BM-DETR','bam_detr':'BAM-DETR','uvcom':'UVCOM'}
STYLES={'gmr':{'label':'GMR','color':'#252525','marker':'o','linestyle':'-'},
        'geq_full':{'label':'GMR + GEQ Full','color':'#0072B2','marker':'D','linestyle':'--'}}


def snapshot(directory, gt):
    path=directory/'history.jsonl'
    if not path.exists(): return [], None
    # Writers atomically replace history and predictions. Only display an
    # oracle when the current history was published after both best files.
    before=path.stat().st_mtime_ns
    try:
        rows=read_jsonl(path)
        improved=[r for r in rows if r['improved']]
        best=improved[-1] if improved else None
        predpath=directory/'best_val_predictions.jsonl'
        ckpath=directory/'best.pt'
        if not best or not predpath.is_file() or not ckpath.is_file(): return rows,None
        pm=predpath.stat().st_mtime_ns; cm=ckpath.stat().st_mtime_ns
        if before < max(pm,cm): return rows,None
        predictions=read_jsonl(predpath)
        if path.stat().st_mtime_ns != before or predpath.stat().st_mtime_ns != pm or ckpath.stat().st_mtime_ns != cm: return rows,None
        byid={p['qid']:p for p in predictions}
        if len(byid)!=len(gt) or set(byid)!={g['qid'] for g in gt}:return rows,None
        oracle=[]
        for k in (1,3):
            total=0.
            for g in gt:
                p=byid[g['qid']]
                if p['vid'] != g['vid']:return rows,None
                windows=_clean_pred_windows(p['pred_relevant_windows'],max_pred_windows=10)
                selected=[w[:2] for w in windows[:k]] if g['relevant_windows'] else []
                total+=_compute_set_iou_score(selected,g['relevant_windows'])
            oracle.append(total/len(gt))
        return rows,oracle
    except (OSError,ValueError,KeyError,json.JSONDecodeError):
        return [],None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    root=Path(args.root).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'.monitor.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        queue=json.loads((root/'queue_state.json').read_text())
        running=queue.get('running',{}); completed=set(queue.get('completed',[])); failure=queue.get('failure') or {}
        fig=plt.figure(figsize=(14,10.2));gs=fig.add_gridspec(3,2,width_ratios=(1.55,1),left=.075,right=.98,top=.84,bottom=.14,hspace=.52,wspace=.10)
        fig.suptitle('SoccerGMR expansion | Standard Val mAP and selected-checkpoint oracle',fontsize=16,fontweight='bold',y=.975)
        fig.text(.5,.935,'BM-DETR, BAM-DETR, UVCOM  ·  seed 2023  ·  GMR → GEQ Full',ha='center',fontsize=10.5,color='#475569')
        legend=[Line2D([0],[0],lw=2,markersize=5,**style) for style in STYLES.values()]
        fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.915),ncol=2,frameon=False,fontsize=10)
        for i,(name,title) in enumerate(NAMES.items()):
            cfg=json.loads((root/'configs'/f'{name}.json').read_text());gt=read_jsonl(cfg['data']['splits']['val'])
            ax=fig.add_subplot(gs[i,0]);table=fig.add_subplot(gs[i,1]);table.axis('off')
            ax.set_title(title,loc='left',fontweight='bold',fontsize=12)
            ax.set_xlabel('Epoch within stage');ax.set_ylabel('Standard Val mAP (%)')
            ax.grid(color='#d5d9df',alpha=.65,linewidth=.7);ax.spines[['top','right']].set_visible(False)
            columns=[.02,.32,.52,.78];heads=['Method','State','Last / picked','Oracle G1 / G3']
            for x,head in zip(columns,heads):table.text(x,.93,head,fontsize=9,fontweight='bold',transform=table.transAxes)
            all_epochs=[];all_values=[]
            for j,stage in enumerate(('gmr','geq_full')):
                key=f'{name}/{stage}';directory=root/name/stage;rows,oracle=snapshot(directory,gt)
                state='running' if key in running else 'done' if key in completed or (directory/'complete.json').exists() else 'failed' if failure.get('job')==[name,stage] else 'waiting'
                style=STYLES[stage];y=.73-j*.26
                table.text(columns[0],y,style['label'],color=style['color'],fontsize=9,fontweight='bold',transform=table.transAxes)
                table.text(columns[1],y,state,fontsize=9,color='#475569',transform=table.transAxes)
                bests=[r for r in rows if r['improved']]
                if rows:
                    epochs=[r['epoch'] for r in rows];values=[r['metrics']['brief']['mAP'] for r in rows]
                    all_epochs.extend(epochs);all_values.extend(values)
                    ax.plot(epochs,values,lw=1.8,markersize=3.5,**{k:style[k] for k in ('color','marker','linestyle')})
                    best=bests[-1] if bests else None
                    if best:
                        ax.scatter([best['epoch']],[best['metrics']['brief']['mAP']],marker='*',s=110,c=style['color'],edgecolors='white',linewidths=.5,zorder=5)
                    valuestr=f"{values[-1]:.2f} / {best['metrics']['brief']['mAP']:.2f}" if best else f'{values[-1]:.2f} / —'
                else:valuestr='— / —'
                table.text(columns[2],y,valuestr,fontsize=9,transform=table.transAxes)
                table.text(columns[3],y,('%.3f / %.3f'%tuple(oracle)) if oracle else '— / —',fontsize=9,transform=table.transAxes)
                if bests:
                    best=bests[-1];selection='Val mAP' if stage=='gmr' else 'mean G1 @ .4/.5/.6'
                    table.text(columns[2],y-.075,f"picked epoch {best['epoch']} · {selection}",fontsize=7.5,color='#64748b',transform=table.transAxes)
            if not any((root/name/stage/'history.jsonl').exists() for stage in STYLES):ax.text(.5,.5,'Waiting for first complete validation',ha='center',transform=ax.transAxes,color='#64748b')
            ax.set_ylim(0,max(1,max(all_values,default=0)*1.15))
            if len(set(all_epochs))==1:
                ax.set_xlim(all_epochs[0]-.5,all_epochs[0]+.5);ax.set_xticks([all_epochs[0]])
            else:ax.xaxis.get_major_locator().set_params(integer=True)
        now=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S CST')
        fig.text(.5,.055,'★ Val-selected epoch: GMR by positive-query mAP; GEQ Full by mean G-mIoU@1 at 0.4 / 0.5 / 0.6.',ha='center',fontsize=9,color='#475569')
        fig.text(.5,.033,'Oracle uses perfect presence and each selected checkpoint’s fixed candidates; Test data excluded.   Updated '+now,ha='center',fontsize=8.5,color='#475569')
        for ext in ('png','svg'):
            tmp=out/('training_val_panel.tmp.'+ext); fig.savefig(tmp,dpi=160,facecolor='white');os.replace(tmp,out/('training_val_panel.'+ext))
        plt.close(fig)
        (out/'updated.txt').write_text(now+'\n')

if __name__=='__main__':main()
