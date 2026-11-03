"""Replot the published experiment archive without dataset/checkpoints."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/clipseg');p.add_argument('--output',default='runs/plots');a=p.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    oracle=json.loads((root/'oracle.json').read_text());chosen=json.loads((root/'selection.json').read_text())['chosen'];suffix=chosen['name'][2:]
    panels=[('GMR baseline and A control',[('A/gmr','GMR','#292929','-'),('A/geq_full','A GEQ','#0072B2','--')]),('B: clip supervision in Stage 2',[(f'B_{s}/geq_full',f'B lambda={w}',c,'--') for s,w,c in [('01',.1,'#E69F00'),('03',.3,'#009E73'),('10',1.,'#CC79A7')]]),('C/D shared GMR parents',[(f'C_{s}/gmr',f'GMR lambda={w}',c,'-') for s,w,c in [('01',.1,'#292929'),('03',.3,'#009E73'),('10',1.,'#CC79A7')]]),('C: Stage 1 clip only',[(f'C_{suffix}/gmr','C GMR','#292929','-'),(f'C_{suffix}/geq_full','C GEQ','#D55E00','--')]),('D: clip in both stages',[(f'C_{suffix}/gmr','Shared GMR','#292929','-'),(f'D_{suffix}/geq_full','D GEQ','#7B4AB5','--')])]
    fig,axes=plt.subplots(5,2,figsize=(16,17),gridspec_kw={'width_ratios':[1.6,1]});fig.suptitle('Moment-DETR clip supervision | archived Standard Val results',fontsize=16,fontweight='bold')
    for pair,(title,series) in zip(axes,panels):
        ax,table=pair;table.axis('off');ax.set_title(title,loc='left');ax.set_xlabel('Epoch within stage');ax.set_ylabel('Standard Val mAP (%)');ax.grid(alpha=.2)
        table.text(0,.98,'Method | Last / picked mAP | Oracle G1 / G3',fontsize=10,fontweight='bold')
        for j,(key,label,color,line) in enumerate(series):
            rows=[json.loads(x) for x in (root/'execution'/key/'history.jsonl').read_text().splitlines() if x];best=max(rows,key=lambda x:x['selection_score']);values=[r['metrics']['brief']['mAP'] for r in rows]
            ax.plot([r['epoch'] for r in rows],values,color=color,linestyle=line,marker='o',markersize=3,label=label);ax.scatter(best['epoch'],best['metrics']['brief']['mAP'],marker='*',s=110,color=color,zorder=4)
            o=oracle[key];text=f"{label}: {values[-1]:.2f} / {best['metrics']['brief']['mAP']:.2f} | {o[0]:.3f} / {o[1]:.3f}\npicked epoch {best['epoch']}"
            table.text(0,.76-j*.25,text,color=color,fontsize=10,linespacing=1.8)
        ax.set_ylim(bottom=0);ax.legend(fontsize=8)
    fig.text(.5,.012,'Stars: GMR selected by Val mAP; GEQ by mean Val G1 at .4/.5/.6. Oracle: perfect presence, fixed candidates. No Test selection.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.97))
    for ext in ['png','svg','pdf']:fig.savefig(out/('training_val_panel.'+ext),dpi=160)
    plt.close(fig)
if __name__=='__main__':main()
