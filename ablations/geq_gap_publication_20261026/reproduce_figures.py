#!/usr/bin/env python3
"""Rebuild three-method figures and threshold data from packaged predictions."""
import csv,gzip,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from sklearn.metrics import roc_auc_score,average_precision_score
from official_eval.metrics import _clean_pred_windows,_compute_set_iou_score,prepare_submission_for_gmiou,compute_G_mIoU
ROOT=Path(__file__).resolve().parent
METHODS=[('E0','GMR', '#64748b','--','o'),('E1','GMR + GEQ','#0072B2','-','s'),('GAP','GMR + GEQ + GAP','#D55E00','-.','^')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
def rows(name):
    with gzip.open(ROOT/'evidence'/f'{name}.jsonl.gz','rt') as f:return [json.loads(l) for l in f if l.strip()]
def save(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=210,bbox_inches='tight')
    plt.close(fig)
def writecsv(name,data):
    with (ROOT/'tables'/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def main():
    gt=rows('test_ground_truth');y=np.array([bool(g['relevant_windows']) for g in gt]);preds={};scores={};summary=[];curve={};hist=[];official_checks=0
    assert len(gt)==1036 and y.sum()==544 and (~y).sum()==492
    for key,label,color,style,marker in METHODS:
        p=rows(key+'_test_predictions');mp={x['qid']:x for x in p};assert len(mp)==len(p)==len(gt) and set(mp)=={g['qid'] for g in gt};p=[mp[g['qid']] for g in gt];preds[key]=p
        s=np.array([x['pred_exist_score'] for x in p]);assert np.isfinite(s).all() and ((s>=0)&(s<=1)).all();scores[key]=s
        pos=s[y].mean();neg=s[~y].mean();m=json.loads((ROOT/'evidence'/f'{key}_test_metrics.json').read_text())['0.4']['brief']
        b=dict(method=label,mu_pos=float(pos),mu_neg=float(neg),mean_gap=float(pos-neg),D_pos=float(1-pos),D_neg=float(neg),D_balanced=float((1-pos+neg)/2),S=float(50*(1+pos-neg)),AUROC=float(100*roc_auc_score(y,s)),AUPRC=float(100*average_precision_score(y,s)))
        gate=s>.4;b.update(PosRecall=float(100*gate[y].mean()),TNR=float(100*(~gate[~y]).mean()),**{k:m[k] for k in ['mAP','mR@1','mR@5','mR+@5','G-mIoU@1','G-mIoU@3']});summary.append(b)
        ungated={k:np.array([_compute_set_iou_score([w[:2] for w in _clean_pred_windows(x.get('pred_relevant_windows',[]),10)[:k]],g['relevant_windows']) for x,g in zip(p,gt)]) for k in [1,3]}
        points=[]
        for t in np.linspace(0,1,101):
            accept=s>t;tn=int((~accept & ~y).sum());fp=int((accept & ~y).sum());fn=int((~accept & y).sum())
            point=dict(method=key,threshold=float(t),**{f'G-mIoU@{k}':float(100*np.where(accept,ungated[k],~y).mean()) for k in [1,3]},PosRecall=float(100*accept[y].mean()),TNR=float(100*(~accept[~y]).mean()),Rej_F1=float(200*tn/max(1,2*tn+fp+fn)))
            # Full-grid cross-check with original evaluator, including score>tau equality semantics.
            gated,_=prepare_submission_for_gmiou(p,float(t),10);official=compute_G_mIoU(gated,gt,k_list=(1,3))
            assert all(abs(point[f'G-mIoU@{k}']-official[f'G-mIoU@{k}'])<=.011 for k in [1,3]);official_checks+=1;points.append(point)
        curve[key]=points
        for group,mask in [('positive',y),('negative',~y)]:
            den,edges=np.histogram(s[mask],bins=np.linspace(0,1,41),density=True);assert abs(np.sum(den*np.diff(edges))-1)<1e-12
            for j,v in enumerate(den):hist.append(dict(method=key,group=group,bin_left=float(edges[j]),bin_right=float(edges[j+1]),density=float(v)))
        for tau in [.4,.5,.6]:
            q=points[int(round(tau*100))];original=json.loads((ROOT/'evidence'/f'{key}_test_metrics.json').read_text())[str(tau)]['brief'];assert abs(q['G-mIoU@1']-original['G-mIoU@1'])<=.011
    writecsv('three_methods_test.csv',summary);writecsv('threshold_sweep_101.csv',[r for key,*_ in METHODS for r in curve[key]]);writecsv('existence_histogram.csv',hist)
    distributions(scores,y);thresholds(curve,summary);overview()
    payload=dict(passed=True,test_queries=len(gt),positive=int(y.sum()),negative=int((~y).sum()),qid_alignment=True,all_density_integrals_one=True,official_threshold_crosschecks=official_checks,thresholds_per_model=101,strict_acceptance='score > threshold',fixed_threshold=.4,selection='Val-selected margin=0.6 weight=2 epoch4',no_test_threshold_selection=True)
    (ROOT/'FIGURE_AUDIT.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload))
def distributions(scores,y):
    fig,axes=plt.subplots(1,2,figsize=(12.4,4.6))
    for ax,mask,title in zip(axes,[y,~y],['Positive queries (n=544)','Negative queries (n=492)']):
        for key,label,c,ls,m in METHODS:
            vals=scores[key][mask];den,edges=np.histogram(vals,bins=np.linspace(0,1,41),density=True)
            ax.stairs(den,edges,color=c,fill=True,alpha=.09);ax.stairs(den,edges,color=c,ls=ls,lw=1.8,label=f'{label}: mean {vals.mean():.3f}, median {np.median(vals):.3f}')
            ax.axvline(np.median(vals),color=c,ls=ls,lw=.9,alpha=.65)
        ax.axvline(.4,color='#334155',ls=':',lw=1.1);ax.set(xlim=(0,1),xlabel='Existence score',ylabel='Density',title=title);ax.legend(fontsize=7.7,frameon=False,loc='upper left');ax.grid(axis='y',alpha=.18)
    fig.suptitle('FlashVTG: Test existence-score distributions',fontsize=13,fontweight='bold');fig.text(.5,.01,'Standard Test: 1,036 queries; 40 equal-width bins. Dotted line: fixed threshold 0.4; colored lines: medians.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.95));save(fig,'three_methods_test_existence_score_distributions')
def thresholds(curve,summary):
    fig,axes=plt.subplots(2,3,figsize=(14.3,8.5));base=curve['E0'];handles=[]
    for i,(key,label,c,ls,marker) in enumerate(METHODS):
        data=curve[key];t=[r['threshold'] for r in data];kwargs=dict(color=c,ls=ls,marker=marker,markevery=10,ms=3.5,lw=1.7)
        h=axes[0,0].plot(t,[r['G-mIoU@1'] for r in data],label=f'{label} (mAP {summary[i]["mAP"]:.2f})',**kwargs)[0];handles.append(h)
        axes[0,1].plot(t,[r['G-mIoU@3'] for r in data],**kwargs);axes[0,2].plot(t,[r['G-mIoU@1']-b['G-mIoU@1'] for r,b in zip(data,base)],**kwargs)
        axes[1,0].plot([r['PosRecall'] for r in data],[r['TNR'] for r in data],**kwargs);axes[1,0].scatter(data[40]['PosRecall'],data[40]['TNR'],s=65,marker=marker,color=c,edgecolor='white',zorder=5)
        axes[1,1].plot(t,[r['Rej_F1'] for r in data],**kwargs)
        xx=np.arange(3)+(i-1)*.24;vv=[data[j]['G-mIoU@1'] for j in [40,50,60]];bars=axes[1,2].bar(xx,vv,width=.23,color=c,edgecolor='white');axes[1,2].bar_label(bars,fmt='%.2f',fontsize=8,padding=3,rotation=90)
    spec=[('(a) Threshold sensitivity','Presence threshold','G-mIoU@1 (%)'),('(b) Top-3 retrieval','Presence threshold','G-mIoU@3 (%)'),('(c) Relative to GMR','Presence threshold','Delta G-mIoU@1 (pp)'),('(d) Rejection / positive retention','Positive presence recall (%)','Negative recall / TNR (%)'),('(e) Negative-class F1','Presence threshold','Rejection F1 (%)'),('(f) Fixed operating points','Presence threshold','G-mIoU@1 (%)')]
    for ax,(title,xl,yl) in zip(axes.flat,spec):ax.set(title=title,xlabel=xl,ylabel=yl);ax.grid(alpha=.18)
    for ax in [axes[0,0],axes[0,1],axes[0,2],axes[1,1]]:ax.set_xlim(0,1);ax.axvline(.4,color='#64748b',ls=':',lw=.9)
    for ax in [axes[0,0],axes[0,1],axes[1,2]]:ax.set_ylim(0,70)
    axes[0,2].axhline(0,color='gray',lw=.8);axes[1,0].set(xlim=(0,100),ylim=(0,105));axes[1,1].set_ylim(0,100);axes[1,2].set_xticks(range(3),['0.4','0.5','0.6'])
    fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False,fontsize=10)
    fig.text(.5,.015,'Standard Test: 1,036 queries (544 positive / 492 negative); 101 fixed thresholds per model.\nLarge markers / dotted lines: threshold 0.4. Localization mAP uses positive queries without an existence hard gate.\nGAP: margin=0.6, weight=2, epoch4, selected by Val AUROC. This sweep does not choose thresholds or checkpoints.',ha='center',fontsize=9,color='#475569')
    fig.tight_layout(rect=(0,.1,1,.935));save(fig,'three_methods_fig7_threshold_sweep')
def overview():
    fig,ax=plt.subplots(figsize=(14,8));ax.set(xlim=(0,14),ylim=(0,8));ax.axis('off')
    def box(x,y,w,h,text,c='#eef2f6',fs=10):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12',fc=c,ec='#64748b',lw=1));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs)
    def arrow(a,b,label=None,style='-'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.2,color='#475569',linestyle=style))
        if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.14,label,fontsize=8,ha='center')
    ax.text(.15,7.65,'FlashVTG + GEQ + GAP: joint supervision and class-balanced evaluation',fontsize=15,fontweight='bold')
    ax.text(.15,7.2,'Training: original pretrained GMR initialization; all GMR + GEQ parameters jointly optimized',fontsize=11,color='#334155')
    box(.2,5.7,2.1,1.,'Offline video / text\nfeatures\n(encoders outside graph)',fs=9)
    box(3,5.65,3.05,1.1,'FlashVTG GMR\nvideo–text Transformer\n+ native prediction heads','#e2e8f0')
    box(7,6.3,2.6,.6,'Native localization output','#e2e8f0',9)
    box(7,5.15,2.6,.75,'Global Event Query\ntext / video readouts','#dbeafe',10)
    box(10.7,5.15,2.9,.75,r'$z\;\rightarrow\;s=\sigma(z)$'+'\nGEQ presence score','#dbeafe',11)
    arrow((2.45,6.2),(2.85,6.2));arrow((6.2,6.55),(6.85,6.6));arrow((6.2,5.9),(6.85,5.55));arrow((9.75,5.55),(10.55,5.55))
    box(.3,3.15,4.4,1.3,'Retain E1 supervision\nNative localization + native presence BCE\n+ GEQ per-query BCE','#e2e8f0',10)
    box(5.15,3.05,8.35,1.5,'Added GAP supervision (mixed batches only)\n'+r'$\mu_+=\mathrm{mean}(s\mid y=1),\quad\mu_-=\mathrm{mean}(s\mid y=0)$'+'\n'+r'$L_{\mathrm{GAP}}=[\max(0,\,m-(\mu_+-\mu_-))]^2$'+'\n'+r'$L=L_{\mathrm{native}}+L_{\mathrm{GEQ}}+\lambda L_{\mathrm{GAP}}$'+'     '+r'$m=0.6,\ \lambda=2$','#ffedd5',12)
    arrow((12.15,5.0),(12.15,4.7));arrow((4.4,5.5),(2.5,4.6));arrow((6.7,3.0),(4.5,5.45),style='--');ax.text(.3,2.55,'Joint end-to-end backpropagation',fontsize=10,color='#475569')
    box(.3,.55,6.25,1.65,'Inference\nGEQ alone supplies video-level presence score.\nKeep native localization, soft calibration and NMS.\nNo added gate, PV, boundary refiner or candidate fusion.','#eff6ff',10)
    box(7,.55,6.5,1.65,'Evaluation (positive / negative classes weighted equally)\n'+r'$D_+=\mathbb{E}[1-s\mid y=1],\quad D_-=\mathbb{E}[s\mid y=0]$'+'\n'+r'$D=(D_++D_-)/2,\quad S=100(1-D)$'+'\n'+r'$S=50(1+\mu_+-\mu_-)$'+'; also AUROC, AUPRC and localization','#ecfdf5',11)
    ax.text(.3,.12,'Single-class batches: graph-connected zero GAP; native + GEQ losses remain active. Selection: epoch4 Val AUROC; threshold fixed at 0.4.',fontsize=9,color='#475569')
    save(fig,'gap_supervision_and_evaluation')
if __name__=='__main__':main()
