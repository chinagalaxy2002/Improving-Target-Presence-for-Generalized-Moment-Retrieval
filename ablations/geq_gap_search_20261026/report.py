#!/usr/bin/env python3
"""Select by Val only, evaluate one selected model, report every Val trial."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import csv,json,sys
from pathlib import Path
EXP=Path(__file__).resolve().parent;PARENT=EXP.parent/'geq_mean_separation_20261026';SRC=EXP.parent/'end_to_end_geq_pv_20261026';OLD=EXP.parent/'global_hard_roi_20261023'
sys.path.insert(0,str(PARENT))
from evaluate import rows,aligned,measure,dump,digest

def metrics(path,split):
    gt=rows(SRC/f'data/{split}.jsonl');pred=aligned(path/f'{split}_predictions.jsonl',gt);values=measure(pred,gt);official=json.loads((path/f'{split}_metrics.json').read_text())['0.4']['brief']
    y=[bool(g['relevant_windows']) for g in gt];gate=[p['pred_exist_score']>.4 for p in pred]
    recall=100*sum(a and b for a,b in zip(gate,y))/sum(y);tnr=100*sum(not a and not b for a,b in zip(gate,y))/(len(y)-sum(y))
    return dict(**{k:v for k,v in values.items() if k!='distribution'},**{k:official[k] for k in ['mAP','mR@1','mR@5','mR+@5','G-mIoU@1','G-mIoU@3','Rej-F1@0.4']},**{'Pos-Recall':recall,'TNR':tnr})

def main():
    report=EXP/'reports';report.mkdir(exist_ok=True);plan=json.loads((EXP/'SEARCH.json').read_text());records=[];paths={}
    for name in plan['trials']+['GAP_reference']:
        path=PARENT/'runs/GAP' if name=='GAP_reference' else EXP/'runs'/name
        if name!='GAP_reference':assert (path/'VAL_COMPLETE.json').exists()
        diag=json.loads((path/'val_diagnostics.json').read_text());assert diag['independent_metric_verification']['passed']
        cfg=json.loads((PARENT/'configs/GAP.json' if name=='GAP_reference' else EXP/f'configs/{name}.json').read_text())
        records.append(dict(trial=name,margin=cfg['added_loss']['margin'],weight=cfg['added_loss']['weight'],**metrics(path,'val')));paths[name]=path
    winner=sorted(records,key=lambda r:(-r['AUROC'],-r['AUPRC'],-r['S'],-r['mAP'],r['trial']))[0]
    selection=dict(status='selection_locked_before_test',rule=plan['selection'],winner=winner['trial'],validation=winner,val_prediction_sha256={n:digest(p/'val_predictions.jsonl') for n,p in paths.items()},epoch=4,candidates=plan['trials']+['GAP_reference'])
    if (report/'SELECTION.json').exists():
        assert json.loads((report/'SELECTION.json').read_text())==selection,'Locked selection changed'
    else:dump(report/'SELECTION.json',selection)
    with (report/'val_search.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    # Test is accessed only after selection has been saved.
    selected=paths[winner['trial']]
    if winner['trial']!='GAP_reference' and not (selected/'TEST_COMPLETE.json').exists():
        sys.path.insert(0,str(EXP));from search_train import PREVIOUS
        import torch
        torch.set_num_threads(2);adapter,model,_=PREVIOUS.construct();checkpoint=selected/'epoch_004.ckpt';state=torch.load(checkpoint,map_location='cpu',weights_only=False);assert state['epoch']==4;model.load_state_dict(state['model'],strict=True)
        from evaluation import predict,evaluate
        test=adapter.dataset('test');pred,diag=predict(adapter,model,test);evaluate(pred,test.data,selected,'test',diag,rows(OLD/'results/flashvtg/gmr/test_predictions.jsonl'))
        dump(selected/'TEST_COMPLETE.json',dict(epoch=4,selection_locked=True,checkpoint_sha256=digest(checkpoint)))
    if winner['trial']!='GAP_reference':
        assert json.loads((selected/'test_diagnostics.json').read_text())['independent_metric_verification']['passed']
        assert digest(selected/'epoch_004.ckpt')==json.loads((selected/'TEST_COMPLETE.json').read_text())['checkpoint_sha256']
    comparison=[];cp={}
    for name,p in [('E0',OLD/'results/flashvtg/gmr'),('E1',SRC/'runs/flashvtg/E1'),('GAP_reference',PARENT/'runs/GAP')]+([] if winner['trial']=='GAP_reference' else [(winner['trial'],selected)]):
        comparison.append(dict(method=name,**metrics(p,'test')));cp[name]=p
    for r in comparison:
        base=comparison[0]
        for k in ['S','AUROC','AUPRC','mean_gap','mAP','mR+@5']:r['delta_'+k+'_vs_E0']=r[k]-base[k]
    with (report/'test_selected.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
    figure(records,comparison,cp,winner['trial'])
    lines=['# GAP 六组参数搜索','', '6组独立原始GMR初始化，各4轮，seed2026。原GAP仅复用作参考。全部配置仅评估Val；按预先固定规则选择后，仅评估所选模型Test。没有对各配置搜索Test、阈值或epoch。','',f"选择规则：{plan['selection']}。选中：**{winner['trial']}**。",'', '| Trial | margin | weight | Val AUROC | Val AUPRC | Val S | Val mean gap | Val mAP |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:lines.append('| '+r['trial']+' | '+' | '.join(f'{r[k]:.4f}' for k in ['margin','weight','AUROC','AUPRC','S','mean_gap','mAP'])+' |')
    lines+=['','## 选定后 Test 对比','','| Method | S | mean gap | AUROC | AUPRC | Pos-Recall | TNR | mAP | mR+@5 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in comparison:lines.append('| '+r['method']+' | '+' | '.join(f'{r[k]:.4f}' if k=='mean_gap' else f'{r[k]:.2f}' for k in ['S','mean_gap','AUROC','AUPRC','Pos-Recall','TNR','mAP','mR+@5'])+' |')
    lines+=['','主分类阈值0.4；0.5/0.6见原始官方评估文件。S=50*(1+mean_gap)，两者不是独立证据。正负端点误差分别保存在CSV。单seed，搜索会对Val产生选择偏差，不声称统计显著性。原参考GAP/E0/E1的Test已在前次研究中观察，当前新配置仍严格仅按Val选择；不把此过程声称为全研究从未查看Test。','', '搜索图为Val热图；最终Test图只含对照和选中配置。非选中配置未生成Test结果。']
    (report/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    assert all(digest(paths[n]/'val_predictions.jsonl')==h for n,h in selection['val_prediction_sha256'].items())
    for name in plan['trials']:
        if name!=winner['trial']:assert not (paths[name]/'test_metrics.json').exists()
    dump(report/'AUDIT.json',dict(passed=True,all_six_val_independent_pass=True,selection_before_new_test=True,nonselected_no_test=True,val_sources_unchanged=True,winner=winner['trial']))
    dump(EXP/'STATUS.json',dict(status='complete',winner=winner['trial'],trials=plan['trials'],selection='Val only',test_evaluated_only_selected=True))

def figure(records,comparison,paths,winner):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    root=EXP/'figures';root.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        for ext in ['pdf','svg','png']:fig.savefig(root/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.8))
    for ax,key in zip(axes,['AUROC','S','mAP']):
        a=np.array([[next(r[key] for r in records if r['margin']==m and r['weight']==w) for w in [.5,1.,2.]] for m in [.4,.6]])
        im=ax.imshow(a,cmap='viridis',aspect='auto');fig.colorbar(im,ax=ax,fraction=.046)
        for i in range(2):
            for j in range(3):ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',color='white',bbox=dict(facecolor='black',alpha=.25,edgecolor='none'))
        ax.set(xticks=range(3),xticklabels=['0.5','1','2'],yticks=range(2),yticklabels=['0.4','0.6'],xlabel='GAP weight',ylabel='Margin',title='Val '+key)
    fig.suptitle('FlashVTG GAP: fixed 4-epoch validation search');fig.tight_layout();save(fig,'val_search')
    fig,axes=plt.subplots(2,4,figsize=(13,7));names=[r['method'] for r in comparison];colors=['#666666','#0072B2','#D55E00','#009E73'][:len(names)]
    for ax,key in zip(axes.flat,['S','AUROC','AUPRC','Pos-Recall','TNR','mAP','mR+@5','G-mIoU@1']):
        vals=[r[key] for r in comparison];ax.bar(range(len(vals)),vals,color=colors)
        for i,v in enumerate(vals):ax.text(i,v+1,f'{v:.2f}',ha='center',fontsize=8)
        ax.set(xticks=range(len(names)),xticklabels=[n.replace('GAP_reference','GAP ref') for n in names],ylim=(0,100),title=key,ylabel='Percent');ax.tick_params(axis='x',labelsize=7,rotation=15)
    fig.suptitle('Test: Val-selected model and references (threshold 0.4)');fig.tight_layout();save(fig,'test_selected_comparison')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));styles=[':', '--','-','-.'];gt=rows(SRC/'data/test.jsonl');y=np.array([bool(g['relevant_windows']) for g in gt])
    for (name,p),color,style in zip(paths.items(),colors,styles):
        pred=aligned(p/'test_predictions.jsonl',gt);s=np.array([x['pred_exist_score'] for x in pred])
        for ax,mask in zip(axes,[y,~y]):
            den,bins=np.histogram(s[mask],bins=np.linspace(0,1,41),density=True);ax.stairs(den,bins,color=color,ls=style,label=name)
    for ax,title in zip(axes,['Positive queries','Negative queries']):ax.axvline(.4,color='gray',ls=':');ax.set(xlim=(0,1),xlabel='Existence score',ylabel='Density',title=title);ax.legend(fontsize=8)
    fig.tight_layout();save(fig,'test_selected_distributions')
if __name__=='__main__':main()
