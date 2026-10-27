#!/usr/bin/env python3
"""Analysis-only Test threshold sweeps: real checkpoints and fixed epoch-4 windows."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from independent_metrics import gmiou
ROOT=Path(__file__).resolve().parent
DEST=ROOT/'figures/FlashVTG_E2_long_6_8';DEST.mkdir(parents=True,exist_ok=True)
def read(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
gt=read(ROOT/'data/test.jsonl');ids=[g['qid'] for g in gt];positive=np.array([bool(g['relevant_windows']) for g in gt]);assert len(gt)==1036 and positive.sum()==544
paths={4:ROOT/'runs/flashvtg/E2',6:ROOT/'runs/flashvtg/E2_long_6_8/epoch_006',8:ROOT/'runs/flashvtg/E2_long_6_8/epoch_008'}
pred={e:{p['qid']:p for p in read(d/'test_predictions.jsonl')} for e,d in paths.items()}
assert all(set(p)==set(ids) for p in pred.values())
tau=np.linspace(0,1,501)
scores={e:np.array([pred[e][i]['pred_exist_score'] for i in ids]) for e in paths}
local={}
for e in paths:
    local[e]=np.array([gmiou([dict(pred[e][g['qid']],pred_exist_score=1.)],[g],0.)['G-mIoU@1']/100 if g['relevant_windows'] else 0. for g in gt])
    for t in (.4,.5,.6):
        value=100*np.mean(np.where(scores[e]>t,local[e],~positive))
        official=json.loads((paths[e]/'test_metrics.json').read_text())[str(t)]['brief']['G-mIoU@1']
        assert abs(value-official)<=.011,(e,t,value,official)
styles={4:('#0072B2','-','o'),6:('#009E73','--','D'),8:('#D55E00','-.','^')}
plt.rcParams.update({'font.size':12,'axes.titlesize':15,'axes.labelsize':13,'legend.fontsize':10,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
all_records=[]
for fixed in (True,False):
    name='fixed_epoch4_localizer' if fixed else 'checkpoint_specific_localizers'
    curves={e:100*np.mean(np.where(scores[e][None,:]>tau[:,None],local[4 if fixed else e][None,:],(~positive)[None,:]),axis=1) for e in paths}
    oracle={e:100*np.mean(np.where(positive,local[4 if fixed else e],1.)) for e in paths}
    fig,axes=plt.subplots(1,2,figsize=(15,5.7))
    a,b=axes
    for e in paths:
        color,ls,marker=styles[e]
        a.plot(tau,curves[e],color=color,ls=ls,lw=2.4,marker=marker,markevery=50,ms=5,label=f'E2 epoch {e}')
        if e!=4:b.plot(tau,curves[e]-curves[4],color=color,ls=ls,lw=2.4,marker=marker,markevery=50,ms=5,label=f'Epoch {e} minus epoch 4')
    if fixed:
        a.axhline(oracle[4],color='#8A661C',ls='--',lw=2.2,label='Perfect existence oracle')
        a.fill_between(tau,curves[4],oracle[4],color='#E69F00',alpha=.07,label='Epoch-4 remaining oracle gap')
        b.plot(tau,oracle[4]-curves[4],color='#8A661C',ls=':',lw=2.3,label='Oracle minus epoch 4')
    else:
        for e in paths:
            a.axhline(oracle[e],color=styles[e][0],ls=':',lw=1.4,label=f'Epoch-{e} existence oracle ({oracle[e]:.2f})')
        b.plot(tau,oracle[4]-curves[4],color='#8A661C',ls=':',lw=2.3,label='Epoch-4 oracle minus epoch 4')
    a.set_title('(a) G-mIoU@1 versus existence threshold',loc='left',weight='bold')
    b.set_title('(b) Gain relative to epoch 4',loc='left',weight='bold')
    a.set_ylabel('G-mIoU@1 (%)');b.set_ylabel('Delta G-mIoU@1 (pp)');a.set_ylim(0,max(oracle.values())+6)
    b.axhline(0,color='#64748B',ls='--',lw=1)
    for ax in axes:
        for t in (.4,.5,.6):ax.axvline(t,color='#94A3B8',ls=':',lw=1)
        ax.set_xlim(0,1);ax.set_xticks(np.linspace(0,1,6));ax.set_xlabel('Existence threshold '+r'$\tau$');ax.grid(axis='y',ls=':',color='#CBD5E1');ax.spines[['top','right']].set_visible(False);ax.legend(loc='best',framealpha=.94)
    subtitle='Fixed epoch-4 temporal windows; only GEQ existence scores vary' if fixed else 'Each checkpoint retains its own temporal windows and GEQ existence scores'
    fig.suptitle('FlashVTG + GEQ + PV: 4 / 6 / 8 fine-tuning epochs\n'+subtitle,fontsize=15,y=.99)
    fig.text(.5,.025,'Official Test: N=1,036 (544 positive, 492 negative). Decision: score > threshold.\nOracle uses Test labels for analysis only; no threshold or checkpoint is selected from this sweep.',ha='center',fontsize=10,color='#475569')
    fig.subplots_adjust(left=.065,right=.99,bottom=.22,top=.79,wspace=.17)
    for ext in ('pdf','svg','png'):fig.savefig(DEST/f'{name}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    for j,t in enumerate(tau):
        for e in paths:all_records.append(dict(setting=name,epoch=e,threshold=float(t),gmiou1=float(curves[e][j]),delta_vs_epoch4=float(curves[e][j]-curves[4][j]),oracle_gmiou1=float(oracle[e])))
with (DEST/'curves.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(all_records[0]));w.writeheader();w.writerows(all_records)
manifest=dict(split='test',n=1036,positive=544,negative=492,grid='0..1 step 0.002, no smoothing',decision='score > threshold',official_checks='each actual checkpoint matches official G-mIoU@1 at 0.4, 0.5, 0.6 within 0.011 pp',oracle='perfect GT presence; retains positive temporal localization errors; analysis only',fixed_localizer='exact postprocessed epoch-4 windows and ordering; epoch-6/8 scores substituted; no repeated score-dependent postprocessing',sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'data/test.jsonl',*[d/'test_predictions.jsonl' for d in paths.values()],Path(__file__).resolve()]})
(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(DEST/'README.md').write_text('# FlashVTG E2 长微调阈值扫描\n\n- `fixed_epoch4_localizer`：与参考图对应，固定第4轮官方后处理后的时间窗及排序，仅替换第4、6、8轮 GEQ 分数。各曲线共享同一个存在性 Oracle。这是受控分析，不是第6/8轮完整模型的官方结果。\n- `checkpoint_specific_localizers`：各轮使用各自真实官方预测，对应完整模型；各自存在性 Oracle 保留各自定位误差。右图 Oracle 差值以第4轮为基准。\n- Test N=1036，544正/492负；阈值0至1、步长0.002，不平滑。正判定严格为 score > threshold，因此端点可能跳变。\n- Oracle 仅替换存在性判定，不修正正查询的定位，不是完美定位 Oracle。使用 Test 标签，仅供事后诊断，不能据此选阈值或 checkpoint。\n- `curves.csv` 保存所有点，`manifest.json` 保存来源哈希及复算检查。\n')
print(DEST)
print('oracle', {e:100*np.mean(np.where(positive,local[e],1.)) for e in paths})
