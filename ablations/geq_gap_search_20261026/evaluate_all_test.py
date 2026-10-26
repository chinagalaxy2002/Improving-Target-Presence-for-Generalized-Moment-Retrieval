#!/usr/bin/env python3
"""User-authorized full Test reporting; preserve the previous Val-only selection."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse,csv,json,subprocess,sys
from pathlib import Path
EXP=Path(__file__).resolve().parent
from report import PARENT,SRC,OLD,metrics
sys.path.insert(0,str(PARENT))
from evaluate import dump,digest

def worker(trial):
    p=EXP/'runs'/trial
    if (p/'TEST_COMPLETE.json').exists():return
    from search_train import PREVIOUS
    import torch
    torch.set_num_threads(2);adapter,model,_=PREVIOUS.construct();checkpoint=p/'epoch_004.ckpt';ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ckpt['epoch']==4;model.load_state_dict(ckpt['model'],strict=True)
    from evaluation import predict,evaluate
    from common import rows
    data=adapter.dataset('test');pred,diag=predict(adapter,model,data);evaluate(pred,data.data,p,'test',diag,rows(OLD/'results/flashvtg/gmr/test_predictions.jsonl'))
    dump(p/'TEST_COMPLETE.json',dict(epoch=4,full_test_reporting_requested=True,checkpoint_sha256=digest(checkpoint),val_selection_unchanged=True))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--trial');args=ap.parse_args()
    if args.trial:return worker(args.trial)
    plan=json.loads((EXP/'SEARCH.json').read_text());selection=EXP/'reports/SELECTION.json';selection_hash=digest(selection)
    out=EXP/'reports/all_test';out.mkdir(exist_ok=True)
    before={str(EXP/f'runs/{t}/val_predictions.jsonl'):digest(EXP/f'runs/{t}/val_predictions.jsonl') for t in plan['trials']}
    pending=[t for t in plan['trials'] if not (EXP/f'runs/{t}/TEST_COMPLETE.json').exists()]
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines();busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
    assert all(not any(u.strip() in l for l in busy) for u in uuids[:2]),'GPU has existing work; reserve free slots before launch'
    workers=[]
    for i,t in enumerate(pending):
        with (EXP/f'logs/test_all_{t}.log').open('w') as f:
            p=subprocess.Popen([sys.executable,str(EXP/'evaluate_all_test.py'),'--trial',t],cwd=EXP,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i%2)),stdout=f,stderr=subprocess.STDOUT)
        workers.append((t,p));print('START',t,p.pid,flush=True)
    for t,p in workers:
        assert p.wait()==0,f'{t} failed; inspect test log'
        print('DONE',t,flush=True)
    assert digest(selection)==selection_hash and all(digest(Path(p))==h for p,h in before.items())
    refs=[('E0',None,None,OLD/'results/flashvtg/gmr'),('E1',None,None,SRC/'runs/flashvtg/E1'),('GAP_reference',.5,1.,PARENT/'runs/GAP')]
    for t in plan['trials']:
        c=json.loads((EXP/f'configs/{t}.json').read_text())['added_loss'];refs.append((t,c['margin'],c['weight'],EXP/'runs'/t))
    table=[];full=[];audits={}
    for name,margin,weight,p in refs:
        if name in plan['trials']:
            d=json.loads((p/'test_diagnostics.json').read_text());assert d['independent_metric_verification']['passed'];assert digest(p/'epoch_004.ckpt')==json.loads((p/'TEST_COMPLETE.json').read_text())['checkpoint_sha256'];audits[name]=True
        table.append(dict(method=name,margin=margin,weight=weight,**metrics(p,'test')))
        for split in ['val','test']:
            score=metrics(p,split);official=json.loads((p/f'{split}_metrics.json').read_text())
            for tau in ['0.4','0.5','0.6']:
                brief=official[tau]['brief'];v=dict(method=name,margin=margin,weight=weight,split=split,threshold=tau,**score)
                for k in ['G-mIoU@1','G-mIoU@3','Pos-Recall','TNR']:
                    if k in brief:v[k]=brief[k]
                v['Rej-F1']=brief[f'Rej-F1@{tau}'];v.pop('Rej-F1@0.4',None)
                # Older E0 brief misses recall: derive it at the requested tau.
                from evaluate import rows,aligned
                gt=rows(SRC/f'data/{split}.jsonl');pred=aligned(p/f'{split}_predictions.jsonl',gt);y=[bool(g['relevant_windows']) for g in gt];gate=[x['pred_exist_score']>float(tau) for x in pred]
                v['Pos-Recall']=100*sum(a and b for a,b in zip(gate,y))/sum(y);v['TNR']=100*sum(not a and not b for a,b in zip(gate,y))/(len(y)-sum(y));full.append(v)
    for filename,data in [('test_summary.csv',table),('all_metrics.csv',full)]:
        with (out/filename).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['# 全部六组 GAP 超参数 Test 评估','', '按用户后续要求补测全部六组第4轮 checkpoint。预先按Val选中的 margin=0.6、weight=2 保持不变；此表用于完整报告，不根据Test重新选择。E0、E1与原GAP复用已有结果。','', '| Method | margin | λ | 正均分 | 负均分 | 均分差距 | S | AUROC | AUPRC | Pos-Recall | TNR | mAP | mR+@5 | G-mIoU@1 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append('| '+r['method']+' | '+('—' if r['margin'] is None else str(r['margin']))+' | '+('—' if r['weight'] is None else str(r['weight']))+' | '+' | '.join(f'{r[k]:.4f}' if k in ['mu_pos','mu_neg','mean_gap'] else f'{r[k]:.2f}' for k in ['mu_pos','mu_neg','mean_gap','S','AUROC','AUPRC','Pos-Recall','TNR','mAP','mR+@5','G-mIoU@1'])+' |')
    lines+=['','Test共1036条（544正、492负）。除均分外单位%，分类阈值0.4；定位指标独立于存在性hard gate。完整Val/Test与0.5/0.6敏感性指标见all_metrics.csv。每组官方评估及独立复算均通过。','', 'S=50*(1+mean_gap)，不把二者视作独立证据。单seed，不报告统计显著性或跨seed稳定性。旧reports/RESULTS.md与AUDIT.json保留为“选中模型首次Test”时的历史快照；本目录是后续全部Test补测报告。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    dump(out/'AUDIT.json',dict(passed=True,all_six_test_verified=audits,selection_unchanged=True,selection_sha256=selection_hash,validation_unchanged=True,all_checkpoints_epoch4=True))
    print((out/'RESULTS.md').read_text(),flush=True)
if __name__=='__main__':main()
