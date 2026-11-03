#!/usr/bin/env python3
"""Portable A/B/C/D runner; Standard Val fixes lambda before any Test."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='runs/reproduction');p.add_argument('--device',default='cuda:0');p.add_argument('--all-gmr-candidates',action='store_true');args=p.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    def command(values):subprocess.run([sys.executable,*values],cwd=ROOT,env=env,check=True)
    def config(name):return str(ROOT/'configs/experiments'/(name+'.json'))
    def train(name,stage,parent=None):
        d=out/'formal'/name/stage
        cmd=['-m','soccergmr','train','--config',config(name),'--stage',stage,'--output',str(d),'--device',args.device]
        if parent:cmd+=['--baseline',str(parent)]
        if (d/'last.pt').exists():cmd+=['--resume']
        command(cmd);return d/'best.pt'
    baseline=train('A','gmr');a=train('A','geq_full',baseline);candidates=[]
    for name,w in [('B_01',.1),('B_03',.3),('B_10',1.)]:
        best=train(name,'geq_full',baseline);m=json.loads((best.parent/'complete.json').read_text())
        candidates.append(dict(name=name,weight=w,val_g1=m['best_score'],checkpoint_sha256=m['best_sha256']))
    chosen=max(candidates,key=lambda c:(c['val_g1'],-c['weight']));selection=dict(chosen=chosen,candidates=candidates,test_used=False)
    marker=out/'selection.json'
    if marker.exists() and json.loads(marker.read_text())!=selection:raise RuntimeError('Previously fixed lambda selection differs')
    if not marker.exists():marker.write_text(json.dumps(selection,indent=2)+'\n')
    suffix=chosen['name'][2:];c,d='C_'+suffix,'D_'+suffix
    clipped=train(c,'gmr')
    if args.all_gmr_candidates:
        for n in ['C_01','C_03','C_10']:
            if n!=c:train(n,'gmr')
    cbest=train(c,'geq_full',clipped);dbest=train(d,'geq_full',clipped)
    for name,stage in [('A','gmr'),('A','geq_full'),(chosen['name'],'geq_full'),(c,'gmr'),(c,'geq_full'),(d,'geq_full')]:
        folder=out/'formal'/name
        command(['-m','soccergmr','test','--config',config(name),'--checkpoint',str(folder/stage/'best.pt'),'--output',str(folder/(stage+'_test')),'--device',args.device])
    for name in [chosen['name'],c,d]:
        command([str(ROOT/'scripts/diagnose_clipseg.py'),'--config',config(name),'--checkpoint',str(out/'formal'/name/'geq_full/best.pt'),'--output',str(out/'formal'/name/'diagnostics')])
    rows=[]
    for group,name in [('A','A'),('B',chosen['name']),('C',c),('D',d)]:
        for split in ['test_standard','test_full']:
            m=json.loads((out/'formal'/name/'geq_full_test'/split/'metrics.json').read_text())
            rows.append(dict(group=group,split=split,mAP=m['brief']['mAP'],G1=sum(m['gmiou'][str(t)]['G-mIoU@1'] for t in [.4,.5,.6])/3,G3=sum(m['gmiou'][str(t)]['G-mIoU@3'] for t in [.4,.5,.6])/3))
    (out/'summary.json').write_text(json.dumps(dict(selection=selection,results=rows),indent=2)+'\n')
    print('Complete:',out/'summary.json')
if __name__=='__main__':main()
