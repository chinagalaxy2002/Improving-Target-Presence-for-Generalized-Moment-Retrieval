"""Synthetic end-to-end A/B/C/D verification, not a benchmark experiment."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from soccergmr.config import load_config
from soccergmr.train import train
from soccergmr.test import test
from soccergmr import checkpoint as ck

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42);splits={};features={}
    for split in ['train','val','test_standard','test_full']:
        namespace='full' if split=='test_full' else 'standard';root=out/'features'/namespace;features[namespace]=str(root)
        vid=split+'_video'
        for kind,width in [('clip',512),('slowfast',2304)]:
            (root/kind).mkdir(parents=True,exist_ok=True);np.savez(root/kind/(vid+'.npz'),features=rng.normal(size=(8,width)).astype('float32'))
        rows=[]
        for index,windows in enumerate([[[2,4]],[]]):
            qid=split+'_'+str(index);(root/'clip_text').mkdir(exist_ok=True);np.savez(root/'clip_text'/('qid'+qid+'.npz'),last_hidden_state=rng.normal(size=(4,512)).astype('float32'))
            rows.append(dict(qid=qid,vid=vid,duration=16,relevant_windows=windows,query='synthetic query'))
        label=out/(split+'.jsonl');label.write_text(''.join(json.dumps(r)+'\n' for r in rows));splits[split]=str(label)
    configs={}
    for name,w1,w2 in [('A',0.,0.),('B',0.,.1),('C',.1,0.),('D',.1,.1)]:
        raw=json.loads((ROOT/'configs/experiments/A.json').read_text());raw['data'].update(splits=splits,features=features,full_label='Synthetic verification only');raw['architecture']=str(ROOT/'configs/backbones/moment_detr.json');raw['protocol'].update(max_epochs=2,geq_max_epochs=2,workers=0,batch_size=2,eval_batch_size=2);raw['clip']={'gmr_weight':w1,'geq_weight':w2}
        path=out/(name+'.json');path.write_text(json.dumps(raw));configs[name]=load_config(path)
    for name in ['A','C']:train(configs[name],out/name/'gmr','gmr','cpu')
    for name in ['A','B','C','D']:
        parent=out/('A' if name in ['A','B'] else 'C')/'gmr/best.pt'
        train(configs[name],out/name/'geq_full','geq_full','cpu',parent)
        result=test(configs[name],out/name/'geq_full/best.pt',out/name/'test','cpu')
        assert all(v['stats']['num_total']==2 for v in result.values())
        saved=ck.load(out/name/'geq_full/last.pt');assert saved['baseline_sha256']==ck.load(out/name/'geq_full/best.pt')['baseline_sha256']
        if name in ['C','D']:
            initial=ck.load(parent)['model'];changed=[not torch.equal(v,saved['model'][k]) for k,v in initial.items() if k.startswith('clip_head.')]
            assert any(changed) if name=='D' else not any(changed)
    print('PASS: synthetic A/B/C/D train -> strict parent transfer -> both Tests; C head frozen, D updated. Not performance results.')
if __name__=='__main__':main()
