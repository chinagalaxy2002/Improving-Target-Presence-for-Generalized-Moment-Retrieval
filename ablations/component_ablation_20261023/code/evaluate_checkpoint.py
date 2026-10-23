"""Read-only strict evaluation of the two published Val-selected checkpoints."""
import argparse
from pathlib import Path
from environment import ROOT,make_dataset,write_json,write_rows,digest
from api import load_selected,SHAS
from prediction import predictions
from evaluation import Evaluation
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=list(SHAS),required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--split',choices=['val','test'],default='test');p.add_argument('--checkpoint');p.add_argument('--output',required=True);a=p.parse_args()
 out=Path(a.output).resolve();assert not out.exists(),'Fresh output directory required';out.mkdir(parents=True)
 write_json(out/'FROZEN_PROTOCOL.json',dict(model=a.model,checkpoint_sha256=SHAS[a.model],split=a.split,data_sha256=digest(ROOT/'data'/f'{a.split}.jsonl'),threshold=.5,Test_threshold_optimization=False))
 model,opt,_=load_selected(a.model,a.device,checkpoint=a.checkpoint);before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};data=make_dataset(opt,a.split)
 with torch.inference_mode():pre,post=predictions(model,data,opt)
 assert all(torch.equal(before[k],v.detach().cpu()) for k,v in model.state_dict().items())
 report=Evaluation(pre,post,data.data).report(a.split,comprehensive=True);write_json(out/'metrics.json',report);write_rows(out/'predictions.jsonl',post);write_rows(out/'predictions_pre_nms.jsonl',pre);write_json(out/'COMPLETE.json',dict(rows=len(post),weights_unchanged=True))
if __name__=='__main__':main()
