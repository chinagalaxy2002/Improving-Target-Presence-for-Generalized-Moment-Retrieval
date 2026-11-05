"""Single repository-owned command line; each invocation loads one backbone."""
import argparse
import json
from pathlib import Path
from .config import load_config
from .backbones.registry import SPECS,get_spec


def main(argv=None):
    parser=argparse.ArgumentParser(prog='soccergmr')
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('backbones')
    for name in ('audit','train','test'):
        sub=commands.add_parser(name)
        sub.add_argument('--config',required=True)
        sub.add_argument('--output',required=True)
        if name=='audit':sub.add_argument('--paths-only',action='store_true',help='Skip array validation, but still hash all assets')
        else:sub.add_argument('--device',default='cuda:0')
        if name=='train':
            sub.add_argument('--stage',choices=['gmr','geq_full'],required=True)
            sub.add_argument('--baseline',help='Finalized GMR best.pt, required for GEQ Full')
            sub.add_argument('--resume',action='store_true',help='Restore last complete epoch, including optimizer and RNG')
        if name=='test':sub.add_argument('--checkpoint',required=True)
    args=parser.parse_args(argv)
    if args.command=='backbones':
        print(json.dumps({name:vars(spec) for name,spec in SPECS.items()},indent=2));return
    config=load_config(args.config)
    if args.command=='audit':
        from .data import audit_data
        result=audit_data(config,get_spec(config['backbone']),args.output,arrays=not args.paths_only)
        print(json.dumps({k:v for k,v in result.items() if k!='feature_sha256'},indent=2))
    elif args.command=='train':
        from .train import train
        print(json.dumps(train(config,args.output,args.stage,args.device,args.baseline,args.resume),indent=2))
    else:
        from .test import test
        result=test(config,args.checkpoint,args.output,args.device)
        print(json.dumps({split:value['brief'] for split,value in result.items()},indent=2))

if __name__=='__main__':main()
