"""Behavioral checks for proposal-free presence and the reference fusion equation."""
import argparse,importlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from run import seed,dump
from variants import NativeVariant

def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',required=True);a=p.parse_args();torch.set_num_threads(1)
    flash=a.backbone=='flashvtg'
    if flash:sys.path.insert(0,str(ROOT/'code'));import flash_backend as backend
    else:sys.path.insert(0,str(ROOT/'backends'/a.backbone));backend=importlib.import_module('backend')
    checks=[]
    for row in [6,7]:
        directory=ROOT/'runs'/a.backbone/f'row_{row:02d}';assert (directory/'COMPLETE.json').exists()
        config=json.loads((directory/'CONFIG.json').read_text());seed()
        if flash:model,_=backend.initialize('cpu',config['job'],row)
        else:
            inner,_=backend.initialize('cpu',config['job'],snapshot=config['initialization']['snapshot']);model=NativeVariant(inner,row)
        model.load_state_dict(torch.load(directory/'epoch_004.ckpt',map_location='cpu',weights_only=False)['model'],strict=True);model.eval()
        data=backend.get_dataset('train');idx=next(i for i,m in enumerate(data.data) if m['relevant_windows']);meta,inputs=backend.collate([data[idx]],'cpu')
        head=model.base.exist_head if flash else next(getattr(model.inner,n) for n in ['event_query','global_query','global_event_query'] if hasattr(model.inner,n))
        extractor=head.extractor if flash else model.inner.extractor
        captured={}
        def save_head(m,args):
            if not flash:captured['head']=args
        def save_extractor(m,args,kwargs):captured['roi']=args if args else (kwargs['final_video'],kwargs['video_mask'],kwargs['candidate_spans'])
        h=head.register_forward_pre_hook(save_head);e=extractor.register_forward_pre_hook(save_extractor,with_kwargs=True)
        with torch.no_grad():expected=model(**inputs)['pred_exist_logits'].clone()
        reference_error=None
        if not flash and row==7:
            f=model.fusion
            with torch.no_grad():f.token.copy_(head.token)
            for name in ['text_read','video_reads','global_head']:getattr(f,name).load_state_dict(getattr(head,name).state_dict(),strict=True)
            f.extractor.load_state_dict(extractor.state_dict(),strict=True);f.job['presence_source']='fused';f.eval()
            text,tv,videos,vv=captured['head'];final,valid,candidates=captured['roi']
            f.context=dict(text=text,text_valid=tv,video=videos,video_valid=vv,candidate_spans=candidates,final_video_units=final,final_video_valid=valid)
            with torch.no_grad():actual=f(None).squeeze(-1)
            f.context=None;reference_error=float((expected-actual).abs().max());assert torch.allclose(expected,actual,rtol=1e-6,atol=1e-6),(a.backbone,reference_error)
        def perturb(m,args,kwargs):
            if args:final,valid,spans=args
            else:final,valid,spans=kwargs['final_video'],kwargs['video_mask'],kwargs['candidate_spans']
            spans=spans.clone();spans[...,0]=(spans[...,0]+30)%100;spans[...,1]=spans[...,0]+10;spans[...,2]=1-spans[...,2]
            if args:return (final,valid,spans),kwargs
            return args,dict(kwargs,candidate_spans=spans)
        # Identical instrumentation and single-thread CPU execution for both arms.
        # A no-op hook controls for hook/dispatch changes before perturbing candidates.
        noop=extractor.register_forward_pre_hook(lambda m,args,kwargs:(args,kwargs),with_kwargs=True)
        noop_differences=[]
        with torch.no_grad():
            for repeat in range(3):
                control=model(**inputs)['pred_exist_logits'].clone()
                noop_differences.append(float((expected-control).abs().max()))
                assert torch.equal(expected,control),'repeated no-op forward changed the presence output'
        noop.remove()
        handle=extractor.register_forward_pre_hook(perturb,with_kwargs=True)
        perturb_differences=[]
        with torch.no_grad():
            for repeat in range(3):
                perturbed=model(**inputs)['pred_exist_logits'].clone()
                perturb_differences.append(float((expected-perturbed).abs().max()))
                if row==6:assert torch.equal(expected,perturbed),'minusFusion presence depends on candidate evidence'
        handle.remove();h.remove();e.remove();difference=max(perturb_differences)
        print(json.dumps(dict(backbone=a.backbone,row=row,expected=expected.tolist(),perturbed=perturbed.tolist(),difference=difference)),flush=True)
        if row==6:assert torch.equal(expected,perturbed),'minusFusion presence depends on candidate evidence'
        if row==7:assert difference>1e-7,'fusion control does not respond to candidate evidence'
        checks.append(dict(scope='formal' if row==6 else 'additional_diagnostic',row=row,candidate_perturbation_logit_difference=difference,reference_equation_max_error=reference_error,cpu_threads=1,noop_logit_differences=noop_differences,perturbation_logit_differences=perturb_differences,passed=True))
    dump(ROOT/'reports'/f'{a.backbone}_FUSION_BEHAVIOR.json',dict(backbone=a.backbone,checks=checks,passed=True))
    print(a.backbone,'proposal-free and fusion checks passed')
if __name__=='__main__':main()
