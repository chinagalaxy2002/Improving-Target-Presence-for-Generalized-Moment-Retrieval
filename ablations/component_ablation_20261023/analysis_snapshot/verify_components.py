"""Reconstruct actual trained variants; verify switches and frozen tensors independently."""
import argparse,importlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from run import seed,dump,sha
from variants import NativeVariant

def group(name):
    if 'lora_' in name:return 'lora'
    if 'sal_proj' in name:return 'saliency'
    if any(x in name for x in ['moment_cross_attn','moment_match_mlp','local_head','fusion_gate']):return 'fusion'
    if 'refiner.' in name:return 'refiner'
    if 'extractor.' in name:return 'extractor'
    if '.base.' in name and '.exist_head.' not in name:return 'frozen_backbone'
    return 'global_event_query'

def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',required=True);a=p.parse_args();torch.set_num_threads(2)
    flash=a.backbone=='flashvtg'
    if flash:
        sys.path.insert(0,str(ROOT/'code'));import flash_backend as backend
    else:
        sys.path.insert(0,str(ROOT/'backends'/a.backbone));backend=importlib.import_module('backend')
    shared=None;checked=[]
    for row in range(1,8):
        directory=ROOT/'runs'/a.backbone/f'row_{row:02d}'
        if not (directory/'COMPLETE.json').exists():continue
        config=json.loads((directory/'CONFIG.json').read_text());job=config['job'];seed()
        if flash:model,init=backend.initialize('cpu',job,row)
        else:
            inner,init=backend.initialize('cpu',job,snapshot=config['initialization']['snapshot']);model=NativeVariant(inner,row)
        assert init['sha256']==config['initialization']['sha256']
        initial={n:t.detach().clone() for n,t in model.state_dict().items()}
        if row>1:
            if shared is not None:
                for n in initial.keys() & shared.keys():assert torch.equal(initial[n],shared[n]),(a.backbone,row,n,'initialization differs')
                for n in initial.keys()-shared.keys():shared[n]=initial[n]
            else:shared=initial
            state=torch.load(directory/'epoch_004.ckpt',map_location='cpu',weights_only=False)
            assert state['epoch']==4
            model.load_state_dict(state['model'],strict=True)
        actual=model.state_dict();frozen_checked=0;changed={};trainable={}
        for name,param in model.named_parameters():
            g=group(name)
            if param.requires_grad:
                trainable[g]=trainable.get(g,0)+param.numel()
                if not torch.equal(initial[name],actual[name]):changed[g]=changed.get(g,0)+param.numel()
            else:
                assert torch.equal(initial[name],actual[name]),(row,name,'frozen tensor changed');frozen_checked+=param.numel()
        for name,buffer in model.named_buffers():
            if name in initial:assert torch.equal(initial[name],actual[name]),(row,name,'persistent buffer changed')
        assert sum(trainable.values())==config['trainable']
        if row>1:assert changed.get('global_event_query',0)>0
        if row>=3:assert changed.get('lora',0)>0
        if row>=4:
            assert changed.get('refiner',0)>0 and changed.get('extractor',0)>0
        if row>=6:assert changed.get('saliency',0)>0
        if row==7:assert changed.get('fusion',0)>0
        if row<3:assert not trainable.get('lora',0)
        if row<6:assert not trainable.get('saliency',0)
        if row<7:assert not trainable.get('fusion',0)
        if row==1:assert not trainable
        if not (flash and row==1):
            for module in model.modules():
                if hasattr(module,'lora_A') and hasattr(module,'scaling'):
                    assert (module.scaling>0)==(row>=3)
            extractor=model.base.exist_head.extractor if flash else model.inner.extractor
            assert extractor.pooling_type==('uniform_box' if row==4 else 'gaussian')
            if row>=4:
                generator=torch.Generator().manual_seed(721)
                toy=torch.randn(1,10,256,generator=generator);valid=torch.ones(1,10,dtype=torch.bool)
                spans=torch.tensor([[[4.,8.,.5]]]);outside=toy.clone();outside[:,4]+=torch.arange(256)*.1
                with torch.no_grad():
                    original=extractor(toy,valid,spans)[0];modified=extractor(outside,valid,spans)[0]
                if row==4:assert torch.equal(original,modified),'hard RoI includes an outside frame center'
                else:assert not torch.allclose(original,modified),'Gaussian RoI fails to weight nearby outside context'
                valid[:,-1]=False;padded=toy.clone();padded[:,-1]+=torch.arange(256)*.1
                with torch.no_grad():assert torch.equal(extractor(toy,valid,spans)[0],extractor(padded,valid,spans)[0]),'masked padding changes pooled features'
            if row in [4,5]:
                refiner=model.base.exist_head.refiner if flash else model.inner.refiner
                with torch.no_grad():assert torch.count_nonzero(refiner.sal_proj(torch.ones(1,1,6)))==0
        effective=dict(global_event_query=row>=2,lora=row>=3,boundary_refiner=row>=4,pooling='hard_rectangle' if row==4 else ('gaussian' if row>=5 else None),saliency_6d=row>=6,candidate_presence_fusion=row==7,baseline_native_presence=row==1)
        result=dict(passed=True,scope='formal' if row<=6 else 'additional_diagnostic',row=row,backbone=a.backbone,effective=effective,effective_training=dict(adaptation_epochs=0 if row==1 else 4,checkpoint_selection='fixed epoch 4; frozen baseline row 1',evaluation='Val and Test after training only',training_queries=4138,validation_queries=465,test_queries=1036),trainable_parameters_by_group=trainable,changed_parameters_by_group=changed,frozen_parameters_exactly_preserved=frozen_checked,shared_initialization_equal=True,initialization_sha256=init['sha256'])
        dump(directory/'EFFECTIVE_CONFIG.json',result);checked.append(result)
        del model,initial,actual
    dump(ROOT/'reports'/f'{a.backbone}_COMPONENT_CHECK.json',dict(checked_formal_rows=sum(c['row']<=6 for c in checked),checked_additional_diagnostic_rows=sum(c['row']==7 for c in checked),checks=checked))
    print(a.backbone,'verified formal rows',sum(c['row']<=6 for c in checked),'and additional diagnostic rows',sum(c['row']==7 for c in checked))
if __name__=='__main__':main()
