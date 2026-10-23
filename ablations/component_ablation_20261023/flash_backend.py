"""FlashVTG implementation using the frozen reference model and native export."""
import os,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
provenance=json.loads((REPO/'provenance/EXTERNAL_CHECKPOINTS.json').read_text())
os.environ['GMR_E0_CHECKPOINT']=provenance['E0']['path']
os.environ['GMR_ASSETS_DIR']='/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval/Soccergmr'
from environment import options,make_dataset,digest,E0_CHECKPOINTS
from runtime import build,training_mode
from model import LoRALinear
from training.flash_vtg_gmr.dataset import start_end_collate,prepare_batch_inputs
opt=None

def initialize(device,job,row):
    global opt
    model,opt,checks=build(job,device)
    head=model.base.exist_head
    use_lora=row>=3;refine=row>=4
    model.use_boundary_refine=refine;head.use_boundary_refine=refine
    source='fused' if row==7 else ('global_refine' if refine else 'global_only')
    model.job.update(presence_source=source,use_boundary_refine=refine,use_lora=use_lora)
    head.job.update(model.job);head.extractor.pooling_type='uniform_box' if row==4 else 'gaussian'
    for module in model.modules():
        if isinstance(module,LoRALinear) and not use_lora:module.scaling=0.
    for name,p in model.named_parameters():
        active=p.requires_grad
        if '.lora_' in name:active=use_lora
        if '.exist_head.extractor.' in name or '.exist_head.refiner.' in name:active=refine
        if '.sal_proj.' in name:active=row>=6
        if any('.exist_head.'+key+'.' in name for key in ['moment_cross_attn','moment_match_mlp','local_head','fusion_gate']):active=row==7
        p.requires_grad_(active)
    if row in [4,5]:head.refiner.sal_proj.register_forward_hook(lambda m,a,o:torch.zeros_like(o))
    if row==1:
        from api import load_selected
        model,opt,_=load_selected('E0',device)
        model.requires_grad_(False)
    else:
        # Reference policy: frozen base dropout off, trainable head dropout on.
        import types
        def train(self,mode=True):
            torch.nn.Module.train(self,mode)
            if mode:training_mode_frozen(self)
            return self
        def training_mode_frozen(self):
            for m in self.base.modules():
                if isinstance(m,torch.nn.Dropout) and not any(p.requires_grad for p in m.parameters(recurse=False)):m.eval()
        model.train=types.MethodType(train,model)
    return model,dict(verified_gmr_baseline=True,sha256=digest(E0_CHECKPOINTS[2026]),snapshot=str(E0_CHECKPOINTS[2026]),baseline_config=vars(opt),checks=checks)

def get_dataset(split):return make_dataset(opt,split)
def collate(items,device):
    metas,batch=start_end_collate(items);inputs,targets=prepare_batch_inputs(batch,device)
    targets.update(label=metas,fps=torch.full((len(metas),),1/opt.clip_length,device=device))
    inputs['targets']=targets
    return metas,inputs
