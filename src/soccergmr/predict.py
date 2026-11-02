import torch
from torch.utils.data import DataLoader
from .data import collate,move
from .backbones.registry import forward


def export(outputs,metas):
    foreground=outputs['pred_logits'].softmax(-1)[...,0]
    if 'iou_scores' in outputs:foreground=foreground*outputs['iou_scores'].squeeze(-1).sigmoid()
    spans=outputs['pred_spans']
    logit=outputs.get('pred_exist_logits',outputs.get('pred_exist_logit')).reshape(-1)
    if not all(torch.isfinite(x).all() for x in (foreground,spans,logit)):
        raise FloatingPointError('Nonfinite prediction')
    result=[]
    for index,meta in enumerate(metas):
        boundaries=torch.stack((spans[index,:,0]-spans[index,:,1]/2,
                                spans[index,:,0]+spans[index,:,1]/2),-1)
        boundaries=boundaries.clamp(0,1)*meta['duration']
        scores=foreground[index]
        ordering=torch.argsort(scores,descending=True,stable=True)[:10]
        windows=[[round(float(boundaries[j,0]),4),round(float(boundaries[j,1]),4),float(scores[j])] for j in ordering]
        result.append(dict(qid=meta['qid'],vid=meta['vid'],pred_relevant_windows=windows,
                           pred_exist_score=float(logit[index].sigmoid())))
    return result


@torch.no_grad()
def predict(model,dataset,config,device):
    model.eval();rows=[]
    loader=DataLoader(dataset,batch_size=config['protocol']['eval_batch_size'],
        num_workers=config['protocol']['workers'],collate_fn=collate,shuffle=False)
    for metas,inputs,_ in loader:
        outputs=forward(model,config['backbone'],move(inputs,device),metas=metas)
        rows.extend(export(outputs,metas))
    return rows
