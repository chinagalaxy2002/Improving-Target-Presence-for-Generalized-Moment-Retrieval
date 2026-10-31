"""Training-only temporal auxiliary routes for the native Moment-DETR."""
import copy
import torch
from torch import nn
from models.moment_detr_gmr.utils.span_utils import span_cxw_to_xx

VARIANTS = ('baseline', 'shared', 'sepffn', 'instruction', 'multi', 'adaptive')

class MultiRoute(nn.Module):
    def __init__(self, base, variant):
        super().__init__()
        self.base, self.variant = base, variant
        self.routes = ([] if variant == 'baseline' else ['shared'] if variant == 'shared'
                       else ['sepffn'] if variant == 'sepffn' else ['instruction'] if variant == 'instruction'
                       else ['sepffn', 'instruction'])
        self.queries = nn.ModuleDict({r: nn.Embedding(24, 256) for r in self.routes})
        self.ffns = nn.ModuleList()
        if 'sepffn' in self.routes:
            for layer in base.transformer.decoder.layers:
                self.ffns.append(nn.ModuleDict({k: copy.deepcopy(getattr(layer, k)) for k in ('linear1', 'linear2', 'norm3')}))
        self.instructions = nn.Parameter(torch.randn(4, 256) * .02) if 'instruction' in self.routes else None
        self._route_outputs = {}
        base.transformer.register_forward_hook(self._after_transformer)

    def _after_transformer(self, module, inputs, outputs):
        self._route_outputs = {}
        if not self.training:
            return
        src, mask, _, pos = inputs
        memory = outputs[1].transpose(0, 1)
        pos = pos.transpose(0, 1)
        for route in self.routes:
            qp = self.queries[route].weight[:, None].expand(-1, src.shape[0], -1)
            tgt = torch.zeros_like(qp)
            states = []
            for i, layer in enumerate(module.decoder.layers):
                if layer.normalize_before:
                    raise ValueError('Only post-norm decoder supported')
                q = tgt + qp
                value = tgt
                if route == 'instruction':
                    ins = self.instructions[:, None].expand(-1, src.shape[0], -1)
                    q = torch.cat((ins, q))
                    value = torch.cat((ins, value))
                att = layer.self_attn(q, q, value=value)[0][-24:]
                tgt = layer.norm1(tgt + layer.dropout1(att))
                att = layer.multihead_attn(tgt + qp, memory + pos, memory, key_padding_mask=mask)[0]
                tgt = layer.norm2(tgt + layer.dropout2(att))
                ffn = self.ffns[i] if route == 'sepffn' else None
                linear1 = ffn['linear1'] if ffn is not None else layer.linear1
                linear2 = ffn['linear2'] if ffn is not None else layer.linear2
                norm = ffn['norm3'] if ffn is not None else layer.norm3
                tgt = norm(tgt + layer.dropout3(linear2(layer.dropout(layer.activation(linear1(tgt))))))
                states.append(module.decoder.norm(tgt).transpose(0, 1))
            predictions = [{'pred_logits': self.base.class_embed(h), 'pred_spans': self.base.span_embed(h).sigmoid()} for h in states]
            self._route_outputs[route] = dict(predictions[-1], aux_outputs=predictions[:-1])

    def forward(self, **inputs):
        out = self.base(**inputs)
        if self.training and self.routes:
            out['o2m_outputs'] = self._route_outputs
        return out

class MultiCriterion(nn.Module):
    def __init__(self, native, variant):
        super().__init__()
        self.native, self.variant = native, variant
        self.weight_dict = dict(native.weight_dict)
        self.routes = 2 if variant in ('multi', 'adaptive') else 1
        for r in ('shared', 'sepffn', 'instruction'):
            for k, v in native.weight_dict.items():
                if k.startswith(('loss_span', 'loss_giou', 'loss_label')):
                    self.weight_dict[f'{r}_{k}'] = v * .3 / self.routes
        self.matches = []

    def forward(self, outputs, targets):
        losses = self.native({k:v for k,v in outputs.items() if k != 'o2m_outputs'}, targets)
        for route, pred in outputs.get('o2m_outputs', {}).items():
            expanded = []
            for b, target in enumerate(targets['span_labels']):
                spans = target['spans']
                counts = torch.full((len(spans),), 2, device=spans.device, dtype=torch.long)
                if self.variant == 'adaptive' and len(spans):
                    p = span_cxw_to_xx(pred['pred_spans'][b].detach())
                    g = span_cxw_to_xx(spans)
                    inter = (torch.minimum(p[:,None,1],g[None,:,1])-torch.maximum(p[:,None,0],g[None,:,0])).clamp(min=0)
                    union = (p[:,1]-p[:,0])[:,None]+(g[:,1]-g[:,0])[None,:]-inter
                    counts = 1 + ((inter/union.clamp(min=1e-8)).topk(2,dim=0).values[1] >= .25).long()
                expanded.append({'spans': spans.repeat_interleave(counts, dim=0)})
                self.matches.append((int(counts.sum()), len(spans)))
            aux_targets = {'span_labels': expanded}
            for depth, p in enumerate([pred] + pred.get('aux_outputs', [])):
                indices = self.native.matcher(p, aux_targets)
                for loss in ('spans', 'labels'):
                    for k,v in self.native.get_loss(loss, p, aux_targets, indices).items():
                        losses[f'{route}_{k}' + (f'_{depth-1}' if depth else '')] = v
        return losses
