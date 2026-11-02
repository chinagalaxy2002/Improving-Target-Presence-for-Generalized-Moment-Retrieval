"""Local paper-style GMR extension; QD-DETR-GMR has no published Table2 row."""
import torch
from torch import nn
from torch.nn import functional as F

class GMRAdapter(nn.Module):
    """Elementwise max over final decoder slots, two-layer ReLU existence MLP."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, slots):
        return self.mlp(slots.max(dim=1).values).squeeze(-1)

class GMRCriterion(nn.Module):
    """Native VMR loss on positive queries; existence BCE on every real query."""
    def __init__(self, native, existence_coef=1.0):
        super().__init__()
        self.native = native
        self.weight_dict = {**native.weight_dict, 'loss_exist': existence_coef}
        self.last_positive_count = None

    def forward(self, outputs, targets):
        logits = outputs['pred_exist_logits'].reshape(-1)
        labels = logits.new_tensor([float(len(t['spans']) > 0) for t in targets['span_labels']])
        positive = labels.bool()
        indices = positive.nonzero(as_tuple=True)[0].tolist()
        self.last_positive_count = len(indices)
        if indices:
            def select(value):
                if torch.is_tensor(value):
                    return value[positive]
                if isinstance(value, dict):
                    return {k:select(v) for k,v in value.items()}
                if isinstance(value, list):
                    return [select(v) for v in value]
                return value
            positive_outputs = {k:select(v) for k,v in outputs.items() if k != 'pred_exist_logits'}
            positive_targets = {k:([v[i] for i in indices] if k == 'span_labels' else v[positive]) for k,v in targets.items()}
            losses = self.native(positive_outputs, positive_targets)
        else:
            zero = outputs['pred_logits'].sum() * 0 + outputs['pred_spans'].sum() * 0
            losses = {k:zero for k in self.native.weight_dict}
        losses['loss_exist'] = F.binary_cross_entropy_with_logits(logits, labels)
        return losses
