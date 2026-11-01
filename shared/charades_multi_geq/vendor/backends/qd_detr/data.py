import json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from qd_detr.span_utils import span_xx_to_cxw

def collate(items):
    (meta, q, v, spans, sal, pos, neg, pm) = zip(*items)
    qm = pad_sequence([torch.ones(len(x)) for x in q], batch_first=True)
    vm = pad_sequence([torch.ones(len(x)) for x in v], batch_first=True)
    inputs = {'src_txt': pad_sequence(q, batch_first=True), 'src_vid': pad_sequence(v, batch_first=True), 'src_txt_mask': qm, 'src_vid_mask': vm}
    target = {'span_labels': [{'spans': s} for s in spans], 'saliency_all_labels': pad_sequence(sal, batch_first=True), 'saliency_pos_labels': torch.tensor(pos), 'saliency_neg_labels': torch.tensor(neg), 'saliency_pair_mask': torch.tensor(pm)}
    return (list(meta), inputs, target)

def move(inputs, targets, device):
    inputs = {k: v.to(device) for (k, v) in inputs.items()}
    target = {k: [{'spans': x['spans'].to(device)} for x in v] if k == 'span_labels' else v.to(device) for (k, v) in targets.items()}
    return (inputs, target)
