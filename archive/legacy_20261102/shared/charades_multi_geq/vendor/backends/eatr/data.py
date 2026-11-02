import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.span_utils import span_xx_to_cxw

def collate(batch):
    (metas, qs, vs, spans) = zip(*batch)
    qmask = pad_sequence([torch.ones(len(q), dtype=torch.bool) for q in qs], batch_first=True)
    vmask = pad_sequence([torch.ones(len(v), dtype=torch.bool) for v in vs], batch_first=True)
    inputs = {'src_txt': pad_sequence(qs, batch_first=True), 'src_txt_mask': qmask, 'src_vid': pad_sequence(vs, batch_first=True), 'src_vid_mask': vmask}
    return (list(metas), inputs, {'span_labels': [{'spans': s} for s in spans], 'exist_label': torch.tensor([bool(len(s)) for s in spans], dtype=torch.float32)})

def move(inputs, targets, device):
    return ({k: v.to(device) for (k, v) in inputs.items()}, {'span_labels': [{'spans': s['spans'].to(device)} for s in targets['span_labels']], 'exist_label': targets['exist_label'].to(device)})
