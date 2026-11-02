import copy,random
import numpy as np
import torch
from torch.utils.data import Dataset
from paths import FEATURES
class Features(Dataset):
    def __init__(self, rows, adapter, dataset='charades'):
        self.data, self.adapter = rows, adapter

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        m, ad = self.data[index], self.adapter
        def read(folder, name, key):
            with np.load(FEATURES / folder / (name+'.npz')) as f:
                x = f[key].astype(np.float32)
            return x/(np.linalg.norm(x, axis=-1, keepdims=True)+1e-5)
        with np.load(FEATURES/'clip_text'/f"qid{m['original_qid']}.npz") as f:
            # Native collators build their padding mask from the unpadded length.
            q = f['last_hidden_state'][f['attention_mask'].astype(bool)].astype(np.float32)
        q = q/(np.linalg.norm(q, axis=-1, keepdims=True)+1e-5)
        order = ('clip', 'slowfast') if ad.name == 'moment_detr' else ('slowfast', 'clip')
        clips = [read(k, m['vid'], 'features') for k in order]
        assert len(clips[0]) == len(clips[1])
        n = len(clips[0])
        assert n > 0 and n <= ad.opt.max_v_l and abs(n*2-float(m['duration'])) <= 1.01, (m['vid'], n, m['duration'])
        v = np.concatenate([x[:n] for x in clips], axis=-1)
        tef = np.stack([np.arange(n)/n, (np.arange(n)+1)/n], axis=-1).astype(np.float32)
        windows = copy.deepcopy(m['relevant_windows'])
        if ad.name in ('moment_detr', 'flashvtg') and len(windows) > ad.opt.max_windows:
            random.shuffle(windows)
            windows = windows[:ad.opt.max_windows]
        if windows:
            a = torch.tensor(windows, dtype=torch.float32)/float(m['duration'])
            spans = torch.stack(((a[:, 0]+a[:, 1])/2, a[:, 1]-a[:, 0]), -1)
        else:
            spans = torch.zeros((0, 2), dtype=torch.float32)
        q = torch.from_numpy(q)
        v = torch.from_numpy(np.concatenate([v, tef], axis=-1))
        if ad.name == 'eatr':
            return m, q, v, spans
        if ad.name in ('qd_detr', 'cg_detr'):
            sal = torch.zeros(n)
            for start, end in windows:
                a = max(0, min(n-1, int(start/ad.opt.clip_length)))
                b = min(n, max(a+1, int(np.ceil(end/ad.opt.clip_length))))
                sal[a:b] = 1
            pos, neg = torch.where(sal > 0)[0].tolist(), torch.where(sal == 0)[0].tolist()
            pi = random.choices(pos, k=2) if pos else [0, 0]
            ni = random.choices(neg, k=2) if neg else pi
            if ad.name == 'qd_detr':
                return m, q, v, spans, sal, pi, ni, float(bool(pos) and bool(neg))
            return m, v, q, spans, sal, pi, ni
        values = {'query_feat': q, 'video_feat': v, 'span_labels': spans, 'exist_label': float(bool(windows))}
        if ad.name == 'flashvtg':
            values.update(vid=m['vid'], qid=m['original_qid'])
            return m, values
        return {'meta': m, 'model_inputs': values}

