"""Train-only multi-aware QCD. Independent RNG; labels and TEF are untouched."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def worldcup(row):
    return row.get('dataset_source', '').lower() == 'worldcup2022'


def masks(windows, length, clip_len=2., context=1):
    gt = np.zeros(length, dtype=bool)
    protected = gt.copy()
    for start, end in windows:
        if not (math.isfinite(start) and math.isfinite(end) and end > start):
            raise ValueError(f'Invalid GT window: {(start, end)}')
        a = max(0, min(length, math.floor(start / clip_len)))
        b = max(0, min(length, math.ceil(end / clip_len)))
        if b > a:
            gt[a:b] = True
            protected[max(0, a-context):min(length, b+context)] = True
    return gt, protected


def replace_count(length, gt, protected, ratio):
    # Python round: ties to even, explicitly fixed for reproducibility.
    return min(int((~protected).sum()), round(ratio * (length-int(gt.sum()))))


class QCD:
    def __init__(self, dataset, config, cache=None):
        self.config = dict(config)
        self.rng = np.random.default_rng(config['seed'])
        self.counts = Counter()
        self.bank = {}
        self.clip_len = dataset.clip_len
        if not 0 <= config['p_aug'] <= 1 or not 0 <= config['replace_ratio'] <= 1:
            raise ValueError('QCD probability and ratio must be in [0,1]')
        if config['scope'] not in ('all', 'multi') or config['context_size'] < 0:
            raise ValueError('Invalid scope/context')
        # Donors are original, normalized Train tensors, never a mixed batch.
        for row, inputs in dataset.preloaded_data:
            if worldcup(row):
                self.bank.setdefault(row['vid'], inputs['video_feat'][:, :2816])
        self.vids = sorted(self.bank)
        self.cache = None
        if config['enabled'] and config['query_filter']:
            cache = Path(cache)
            meta = json.loads(cache.with_suffix('.json').read_text())
            if meta['train_sha256'] != sha(dataset.data_path):
                raise ValueError('QCD cache Train manifest mismatch')
            if meta['vids'] != self.vids:
                raise ValueError('QCD donor membership mismatch')
            if meta['clip_len'] != dataset.clip_len or meta['max_v_l'] != dataset.max_v_l:
                raise ValueError('QCD cache temporal protocol mismatch')
            if meta['theta_min'] > meta['theta_max']:
                raise ValueError('Empty QCD threshold interval')
            if meta['cache_sha256'] != sha(cache):
                raise ValueError('QCD cache content mismatch')
            with np.load(cache, allow_pickle=False) as archive:
                self.cache = {key: archive[key] for key in archive.files}
            self.queries = {q: i for i, q in enumerate(meta['queries'])}
            self.thresholds = meta['theta_min'], meta['theta_max']
            for i, vid in enumerate(self.vids):
                n = int(self.cache['offsets'][i+1]-self.cache['offsets'][i])
                if len(self.bank[vid]) != n:
                    raise ValueError(f'QCD donor length mismatch: {vid}')
                clip = self.bank[vid][:, 2304:2816].numpy()
                clip = clip / np.maximum(np.linalg.norm(clip, axis=1, keepdims=True), 1e-12)
                if not np.allclose(clip, self.cache['clip'][self.cache['offsets'][i]:self.cache['offsets'][i+1]], atol=2e-6):
                    raise ValueError(f'QCD donor CLIP mismatch: {vid}')

    @torch.no_grad()
    def apply(self, features, metadata):
        if not self.config['enabled']:
            return []
        tensor = features['src_vid']
        if tensor.shape[-1] != 2818:
            raise ValueError('QCD requires SlowFast2304 + CLIP512 + TEF2')
        records = []
        for i, row in enumerate(metadata):
            windows = row['relevant_windows']
            if not worldcup(row) or not windows or (self.config['scope'] == 'multi' and len(windows) < 2):
                continue
            self.counts['eligible_samples'] += 1
            if self.rng.random() >= self.config['p_aug']:
                continue
            self.counts['triggered'] += 1
            valid = features['src_vid_mask'][i].bool()
            length = int(valid.sum())
            if not bool(valid[:length].all()) or bool(valid[length:].any()):
                raise ValueError('QCD requires prefix padding')
            gt, protected = masks(windows, length, self.clip_len, self.config['context_size'])
            n = replace_count(length, gt, protected, self.config['replace_ratio'])
            if not n:
                self.counts['no_background'] += 1
                continue
            donors = [j for j, vid in enumerate(self.vids) if vid != row['vid']]
            if not donors:
                self.counts['no_donor'] += 1
                continue
            j = int(self.rng.choice(donors))
            donor = self.vids[j]
            pool = np.arange(len(self.bank[donor]))
            if self.cache is not None:
                scores = self.cache['scores'][self.queries[row['query']], self.cache['offsets'][j]:self.cache['offsets'][j+1]]
                pool = pool[(scores >= self.thresholds[0]) & (scores <= self.thresholds[1])]
            self.counts['candidate_clips'] += len(pool)
            if not len(pool):
                self.counts['empty_pool'] += 1
                continue
            dst = self.rng.choice(np.flatnonzero(~protected), size=n, replace=False)
            src = self.rng.choice(pool, size=n, replace=True)
            donor_features = self.bank[donor][torch.from_numpy(src)].to(tensor.device, tensor.dtype)
            tensor[i, torch.as_tensor(dst, device=tensor.device), :2816] = donor_features
            self.counts['augmented'] += 1
            self.counts['replaced_clips'] += n
            self.counts['non_gt_clips_augmented'] += int((~gt).sum())
            self.counts['multi_augmented' if len(windows) > 1 else 'single_augmented'] += 1
            records.append(dict(qid=row['qid'], donor=donor, dst=dst.tolist(), src=src.tolist(),
                                gt_clips=int(gt.sum()), protected_clips=int(protected.sum()), pool_size=len(pool)))
        return records
