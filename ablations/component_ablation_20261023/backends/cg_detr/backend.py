"""CG-DETR-GMR initialization and exact S4 objective for the shared trainer."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from model import ROOT, S4CGDETR, build_model
from reference_modules import compute_mcev_loss

SOURCE = Path('/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/cg_detr/gmr/best.pt')
GMR = Path('/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def initialize(device, job, snapshot=None):
    """Strictly load the complete native GMR model before adding LoRA and S4."""
    (ROOT / 'snapshots').mkdir(exist_ok=True)
    if snapshot is None:
        directory = ROOT / 'snapshots'
        directory.mkdir(exist_ok=True)
        temporary = directory / f'capture_{os.getpid()}.tmp'
        with SOURCE.open('rb') as src, temporary.open('xb') as dst:
            before = os.fstat(src.fileno())
            shutil.copyfileobj(src, dst)
            after = os.fstat(src.fileno())
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        digest = sha256(temporary)
        snapshot = directory / f'cg_gmr_best_{digest}.pt'
        if snapshot.exists():
            assert sha256(snapshot) == digest
            temporary.unlink()
        else:
            temporary.rename(snapshot)
            snapshot.chmod(0o444)
    snapshot = Path(snapshot)
    digest = sha256(snapshot)
    checkpoint = torch.load(snapshot, map_location='cpu', weights_only=False)
    baseline_config = checkpoint['config']
    assert baseline_config['variant'] == 'cg_detr_gmr_requested_extension'
    baseline = checkpoint['model']
    assert all(k in baseline for k in ('exist_head.0.weight', 'exist_head.0.bias', 'exist_head.2.weight', 'exist_head.2.bias'))
    args = argparse.Namespace(**baseline_config)
    args.device = torch.device('cpu')
    assert args.disable_synthetic_negatives and args.txt_drop_ratio == 0
    base, _ = build_model(args)
    base.load_state_dict(baseline, strict=True)
    model = S4CGDETR(base, job)
    preserved = 0
    for name, tensor in model.base.state_dict().items():
        if '.lora_' in name:
            continue
        source_name = name.replace('.base_linear.', '.')
        assert torch.equal(tensor, baseline[source_name]), source_name
        preserved += 1
    assert preserved == len(baseline)
    assert all(not p.requires_grad for n, p in model.base.named_parameters() if '.lora_' not in n)
    counts = dict(total=sum(p.numel() for p in model.parameters()),
                  trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad),
                  lora=sum(p.numel() for n, p in model.named_parameters() if '.lora_' in n))
    initialization = dict(source_checkpoint=str(SOURCE), snapshot=str(snapshot), sha256=digest,
                          source_epoch=checkpoint['epoch'], source_best=checkpoint['best'],
                          verified_gmr_baseline=True, baseline_config=baseline_config, counts=counts,
                          checkpoint_tensors_preserved=preserved,
                          caveat='CG-DETR-GMR is the locally reconstructed Phase 1 requested extension; no published Table 2 CG-GMR target.')
    (ROOT / 'snapshots' / f'{digest}.json').write_text(json.dumps(initialization, indent=2, default=str) + '\n')
    return model.to(device), initialization


class Soccer(Dataset):
    def __init__(self, split):
        assert split in ('train', 'val', 'test'), 'Fixed-epoch test curves do not select checkpoints'
        self.data = [json.loads(line) for line in (GMR / 'data/label/Standard' / f'{split}.jsonl').read_text().splitlines() if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        features = []
        for feature_type in ('slowfast', 'clip'):
            with np.load(GMR / 'Soccergmr' / feature_type / f'{row["vid"]}.npz') as archive:
                feature = archive['features'].astype('float32')
            assert len(feature) <= 75
            features.append(feature / (np.linalg.norm(feature, axis=-1, keepdims=True) + 1e-5))
        n = min(map(len, features))
        assert n > 0
        video = torch.from_numpy(np.concatenate([feature[:n] for feature in features], axis=-1))
        tef = torch.stack([torch.arange(n) / n, (torch.arange(n) + 1) / n], dim=-1)
        with np.load(GMR / 'Soccergmr/clip_text' / f'qid{row["qid"]}.npz') as archive:
            text = archive['last_hidden_state'][:32].astype('float32')
        text = torch.from_numpy(text / (np.linalg.norm(text, axis=-1, keepdims=True) + 1e-5))
        return row, torch.cat([video, tef], dim=-1), text


def get_dataset(split):
    return Soccer(split)


def collate(items, device):
    rows, videos, texts = zip(*items)
    def pad(sequence):
        values = torch.nn.utils.rnn.pad_sequence(sequence, batch_first=True).to(device)
        mask = torch.arange(values.shape[1], device=device)[None] < torch.tensor([len(x) for x in sequence], device=device)[:, None]
        return values, mask.float()
    video, vm = pad(videos)
    text, tm = pad(texts)
    return list(rows), dict(src_vid=video, src_vid_mask=vm, src_txt=text, src_txt_mask=tm,
                            vid=None, qid=[r['qid'] for r in rows],
                            durations=torch.tensor([r['duration'] for r in rows], dtype=torch.float32, device=device))


def loss(output, metas, job):
    assert not job.get('hard_negative', False)
    assert not job.get('balanced_safeguard', False)
    assert not job.get('focal', False) and not job.get('margin_calibrated', False)
    assert not job.get('evidence_alignment', False)
    assert float(job.get('multi_weight', 0)) == 0
    target = torch.tensor([bool(row['relevant_windows']) for row in metas], device=output['pred_exist_logits'].device)
    return compute_mcev_loss(output, metas, {'exist_label': target}, job)
