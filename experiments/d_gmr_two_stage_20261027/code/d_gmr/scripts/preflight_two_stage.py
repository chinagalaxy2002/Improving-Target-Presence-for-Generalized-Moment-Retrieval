"""Real-batch checks for decoupled two-stage initialization; no optimizer update is made."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from d_gmr.common import BACKBONES, RUNS, configuration, dump, seed_all
from d_gmr.engine.train import construct, optimizer_for
from d_gmr.losses import objective


def gradient_norm(loss, parameters):
    grads = torch.autograd.grad(loss, list(parameters), allow_unused=True, retain_graph=True)
    values = [g.square().sum() for g in grads if g is not None]
    value = torch.stack(values).sum().sqrt() if values else loss.new_zeros(())
    assert torch.isfinite(value)
    return float(value.detach())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=BACKBONES, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(2)
    seed_all()
    config = configuration()
    assert config['two_stage_geq_init'] is True
    assert config['freeze_geq'] is True
    assert config['detach_geq_event_for_tem'] is True

    stage1_ckpt = config['stage1_geq_checkpoints'][args.backbone]
    assert os.path.exists(stage1_ckpt), f"Missing Stage 1 GEQ checkpoint: {stage1_ckpt}"

    config['backbone'] = args.backbone
    adapter, model = construct(args.backbone, args.device, config)

    # Verify GEQ weights exactly match Stage 1 checkpoint
    raw_ckpt = torch.load(stage1_ckpt, map_location='cpu', weights_only=False)
    m = raw_ckpt.get('model', raw_ckpt)
    expected_geq = {k[4:]: v for k, v in m.items() if k.startswith('geq.')}
    model_geq = model.geq.state_dict()
    assert set(expected_geq.keys()) == set(model_geq.keys()), "GEQ keys mismatch"
    for k in expected_geq:
        assert torch.equal(expected_geq[k].to(args.device), model_geq[k]), f"GEQ tensor mismatch on {k}"

    # Verify parameter requires_grad status
    geq_params = list(model.geq.parameters())
    assert all(not p.requires_grad for p in geq_params), "All GEQ parameters must be frozen"

    trainable_base = [p for p in model.base.parameters() if p.requires_grad]
    trainable_tem = [p for p in model.temporal_evidence.parameters() if p.requires_grad]
    trainable_ranker = [p for p in model.ranker.parameters() if p.requires_grad]
    assert len(trainable_base) > 0 and len(trainable_tem) > 0 and len(trainable_ranker) > 0

    # Verify optimizer parameter groups
    optimizer, scheduler = optimizer_for(model, config)
    opt_names = [g.get('name') for g in optimizer.param_groups]
    assert 'native_gmr' in opt_names and 'tem_tecr' in opt_names
    assert 'geq' not in opt_names, "Frozen GEQ must not be in optimizer groups"

    # Forward & Backward check with real train batch
    data = adapter.dataset('train')
    indices = [next(i for i, meta in enumerate(data.data)
                    if len(meta['relevant_windows']) == count) for count in (0, 1, 2)]
    metas, inputs, targets = adapter.collate([data[i] for i in indices])

    model.train()
    adapter.criterion.train()
    output = model(inputs)

    for key in ('presence_logit', 'ranked_proposals', 'quality_logit', 'ranking_logits'):
        assert torch.isfinite(output[key]).all(), f"Non-finite output in {key}"

    loss, parts, contributions, info, native_details = objective(adapter, output, metas, targets, config)
    assert torch.isfinite(loss), f"Non-finite loss: {loss}"

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # GEQ must have no gradients
    for p in model.geq.parameters():
        assert p.grad is None, "Frozen GEQ must have None grad"

    # Trainable parameters with gradients must be finite
    trainable_grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(trainable_grads) > 0 and all(torch.isfinite(g).all() for g in trainable_grads), "Trainable params with grad must be finite"

    report = dict(
        passed=True, backbone=args.backbone, stage1_checkpoint=stage1_ckpt,
        sample_indices=indices, loss_total=float(loss.detach()),
        geq_params_count=len(geq_params),
        geq_frozen=True,
        trainable_params_count=len(trainable_grads),
        optimizer_groups=opt_names
    )
    dump(RUNS/'preflights'/args.backbone/'PREFLIGHT.json', report, exclusive=True)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
