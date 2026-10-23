"""Small real-data CPU integrity check, never launches CUDA training."""
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import backend


def frozen_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def group(name):
    if '.lora_A' in name:
        return 'lora_A'
    if '.lora_B' in name:
        return 'lora_B'
    return name.split('.')[0]


def main():
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    model, initialization = backend.initialize('cpu', {})
    dataset = backend.get_dataset('train')
    ids = [next(i for i, item in enumerate(dataset.data) if len(item['relevant_windows']) == count)
           for count in (0, 1)]
    ids.append(next(i for i, item in enumerate(dataset.data) if len(item['relevant_windows']) >= 2))
    metas, inputs = backend.collate([dataset[i] for i in ids], 'cpu')
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    before = {n: p.detach().clone() for n, p in parameters}
    frozen_before = frozen_hash(model)
    gradients = set()
    optimizer = torch.optim.AdamW([p for _, p in parameters], lr=3e-5)
    records = []
    for step in range(3):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(**inputs)
        assert output['candidate_seconds'].shape[1] == 10
        windows = output['candidate_seconds']
        assert torch.isfinite(windows).all()
        assert (windows[..., 1] > windows[..., 0]).all()
        value, details = backend.loss(output, metas, {})
        assert torch.isfinite(value)
        value.backward()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                assert parameter.grad is None, name
            elif parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), name
                if torch.count_nonzero(parameter.grad):
                    gradients.add(group(name))
        optimizer.step()
        records.append({'step': step, 'loss': float(value.detach())})
        print(json.dumps(records[-1]), flush=True)
    required = {'lora_A', 'lora_B', 'global_query', 'extractor', 'refiner'}
    updates = {group(name) for name, parameter in parameters if not torch.equal(before[name], parameter)}
    assert required <= gradients and required <= updates
    assert frozen_hash(model) == frozen_before
    # Natural negatives have an empty GT list and still train the independent presence head.
    negative_metas, negative_inputs = backend.collate([dataset[ids[0]]], 'cpu')
    optimizer.zero_grad(set_to_none=True)
    negative_output = model(**negative_inputs)
    negative_loss, _ = backend.loss(negative_output, negative_metas, {})
    assert torch.isfinite(negative_loss)
    negative_loss.backward()
    model.eval()
    with torch.no_grad():
        original = model(**inputs)
        changed = original['selected_candidates'].clone()
        changed[..., :2] = torch.tensor([0., 2.])
        changed[..., 2] = 0.
        altered = model(**inputs, candidate_override=changed)
        assert torch.equal(original['pred_exist_logits'], altered['pred_exist_logits'])
    checkpoint = Path(__file__).parent / 'cpu_integrity.ckpt'
    torch.save({'model': model.state_dict(), 'initialization': initialization,
                's4_config': model.config}, checkpoint)
    reloaded, _ = backend.initialize('cpu', model.config, snapshot=initialization['snapshot'])
    reloaded.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    reloaded.eval()
    with torch.no_grad():
        replay = reloaded(**inputs)
    assert torch.equal(original['pred_exist_logits'], replay['pred_exist_logits'])
    assert torch.equal(original['candidate_seconds'], replay['candidate_seconds'])
    result = {'status': 'CPU_VALIDATED', 'initialization': initialization,
              'steps': records, 'negative_loss': float(negative_loss.detach()),
              'gradient_groups': sorted(gradients), 'updated_groups': sorted(updates),
              'frozen_unchanged': True, 'strict_reload': True,
              'candidate_independent_presence': True, 'sample_ids': ids,
              'cuda_used': False, 'evaluation': 'Parent common trainer runs official validation.'}
    (Path(__file__).parent / 'CPU_VALIDATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'initialization'}), flush=True)


if __name__ == '__main__':
    main()
