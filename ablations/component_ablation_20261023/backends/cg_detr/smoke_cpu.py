"""Three-step real-data correctness check; deliberately CPU-only."""
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import backend

torch.set_num_threads(2)
torch.manual_seed(2026)
job = json.loads((Path(__file__).parent.parent / 'shared/config.json').read_text())
model, initialization = backend.initialize('cpu', job)
dataset = backend.get_dataset('train')
indices = [next(i for i, row in enumerate(dataset.data) if len(row['relevant_windows']) == n) for n in (0, 1)]
indices += [next(i for i, row in enumerate(dataset.data) if len(row['relevant_windows']) > 1),
            next(i for i, row in enumerate(dataset.data) if row['duration'] < 150 and row['relevant_windows'])]
rows, inputs = backend.collate([dataset[i] for i in indices], 'cpu')
frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
active = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5, foreach=False)
steps = []
for step in range(3):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(**inputs)
    loss, details = backend.loss(output, rows, job)
    assert torch.isfinite(loss)
    assert (output['candidate_seconds'][..., 1] > output['candidate_seconds'][..., 0]).all()
    loss.backward()
    gradients = {n: float(p.grad.norm()) for n, p in model.named_parameters() if p.grad is not None}
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert not any(p.grad is not None for p in model.parameters() if not p.requires_grad)
    assert any(v > 0 for n, v in gradients.items() if '.lora_B' in n)
    assert any(v > 0 for n, v in gradients.items() if n.startswith('global_event_query.'))
    if step:
        assert any(v > 0 for n, v in gradients.items() if '.lora_A' in n)
        assert any(v > 0 for n, v in gradients.items() if n.startswith('extractor.'))
    assert any(v > 0 for n, v in gradients.items() if n.startswith('refiner.'))
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], .1, error_if_nonfinite=True)
    optimizer.step()
    steps.append(dict(step=step + 1, loss=float(loss.detach()), gradient_norms=gradients))
    print('CPU_SMOKE', step + 1, float(loss.detach()), flush=True)
assert all(torch.equal(p, frozen[n]) for n, p in model.named_parameters() if n in frozen)
changed = [n for n, p in model.named_parameters() if n in active and not torch.equal(p, active[n])]
for group in ('global_event_query.', 'extractor.', 'refiner.', '.lora_A', '.lora_B'):
    assert any(group in n for n in changed), group
model.eval()
with torch.no_grad():
    original = model(**inputs)
    override = original['candidate_spans'].clone()
    override[..., :2] = torch.tensor([0., 2.])
    override[..., 2] = .01
    modified = model(**inputs, candidate_override=override)
    assert torch.equal(original['pred_exist_logits'], modified['pred_exist_logits'])
saved = Path(__file__).parent / 'cpu_smoke.pt'
torch.save(dict(model=model.state_dict(), job=job, initialization=initialization), saved)
restored, _ = backend.initialize('cpu', job, snapshot=initialization['snapshot'])
restored.load_state_dict(torch.load(saved, weights_only=False)['model'], strict=True)
restored.eval()
with torch.no_grad():
    result = restored(**inputs)
assert torch.equal(original['pred_exist_logits'], result['pred_exist_logits'])
assert torch.equal(original['candidate_seconds'], result['candidate_seconds'])
report = dict(status='CPU_VALIDATED', initialization=initialization, qids=[r['qid'] for r in rows], steps=steps,
              frozen_unchanged=True, exact_reload=True, presence_candidate_independent=True,
              trainable_parameters_updated=changed, augmentation=False)
(Path(__file__).parent / 'cpu_smoke.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
print('CPU_SMOKE_PASSED', initialization['counts'], flush=True)
