"""CPU-only regression gate using real natural positive/negative/multimoment data."""
import json
from pathlib import Path
import torch
from backend import initialize, get_dataset, collate, loss
ROOT = Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2)
    job = json.loads((ROOT.parent / 'shared' / 'config.json').read_text())
    job['full_training_authorized'] = True
    torch.manual_seed(2026)
    model, initialization = initialize('cpu', job)
    (ROOT / 'initialization.json').write_text(json.dumps(initialization, indent=2) + '\n')
    ds = get_dataset('train')
    indices = [next(i for i, item in enumerate(ds.data) if not item['relevant_windows']),
               next(i for i, item in enumerate(ds.data) if len(item['relevant_windows']) == 1),
               next(i for i, item in enumerate(ds.data) if len(item['relevant_windows']) > 1)]
    metas, inputs = collate([ds[index] for index in indices], 'cpu')
    frozen = {name: value.clone() for name, value in model.named_parameters() if not value.requires_grad}
    trainable = {name: value.clone() for name, value in model.named_parameters() if value.requires_grad}
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=3e-5, weight_decay=1e-4)
    losses = []
    for step in range(3):
        model.train(); optimizer.zero_grad(set_to_none=True)
        output = model(**inputs)
        total, detail = loss(output, metas, job)
        assert torch.isfinite(total)
        total.backward()
        assert all(value.grad is None for name, value in model.named_parameters() if name in frozen)
        assert all(torch.isfinite(value.grad).all() for value in model.parameters() if value.grad is not None)
        for prefix in ('base.', 'global_query.', 'extractor.', 'refiner.'):
            grads = [value.grad for name, value in model.named_parameters() if name.startswith(prefix) and value.requires_grad]
            assert any(g is not None and torch.isfinite(g).all() for g in grads), (step, prefix)
            if step > 0 or prefix != 'extractor.':
                assert any(g is not None and g.abs().sum() > 0 for g in grads), (step, prefix)
        optimizer.step()
        losses.append(float(total.detach()))
        assert all(torch.equal(value, frozen[name]) for name, value in model.named_parameters() if name in frozen)
    updated = [name for name, value in model.named_parameters() if name in trainable and not torch.equal(value, trainable[name])]
    assert any('.lora_' in name for name in updated)
    for prefix in ('global_query.', 'extractor.', 'refiner.'):
        assert any(name.startswith(prefix) for name in updated)
    model.eval()
    with torch.no_grad():
        before = model(**inputs)
        altered = before['selected_candidates'].clone()
        altered[..., :2] = torch.tensor([0.1, 0.9])
        altered[..., 2] = 0.123
        independent = model(**inputs, candidate_override=altered)
        assert torch.equal(before['pred_exist_logits'], independent['pred_exist_logits'])
        for bounds in (before['candidate_seconds'], before['refined_spans'], independent['candidate_seconds']):
            assert torch.isfinite(bounds).all() and torch.all(bounds[..., 1] > bounds[..., 0])
            assert torch.all(bounds[..., 0] >= 0) and torch.all(bounds[..., 1] <= inputs['durations'][:,None])
    checkpoint = ROOT / 'cpu_smoke.ckpt'
    torch.save({'model':model.state_dict(), 'job':job, 'initialization':initialization}, checkpoint)
    reload, reload_initialization = initialize('cpu', job, snapshot=initialization['snapshot'])
    reload.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    reload.eval()
    with torch.no_grad():
        after = reload(**inputs)
    for key in ('pred_exist_logits', 'candidate_seconds', 'refined_spans'):
        assert torch.equal(before[key], after[key]), key
    result = {'status':'PASSED', 'device':'cpu', 'steps':3, 'losses':losses, 'sample_qids':[item['qid'] for item in metas],
              'strict_base_load':True, 'frozen_unchanged':True, 'all_module_groups_updated':True, 'presence_candidate_independent':True,
              'positive_boundaries':True, 'strict_reload_exact_outputs':True, 'counts':initialization['counts'], 'sha256':initialization['sha256']}
    (ROOT / 'cpu_smoke.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)

if __name__ == '__main__': main()
