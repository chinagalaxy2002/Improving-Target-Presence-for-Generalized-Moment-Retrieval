"""Reconstruct each newly trained model and independently inspect all tensors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import torch
from components import build
from run import seed, dump, sha, group, controls

ROOT = Path(__file__).resolve().parent


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    shared = None
    results = []
    for setting in ['hard_only', 'global_hard']:
        directory = ROOT / 'runs' / args.backbone / setting
        config = json.loads((directory / 'CONFIG.json').read_text())
        complete = json.loads((directory / 'COMPLETE.json').read_text())
        assert complete['epochs'] == 4 and complete['fixed_epoch'] == 4
        assert [h['epoch'] for h in read_rows(directory / 'history.jsonl')] == [1, 2, 3, 4]
        checkpoint = directory / 'epoch_004.ckpt'
        assert sha(checkpoint) == json.loads((directory / 'CHECKPOINT.json').read_text())['sha256']
        seed()
        backend, model, init, extractor, refiner = build(args.backbone, config['job'], 'cpu')
        assert init['sha256'] == config['initialization']['sha256']
        baseline = json.loads((ROOT / 'runs' / args.backbone / 'gmr/CONFIG.json').read_text())
        assert init['sha256'] == baseline['initialization']['sha256']
        initial = {n: t.clone() for n, t in model.state_dict().items()}
        digests = {n: hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest() for n, t in initial.items()}
        assert digests == config['initial_tensor_sha256']
        if shared is not None:
            assert digests == shared, 'The two newly trained cells have different initial tensors'
        shared = digests
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['epoch'] == 4
        model.load_state_dict(state['model'], strict=True)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        changed = {}
        for n, t in model.state_dict().items():
            if n not in trainable:
                assert torch.equal(t, initial[n]), ('Frozen tensor changed', n)
            elif not torch.equal(t, initial[n]):
                changed[group(n)] = changed.get(group(n), 0) + t.numel()
        assert set(changed) == set(config['trainable_parameters_by_group'])
        assert not any(hasattr(m, 'lora_A') for m in model.modules())
        # Deliberately make the dormant saliency projection nonzero, checking
        # that the entire branch is bypassed rather than just zero initialized.
        saved_sal = {n: t.clone() for n, t in refiner.sal_proj.state_dict().items()}
        with torch.no_grad():
            refiner.sal_proj.weight.fill_(.2)
            refiner.sal_proj.bias.fill_(.7)
        checks = controls(extractor, refiner, 'cpu')
        refiner.sal_proj.load_state_dict(saved_sal, strict=True)
        # Presence must remain independent of RoI tokens in both cells.
        model.eval()
        metas, inputs = backend.collate([backend.get_dataset('val')[0]], 'cpu')
        with torch.no_grad():
            expected = model(**inputs)['pred_exist_logits'].clone()
            handle = extractor.register_forward_hook(lambda m, a, o: (o[0] + 1000, o[1], o[2]))
            actual = model(**inputs)['pred_exist_logits']
            handle.remove()
            assert torch.equal(expected, actual), 'Candidate features feed back into presence'
        prediction_checks = []
        for split in ['val', 'test']:
            pred = read_rows(directory / f'{split}_predictions.jsonl')
            base = {p['qid']: p for p in read_rows(ROOT / 'runs' / args.backbone / 'gmr' / f'{split}_predictions.jsonl')}
            if setting == 'hard_only':
                max_presence_error = max(abs(p['pred_exist_score'] - base[p['qid']]['pred_exist_score']) for p in pred)
                assert max_presence_error < 1e-7, ('Native presence changed', max_presence_error)
            else:
                max_presence_error = None
            if args.backbone != 'flashvtg':
                assert all(p['pred_relevant_windows_pre_exist'] == base[p['qid']]['pred_relevant_windows_pre_exist'] for p in pred), 'Frozen raw localizer changed'
            prediction_checks.append(dict(split=split, native_presence_max_error=max_presence_error))
        result = dict(passed=True, backbone=args.backbone, setting=setting, epoch=4, checkpoint_sha256=sha(checkpoint),
                      same_gmr_checkpoint=True, identical_initial_tensors_between_new_cells=True,
                      frozen_parameters_and_buffers_exactly_preserved=True,
                      changed_parameters_by_group=changed, lora_modules_absent=True,
                      saliency_bypassed_even_with_nonzero_projection=True,
                      candidate_perturbation_presence_invariant=True, prediction_checks=prediction_checks, **checks)
        dump(directory / 'INDEPENDENT_AUDIT.json', result)
        results.append(result)
        print('VERIFIED', args.backbone, setting, flush=True)
        del model, state, initial
    dump(ROOT / 'reports' / f'{args.backbone}_AUDIT.json', dict(passed=True, checks=results))


if __name__ == '__main__':
    main()
