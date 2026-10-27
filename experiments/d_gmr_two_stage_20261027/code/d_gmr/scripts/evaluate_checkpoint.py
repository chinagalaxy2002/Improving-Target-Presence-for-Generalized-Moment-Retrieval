"""Evaluate any D-GMR epoch, with a guarded Val-selected formal Test mode."""
import argparse
import json
from pathlib import Path

import torch

from gemr.engine import checkpoint
from gemr.engine.evaluate import score
from d_gmr.common import BACKBONES, LABELS, RUNS, dump, rows, seed_all
from d_gmr.engine.train import construct
from d_gmr.evaluate import save_evaluation, standalone_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', required=True, choices=BACKBONES)
    parser.add_argument('--checkpoint')
    parser.add_argument('--split', required=True, choices=['val', 'test'])
    parser.add_argument('--threshold', type=float, default=.4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output')
    parser.add_argument('--predictions', help='Score a frozen prediction JSONL without model inference')
    parser.add_argument('--formal-test', action='store_true')
    args = parser.parse_args()
    assert 0 < args.threshold < 1
    torch.set_num_threads(2)
    seed_all()
    if args.predictions:
        assert not args.formal_test
        output = Path(args.output) if args.output else standalone_directory(
            RUNS/args.backbone, args.split, 0, args.threshold)
        output.mkdir(parents=True, exist_ok=False)
        metrics = score(rows(args.predictions), rows(LABELS/f'{args.split}.jsonl'), args.threshold)
        dump(output/'metrics.json', metrics, exclusive=True)
        (output/'RESULTS.md').write_text(
            '# Frozen-prediction evaluation\n\n```json\n'+json.dumps(metrics['brief'], indent=2)+'\n```\n')
        print(output, flush=True)
        return
    assert args.checkpoint, '--checkpoint or --predictions is required'
    state = checkpoint.load(args.checkpoint)
    config = state['config']
    assert config['backbone'] == args.backbone
    run = Path(args.checkpoint).resolve().parent.parent
    if args.formal_test:
        assert args.split == 'test' and args.threshold == .4
        complete = json.loads((run/'TRAIN_COMPLETE.json').read_text())
        best = json.loads((run/'best_epoch.json').read_text())
        assert complete['epochs'] == 4 and best['epochs_considered'] == [1, 2, 3, 4]
        assert Path(best['checkpoint_path']).resolve() == Path(args.checkpoint).resolve()
        assert best['selection_split'] == 'val' and state['epoch'] == best['best_epoch']
        assert not (run/'TEST_COMPLETE.json').exists()
        dump(run/'FORMAL_TEST_SELECTION.json', best, exclusive=True)
    adapter, model = construct(args.backbone, args.device, config)
    model.load_state_dict(state['model_state_dict'], strict=True)
    data = adapter.dataset(args.split)
    output = Path(args.output) if args.output else standalone_directory(
        run, args.split, state['epoch'], args.threshold)
    metrics, original, joint_native, joint_geq = save_evaluation(
        model, data, args.split, output,
        1 if args.backbone == 'flashvtg' else config['batch_sizes'][args.backbone],
        args.threshold, args.checkpoint)
    if args.formal_test:
        dump(run/'TEST_COMPLETE.json', dict(
            epoch=state['epoch'], checkpoint=str(Path(args.checkpoint).resolve()),
            selection_split='val', output=str(output.resolve()), metrics=metrics['brief'],
            original_gmr_metrics=original['brief'], joint_native_metrics=joint_native['brief'],
            joint_geq_metrics=joint_geq['brief']), exclusive=True)
    print(json.dumps(dict(output=str(output), metrics=metrics['brief'])), flush=True)


if __name__ == '__main__':
    main()
