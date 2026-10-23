"""Per-epoch Test reporting with training RNG preserved."""
import random
import numpy as np
import torch
from environment import write_json, write_rows
from prediction import predictions
from evaluation import Evaluation
from qcd_metrics import worldcup_report


def capture_rng(loader=None, qcd=None):
    state = dict(python=random.getstate(), numpy=np.random.get_state(),
                 torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())
    if loader is not None:
        state['loader'] = loader.generator.get_state()
    if qcd is not None:
        import copy
        state['qcd'] = copy.deepcopy(qcd.rng.bit_generator.state)
    return state


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def evaluate_epoch(model, dataset, opt, calibration, directory, epoch):
    rng = capture_rng()
    try:
        model.eval()
        with torch.inference_mode():
            pre, post = predictions(model, dataset, opt)
        report = Evaluation(pre, post, dataset.data).report('test', calibration=calibration, comprehensive=True)
        report['worldcup_qcd'] = worldcup_report(pre, post, dataset.data)
        report['checkpoint_epoch'] = epoch
        report['protocol'] = 'User-requested per-epoch Test diagnostic; no Test-based checkpoint selection; calibration from same-epoch Val'
        write_rows(directory/'predictions_pre_nms.jsonl', pre)
        write_rows(directory/'predictions.jsonl', post)
        write_json(directory/'metrics.json', report)
        write_json(directory/'COMPLETE.json', dict(epoch=epoch, rows=len(post), training_rng_preserved=True))
        return report
    finally:
        restore_rng(rng)
