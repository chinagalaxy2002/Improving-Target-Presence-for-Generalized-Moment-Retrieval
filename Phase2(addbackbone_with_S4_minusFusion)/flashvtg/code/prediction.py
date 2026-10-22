"""Validation and evaluation functions for Moment-Conditioned Event Verification."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from training.flash_vtg_gmr.dataset import start_end_collate
from training.flash_vtg_gmr.inference import compute_mr_results, post_processing_mr_nms
from models.flash_vtg_gmr.standalone_eval.eval import eval_submission
from eval.metrics import compute_G_mIoU, compute_gmr_cls, prepare_submission_for_gmiou


def _stable_localizer_payload(rows):
    payload = []
    for row in rows:
        windows = row.get('pred_relevant_windows_pre_exist')
        if windows is None:
            raise AssertionError('pred_relevant_windows_pre_exist is required for the safety check')
        payload.append([row['qid'], [[round(float(v), 6) for v in window] for window in windows]])
    return payload


def localizer_fingerprint(rows):
    raw = json.dumps(_stable_localizer_payload(rows), separators=(',', ':'), sort_keys=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def assert_localizer_invariant(rows, reference_path, tol=0.005):
    reference = [json.loads(line) for line in Path(reference_path).read_text().splitlines() if line.strip()]
    got_payload = _stable_localizer_payload(rows)
    ref_payload = _stable_localizer_payload(reference)
    if len(got_payload) != len(ref_payload):
        raise AssertionError(f'localizer output count changed: {len(got_payload)} vs {len(ref_payload)}')
    for index, (got, ref) in enumerate(zip(got_payload, ref_payload)):
        if got[0] != ref[0]:
            raise AssertionError(f'qid mismatch at row {index}: {got[0]} vs {ref[0]}')
        got_w = np.array(got[1])
        ref_w = np.array(ref[1])
        if got_w.shape != ref_w.shape or not np.allclose(got_w, ref_w, atol=tol):
            raise AssertionError(f'localizer output changed at row {index}, qid={got[0]}')
    return localizer_fingerprint(rows)


@torch.no_grad()
def predictions(model, dataset, opt):
    captured = []

    def collect(_module, _args, output):
        logits = output['pred_exist_logits'].detach().cpu().double().reshape(-1)
        multi = output.get('pred_multi_logits')
        multi = multi.detach().cpu().double() if multi is not None else None
        
        for index, value in enumerate(logits.tolist()):
            row = {
                'pred_exist_logit_raw': float(value),
                'pred_exist_score': float(torch.sigmoid(torch.tensor(value)).item()),
            }
            if multi is not None:
                row['pred_multi_logits_raw'] = multi[index].tolist()
            captured.append(row)

    handle = model.register_forward_hook(collect)
    try:
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=start_end_collate)
        pre, _ = compute_mr_results(model, loader, opt, None, None, None)
    finally:
        handle.remove()
        
    assert len(captured) == len(pre) == len(dataset), f"Mismatch in count: {len(captured)} vs {len(pre)}"
    for prediction, extra in zip(pre, captured):
        prediction.update(extra)
        # Update pred_exist_score in the prediction dict
        prediction['pred_exist_score'] = extra['pred_exist_score']
        
    post = post_processing_mr_nms(
        copy.deepcopy(pre), opt.nms_thd, opt.max_before_nms, opt.max_after_nms, opt.nms_type
    )
    return pre, post

