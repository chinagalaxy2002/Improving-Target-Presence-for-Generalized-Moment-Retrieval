import copy, hashlib, json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from official_eval.metrics import _ap_worker, _clean_pred_windows, _compute_set_iou_score, compute_G_mIoU, compute_mAP

class Evaluation:
    def __init__(self, pre, post, ground_truth):
        self.gt = ground_truth
        self.y = np.array([bool(g['relevant_windows']) for g in ground_truth])
        self.count = np.array([len(g['relevant_windows']) for g in ground_truth])
        post_map, pre_map = {p['qid']: p for p in post}, {p['qid']: p for p in pre}
        assert len(post) == len(post_map) == len(ground_truth)
        assert len(pre) == len(pre_map) == len(ground_truth)
        assert set(post_map) == set(pre_map) == {g['qid'] for g in ground_truth}
        self.post = [post_map[g['qid']] for g in ground_truth]
        self.pre = [pre_map[g['qid']] for g in ground_truth]
        self.logit = np.array([p['pred_exist_logit_raw'] for p in self.post])
        self.score = np.array([p['pred_exist_score'] for p in self.post], dtype=np.float64)
        self.local = np.array([[_compute_set_iou_score([w[:2] for w in p['pred_relevant_windows'][:k]], g['relevant_windows'])
                                for k in [1, 3, 5]] for p, g in zip(self.post, ground_truth)])
        self.ap = self.ap_values(self.post, 'pred_relevant_windows')
        self.raw_ap = self.ap_values(self.pre, 'pred_relevant_windows_pre_exist')
        self.native_ap = self.ap_values(self.pre, 'pred_relevant_windows')

    def ap_values(self, rows, key):
        result = []
        for p, g in zip(rows, self.gt):
            ground = [{'video-id': g['qid'], 't-start': w[0], 't-end': w[1]} for w in g['relevant_windows']]
            pred = [{'video-id': g['qid'], 't-start': w[0], 't-end': w[1], 'score': w[2]} for w in p[key][:10]]
            result.append(_ap_worker((g['qid'], ground, pred), np.round(np.linspace(.5, .95, 10), 2))[1].mean())
        return np.array(result)

    def metrics(self, threshold=.5, mask=None, accept=None):
        mask = np.ones(len(self.gt), bool) if mask is None else mask
        y = self.y[mask]
        gate_all = self.score > threshold if accept is None else accept
        gate = gate_all[mask]
        contributions = np.where(gate[:, None], self.local[mask], (~y)[:, None])
        pos = mask & self.y
        tp, tn = int((gate & y).sum()), int((~gate & ~y).sum())
        fp, fn = int((gate & ~y).sum()), int((~gate & y).sum())
        positive = 100 * float(gate[y].mean()) if y.any() else None
        negative = 100 * float((~gate[~y]).mean()) if (~y).any() else None
        result = {f'G-mIoU@{k}': round(100 * float(sum(contributions[:, i].tolist()) / len(contributions)), 2) for i, k in enumerate([1, 3, 5])}
        result.update(threshold=float(threshold), total=int(mask.sum()), positive=int(y.sum()), negative=int((~y).sum()),
                      TP=tp, TN=tn, FP=fp, FN=fn, negative_recall=negative, N_acc=negative,
                      positive_recall=positive, balanced_accuracy=(positive + negative) / 2 if positive is not None and negative is not None else None,
                      Acc=round(100 * (tp + tn) / len(y), 2),
                      mAP=round(100 * float(self.ap[pos].mean()), 2) if y.any() else None,
                      raw_mAP=round(100 * float(self.raw_ap[pos].mean()), 2) if y.any() else None,
                      hard_gated_mAP_diagnostic=round(100 * float((self.ap[pos] * gate_all[pos]).mean()), 2) if y.any() else None,
                      AUROC=100 * float(roc_auc_score(y, self.logit[mask])) if y.any() and (~y).any() else None)
        return result

    def verify(self):
        gated = copy.deepcopy(self.post)
        for row, score in zip(gated, self.score):
            if score <= .5:
                row['pred_relevant_windows'] = []
        metric = self.metrics()
        official = compute_G_mIoU(gated, self.gt, (1, 3, 5))
        assert all(metric[k] == v for k, v in official.items()), (metric, official)
        assert compute_mAP(self.post, self.gt, num_workers=0)['mAP'] == metric['mAP']
        assert compute_mAP(gated, self.gt, num_workers=0)['mAP'] == metric['hard_gated_mAP_diagnostic']
        raw = copy.deepcopy(self.pre)
        for row in raw:
            row['pred_relevant_windows'] = row['pred_relevant_windows_pre_exist']
        assert compute_mAP(raw, self.gt, num_workers=0)['mAP'] == metric['raw_mAP']
        return 'verified against official compute_G_mIoU / compute_mAP'

    def report(self, split, calibration=None, comprehensive=False):
        payload = [[p['qid'], p['pred_relevant_windows_pre_exist']] for p in self.pre]
        report = {'split': split, 'primary_threshold': .5, 'fixed_05': self.metrics(.5),
                  'gate_03': self.metrics(.3), 'gate_04': self.metrics(.4), 'gate_06': self.metrics(.6), 'gate_07': self.metrics(.7),
                  'AUROC': 100 * float(roc_auc_score(self.y, self.logit)),
                  'raw_localizer_pre_exist': {'MR-full-mAP': round(100 * float(self.raw_ap[self.y].mean()), 2)},
                  'native_pre_nms': {'MR-full-mAP': round(100 * float(self.native_ap[self.y].mean()), 2)},
                  'official_postprocessed': {'mAP': self.metrics()['mAP']},
                  'counts': {'total': len(self.y), 'positive': int(self.y.sum()), 'negative': int((~self.y).sum())},
                  'ground_truth_fingerprint': hashlib.sha256(json.dumps(
                      [[g['qid'], g['vid'], g['duration'], g['relevant_windows']] for g in self.gt],
                      separators=(',', ':')).encode()).hexdigest(),
                  'localizer_fingerprint': hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest(),
                  'mAP_protocol': 'positive-only IoU=.50:.05:.95 top10, no hard presence gate; DETR native timestamp postprocessing, no NMS; raw_mAP is original candidates before S4 refinement'}
        if split == 'val':
            candidates = np.round(np.arange(.01, 1, .01), 2)
            accepts = self.score[None, :] > candidates[:, None]
            g = np.round(100 * np.where(accepts, self.local[:, 0][None, :], (~self.y)[None, :]).mean(1), 2)
            ba = .5 * (accepts[:, self.y].mean(1) + (~accepts[:, ~self.y]).mean(1))
            best_g, best_ba = float(candidates[int(g.argmax())]), float(candidates[int(ba.argmax())])
            report.update(best_t_gmiou=best_g, best_t_ba=best_ba, max_gmiou_scan=float(g.max()),
                          val_G_threshold=self.metrics(best_g), val_BA_threshold=self.metrics(best_ba))
        elif calibration is not None:
            report['val_G_calibrated'] = self.metrics(calibration['best_t_gmiou'])
            report['val_BA_calibrated'] = self.metrics(calibration['best_t_ba'])
        else:
            raise ValueError('Test evaluation requires previously saved Val calibration.')
        if all('pred_multi_logits_raw' in p for p, y in zip(self.post, self.y) if y):
            plural = self.count[self.y] >= 2
            logits = np.array([p['pred_multi_logits_raw'] for p, y in zip(self.post, self.y) if y])
            guess = logits.argmax(1)
            probability = torch.softmax(torch.tensor(logits), -1).numpy()[:, 1]
            report['plurality'] = {'accuracy': 100 * float(accuracy_score(plural, guess)),
                                  'balanced_accuracy': 100 * float(balanced_accuracy_score(plural, guess)),
                                  'AUROC': 100 * float(roc_auc_score(plural, probability)),
                                  'confusion': confusion_matrix(plural, guess, labels=[0, 1]).tolist()}
        if comprehensive:
            source = np.array([g.get('dataset_source', 'unknown') for g in self.gt])
            masks = {'no_target': ~self.y, 'single': self.count == 1, 'multi': self.count >= 2}
            masks.update({str(s): source == s for s in np.unique(source)})
            report['subsets_tau05'] = {name: self.metrics(mask=mask) for name, mask in masks.items() if mask.any()}
            report['oracle_presence'] = {name: self.metrics(accept=gate) for name, gate in {
                'always_accept': np.ones(len(self.y), bool), 'learned_presence_tau05': self.score > .5,
                'GT_presence': self.y, 'always_reject': np.zeros(len(self.y), bool)}.items()}
            report['verification'] = self.verify()
        return report

def selection_score(report):
    spectrum = sum(report[key]['G-mIoU@1'] for key in ['gate_04', 'fixed_05', 'gate_06']) / 3
    minimum = min(report['fixed_05']['positive_recall'], report['fixed_05']['negative_recall'])
    return spectrum + .4 * report['val_G_threshold']['G-mIoU@1'] + .5 * report['raw_localizer_pre_exist']['MR-full-mAP'] + .1 * minimum
