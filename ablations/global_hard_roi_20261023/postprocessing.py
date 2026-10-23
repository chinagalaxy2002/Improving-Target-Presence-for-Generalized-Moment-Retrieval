"""Native two-second grid export with positive-width, duration-bounded windows."""
import math
import torch


def windows(seconds, duration):
    values = seconds.detach().cpu().clone()
    assert values.ndim == 2 and values.shape[-1] == 3
    assert math.isfinite(duration) and duration > 0 and torch.isfinite(values).all()
    values[:, :2] = (values[:, :2] / 2).round() * 2
    values[:, :2].clamp_(0, duration)
    # Rounding can collapse even a valid one-second proposal. Repair only these
    # degenerate windows; all other native rounded coordinates stay unchanged.
    for row in values:
        if row[1] <= row[0]:
            if row[1] > 0:
                row[0] = max(0., float(row[1]) - min(2., duration))
            else:
                row[0] = 0.; row[1] = min(2., duration)
    assert ((values[:, 0] >= 0) & (values[:, 1] > values[:, 0]) & (values[:, 1] <= duration)).all()
    assert ((values[:, 2] >= 0) & (values[:, 2] <= 1)).all()
    return values[values[:, 2].argsort(descending=True)].tolist()


def export(output, metas):
    centers, widths = output['pred_spans'].unbind(-1)
    durations = centers.new_tensor([m['duration'] for m in metas])
    xy = torch.stack([centers - widths / 2, centers + widths / 2], -1) * durations[:, None, None]
    xy = torch.minimum(xy.clamp_min(0), durations[:, None, None])
    native = torch.cat([xy, output['pred_logits'].softmax(-1)[..., :1]], -1)
    predictions = []
    for i, meta in enumerate(metas):
        refined = windows(output['candidate_seconds'][i], float(meta['duration']))
        raw = windows(native[i], float(meta['duration']))
        logit = float(output['pred_exist_logits'][i].detach())
        assert math.isfinite(logit) and len(refined) == len(raw) == 10
        predictions.append({'qid': meta['qid'], 'vid': meta['vid'], 'query': meta['query'],
                            'pred_relevant_windows': refined, 'pred_relevant_windows_pre_exist': raw,
                            'pred_exist_logit_raw': logit,
                            'pred_exist_score': float(torch.sigmoid(output['pred_exist_logits'][i]).detach())})
    return predictions
