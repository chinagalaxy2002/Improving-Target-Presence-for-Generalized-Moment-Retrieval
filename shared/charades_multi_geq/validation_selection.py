"""User-selected Val G-mIoU operating points, with no Test-based selection."""
from __future__ import annotations

import math

THRESHOLDS = ("0.4", "0.5", "0.6")


def values(reports, metric="G-mIoU@1"):
    result = [float(reports[threshold]["brief"][metric]) / 100 for threshold in THRESHOLDS]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in result):
        raise ValueError("Expected official percentage metrics convertible to 0–1")
    return result


class ValidationSelector:
    def __init__(self, baseline_reports, metric="G-mIoU@1", patience=5):
        self.metric = metric
        self.patience = patience
        self.baseline_values = values(baseline_reports, metric)
        self.baseline_mean = math.fsum(self.baseline_values) / 3
        self.best_mean = None
        self.best_epoch = None
        self.best_all_above = False
        self.stale_epochs = 0

    def update(self, epoch, reports):
        current = values(reports, self.metric)
        score = math.fsum(current) / 3
        all_above = all(a > b for a, b in zip(current, self.baseline_values))
        improved = self.best_mean is None or score > self.best_mean
        tie_better = (self.best_mean is not None and score == self.best_mean
                      and all_above and not self.best_all_above)
        if improved:
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        if improved or tie_better:
            self.best_mean, self.best_epoch, self.best_all_above = score, epoch, all_above
        return dict(epoch=epoch, thresholds=list(THRESHOLDS), metric=self.metric,
            values=current, mean=score, baseline_values=self.baseline_values,
            baseline_mean=self.baseline_mean, mean_gain=score - self.baseline_mean,
            per_threshold_gain=[a - b for a, b in zip(current, self.baseline_values)],
            all_three_above_baseline=all_above, save_best=improved or tie_better,
            best_epoch=self.best_epoch, best_mean=self.best_mean,
            best_all_three_above_baseline=self.best_all_above, stale_epochs=self.stale_epochs,
            stop=self.stale_epochs > self.patience,
            stop_rule="strictly more than 5 consecutive epochs without mean improvement")

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self, state):
        if state["metric"] != self.metric or state["baseline_values"] != self.baseline_values:
            raise ValueError("Selection protocol changed during resume")
        self.__dict__.update(state)
