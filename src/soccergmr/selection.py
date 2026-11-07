"""A strict validation-only selector; ties count as non-improvements."""
from dataclasses import dataclass, asdict
import math


@dataclass
class Selector:
    metric: str
    patience: int
    mode: str = 'min'  # 'min' for Orc-gap (lower is better), 'max' for mAP (higher is better)
    best: float = float('inf')
    best_epoch: int = 0
    stale: int = 0

    def __post_init__(self):
        if self.best == float('inf') and self.mode == 'max':
            self.best = float('-inf')
        elif self.best == float('-inf') and self.mode == 'min':
            self.best = float('inf')

    def update(self, epoch, score):
        if not math.isfinite(score):
            raise ValueError('Nonfinite validation selection score')
        if self.mode == 'min':
            improved = score < self.best
        else:
            improved = score > self.best
        if improved:
            self.best, self.best_epoch, self.stale = float(score), epoch, 0
        else:
            self.stale += 1
        return dict(epoch=epoch, score=float(score), best=self.best, best_epoch=self.best_epoch,
                    improved=improved, stale=self.stale, stop=self.stale >= self.patience,
                    metric=self.metric, patience=self.patience, mode=self.mode)

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state(cls, state):
        return cls(**state)


def score(stage, metrics):
    if stage == 'gmr':
        return float(metrics['brief']['mAP'])
    if stage == 'gmr_geq':
        return float(metrics['orc_gap']['gap_pct'])
    if stage == 'geq_full':
        return sum(metrics['gmiou'][str(t)]['G-mIoU@1'] for t in (0.4, 0.5, 0.6)) / 3
    raise ValueError(f"Unknown stage: {stage}")
