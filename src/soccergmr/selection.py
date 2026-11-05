"""A strict validation-only selector; ties count as non-improvements."""
from dataclasses import dataclass, asdict
import math


@dataclass
class Selector:
    metric: str
    patience: int
    best: float = float('-inf')
    best_epoch: int = 0
    stale: int = 0

    def update(self, epoch, score):
        if not math.isfinite(score):
            raise ValueError('Nonfinite validation selection score')
        improved = score > self.best
        if improved:
            self.best, self.best_epoch, self.stale = float(score), epoch, 0
        else:
            self.stale += 1
        return dict(epoch=epoch, score=float(score), best=self.best, best_epoch=self.best_epoch,
                    improved=improved, stale=self.stale, stop=self.stale>=self.patience,
                    metric=self.metric, patience=self.patience)

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state(cls, state):
        return cls(**state)


def score(stage, metrics):
    if stage == 'gmr':
        return float(metrics['brief']['mAP'])
    if stage != 'geq_full':
        raise ValueError(stage)
    return sum(metrics['gmiou'][str(t)]['G-mIoU@1'] for t in (0.4,0.5,0.6))/3
