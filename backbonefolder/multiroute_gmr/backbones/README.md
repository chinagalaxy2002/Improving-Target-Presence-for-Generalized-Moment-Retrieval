# Per-backbone entry points

Each backbone has an independent folder. EaTR, QD-DETR, and CG-DETR use the
same audited multi-route core in `../run.py` and fix the backbone argument in a
small local entry point. This avoids maintaining three divergent copies of the
model, criterion, early-stop, and frozen-best test implementation.

| Folder | Implementation |
|---|---|
| `moment_detr_gmr/` | Original Moment-DETR-GMR six-route scripts |
| `eatr_gmr/` | EaTR-GMR smoke/train/test entry point |
| `qd_detr_gmr/` | QD-DETR-GMR smoke/train/test entry point |
| `cg_detr_gmr/` | CG-DETR-GMR smoke/train/test entry point with worker-safe CPU collate |

For the three migrated backbones:

```bash
CUDA_VISIBLE_DEVICES=0 python eatr_gmr/run.py --variant baseline --mode smoke
CUDA_VISIBLE_DEVICES=0 python eatr_gmr/run.py --variant baseline --mode train
CUDA_VISIBLE_DEVICES=0 python eatr_gmr/run.py --variant baseline --mode test
```

Valid variants are `baseline`, `shared`, `sepffn`, `instruction`, `multi`, and
`adaptive`. Training uses seed 2023 and `num_workers=2`. Default microbatch2 ×
accumulation4 is overridden by `../execution_plan.json` for queued QD/CG
instruction, multi, adaptive (batch32 × accumulation1). Only EaTR baseline
reuses a completed seed2026 checkpoint. Validation occurs every five epochs;
early stopping occurs after 20 epochs without a new best validation-mAP checkpoint.

All absolute and expected checkpoint paths, result folders and complete
reproduction commands are listed in the branch-root
[README](../../../README.md#10-全部检查点与实验目录).
