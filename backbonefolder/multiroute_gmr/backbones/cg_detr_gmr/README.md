# CG-DETR-GMR

This folder is the CG-DETR-specific entry point for the six Mr. DETR-style
multi-route variants. The shared runner contains the worker-safe CPU collate
fix and transfers batches to CUDA in the main process before inference.

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --variant baseline --mode smoke
CUDA_VISIBLE_DEVICES=0 python run.py --variant baseline --mode train
CUDA_VISIBLE_DEVICES=0 python run.py --variant baseline --mode test
```

Training automatically resumes from `../../runs/cg_detr/<variant>/last.ckpt`
and testing always loads the frozen `best.ckpt`.

Active baseline/shared/sepffn keep microbatch2/accumulation4. Queued
instruction/multi/adaptive restart at batch32/accumulation1 with a new full-batch
smoke gate; old checkpoints are archived. See `../../execution_plan.json` and
the [full checkpoint inventory](../../../../README.md#10-全部检查点与实验目录).
