# QD-DETR-GMR

This folder is the QD-DETR-specific entry point for the six Mr. DETR-style
multi-route variants. The executable implementation is shared from
`../../run.py` and `../../multiroute.py`.

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --variant baseline --mode smoke
CUDA_VISIBLE_DEVICES=0 python run.py --variant baseline --mode train
CUDA_VISIBLE_DEVICES=0 python run.py --variant baseline --mode test
```

Training automatically resumes from `../../runs/qd_detr/<variant>/last.ckpt`
and testing always loads the frozen `best.ckpt`.

Queued `instruction`, `multi`, and `adaptive` start afresh at batch32 and
accumulation1, as specified by `../../execution_plan.json`. Their old run
directories are archived; baseline/shared/sepffn retain their completed runs.
See the [full checkpoint inventory](../../../../README.md#10-全部检查点与实验目录).
