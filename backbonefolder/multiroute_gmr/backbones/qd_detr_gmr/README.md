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
