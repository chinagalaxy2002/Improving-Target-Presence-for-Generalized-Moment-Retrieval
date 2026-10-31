# EaTR-GMR

This folder is the EaTR-specific entry point for the six Mr. DETR-style
multi-route variants. The executable implementation is shared from
`../../run.py` and `../../multiroute.py`.

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --variant baseline --mode smoke
CUDA_VISIBLE_DEVICES=0 python run.py --variant baseline --mode train
CUDA_VISIBLE_DEVICES=0 python run.py --variant baseline --mode test
```

Training automatically resumes from `../../runs/eatr/<variant>/last.ckpt` and
testing always loads the frozen `best.ckpt`.

Exception: `baseline` reuses the completed seed2026, batch32 GMR checkpoint;
smoke/train commands detect the reuse marker and skip new training. Other
EaTR variants keep their seed2023, microbatch2/accumulation4 runs. Check the
[checkpoint inventory](../../../../README.md#10-全部检查点与实验目录)
for existing/expected files and source provenance.
