# Multi-Route GMR on EaTR, QD-DETR, and CG-DETR

Per-backbone entry points are organized under [`backbones/`](backbones/):
`moment_detr_gmr`, `eatr_gmr`, `qd_detr_gmr`, and `cg_detr_gmr`. The three new
backbones call this directory's shared audited runner; Moment-DETR-GMR keeps its
original experiment kit inside its own folder.

This directory migrates the training-only Mr. DETR-style routes used in the
Moment-DETR-GMR experiments to three additional backbones.  It runs the same six
single-seed (`2023`) configurations for every backbone:

| Variant | O2M | Separate FFN | Instruction tokens | Adaptive matching |
|---|---:|---:|---:|---:|
| `baseline` | no | no | no | no |
| `shared` | yes | no | no | no |
| `sepffn` | yes | yes | no | no |
| `instruction` | yes | no | 4 | no |
| `multi` | two routes | yes | 4 | no |
| `adaptive` | two routes | yes | 4 | tIoU >= 0.25 |

The main route retains 10 queries and native Hungarian O2O supervision. Each
training-only auxiliary route has 24 queries, `K<=2`, and the total O2M loss
weight is always 0.3 (0.15 per route for `multi`/`adaptive`). The native
GMR-Adapter and existence BCE remain active. Null queries receive existence and
auxiliary background classification supervision but no span regression.

EaTR uses Slot Attention instead of a DAB query embedding. Its auxiliary route
therefore uses 24 learned auxiliary slots while sharing the native Slot
Attention projections and update MLP. QD-DETR and CG-DETR use 24 learned DAB
reference queries. All backbones share decoder SA/CA; `sepffn` owns only
`linear1`, `linear2`, and post-FFN `norm3`. Instruction tokens are inserted only
inside decoder self-attention and are discarded before cross-attention.

Training protocol: 400 epochs maximum, learning rate `5e-5`, microbatch 2,
gradient accumulation 4 (effective batch 8), validation every 5 epochs, early
stop when 20 consecutive epochs pass without improving the validation-mAP best
checkpoint, and checkpoint selection by validation mAP. Every task uses
`num_workers=2`, pinned memory, two-batch prefetching, and
persistent workers. CG-DETR uses a worker-safe CPU collate function and transfers
the batch to CUDA only in the main process. Test inference occurs once after
freezing `best.ckpt`.

Launch the recoverable two-GPU queue:

```bash
GPU0=0 GPU1=1 bash backbonefolder/multiroute_gmr/launch_queue.sh
```

GPU 0 queues `baseline`, `sepffn`, and `multi`; GPU 1 queues `shared`,
`instruction`, and `adaptive`. Each GPU has two queue workers, so it runs at
most two tasks concurrently. Every job must pass its real-data smoke gate before
training, then automatically tests the frozen best checkpoint. A stopped job
resumes from `last.ckpt` when the launcher is run again. To validate all 18
configurations before launching the queue, run `python validate_all.py`.
Each smoke/train/test stage is retried up to three times. A terminal queue with
failed jobs is labelled `COMPLETED_WITH_FAILURES` and lists `failed_jobs`; it is
never reported as a clean completion.

Monitor and collect:

```bash
cat backbonefolder/multiroute_gmr/QUEUE.json
tail -f backbonefolder/multiroute_gmr/logs/queue.log
python backbonefolder/multiroute_gmr/collect_results.py
```

The official evaluator returns percentages. Each evaluation JSON preserves that
canonical `brief` block and additionally writes `G-mIoU_0_1` plus `brief_0_1`,
where every G-mIoU@k is divided by 100 into the requested full 0-1 domain.
`collect_results.py` uses this 0-1 representation for G-mIoU while retaining
the official percentage convention for the other metrics.
