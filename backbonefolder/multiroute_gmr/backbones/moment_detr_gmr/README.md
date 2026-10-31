# Moment-DETR-GMR

This directory contains the current Moment-DETR-GMR training and evaluation
scripts used for the original six-route ablation. The scripts live in
`gmr_multiroute_kit/` and cover preflight, real-data smoke testing, training,
early-stop monitoring, frozen-best evaluation, and result collection.

The code expects the original project layout. Set `GMR_PROJECT_ROOT` to the
directory containing `generalized-moment-retrieval/`, `soccergmr_standard/`, and
`mr_experiments/` when invoking the copied scripts from another checkout.

```bash
export GMR_PROJECT_ROOT=/path/to/ImprovingGeneralizedMomentRetrieval
python gmr_multiroute_kit/preflight.py \
  --data-root "$GMR_PROJECT_ROOT/soccergmr_standard" \
  --repo "$GMR_PROJECT_ROOT/generalized-moment-retrieval" \
  --out "$GMR_PROJECT_ROOT/mr_experiments/resolved_paths.json"
GPU0=0 GPU1=1 bash gmr_multiroute_kit/launch6.sh
SPLIT=test bash gmr_multiroute_kit/evaluate6.sh
```
