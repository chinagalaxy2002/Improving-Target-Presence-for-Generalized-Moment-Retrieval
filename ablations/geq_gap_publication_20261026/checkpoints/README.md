# Checkpoint downloads

Large checkpoint binaries are published as GitHub Release assets, not Git blobs:

https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/releases/tag/geq-gap-flashvtg-20261026

Use ../CHECKPOINT_MANIFEST.json for direct links, byte counts and SHA256. The fine-tuned checkpoint is margin0.6/weight2/epoch4, selected by Val AUROC. Compact inference weights contain the same model tensors as the full optimizer checkpoint. The original GMR initialization is also supplied for fresh independent fine tuning. Offline feature tensors are not included.
