# Reproduction and verification

## Documentation tables: CPU, annotations and saved predictions

From the repository root:

```bash
python docs/research/audit_repository.py
python docs/research/build_evidence.py
```

The audit needs NumPy and the local evaluator. In the current workspace it was run successfully with `/home/guoxiangyu/miniconda3/envs/gmr/bin/python`; use the corresponding project Python environment on another machine. It writes only `docs/research/generated/`. Evidence export uses the bundled ResearchPilot BibTeX exporter and the saved independent review; it does not run another network search or claim to repeat independent verification.

The documentation scripts can also be used from a publication-only checkout by setting `GMR_DOCUMENTATION_SOURCE_ROOT=/path/to/full-source-workspace`. This selects the original annotation, prediction and ResearchPilot inputs while outputs stay beside the scripts. The GitHub publication archive has its own `tools/reproduce_results.py`; it does not contain every historical root-level FlashVTG/A3 input used by the documentation audit.

The audit validates annotation intervals and identifiers, reports source/cardinality/split statistics, checks complete Test prediction coverage, and recomputes official G-mIoU@1/3/5, AUROC, Rej-F1 and ungated mAP at .4/.5/.6 for the three root models. It also checks the five fixed-epoch adaptation outputs at .4 against the archive table. Baseline comparison values are read from saved baseline curve summaries, explicitly marked as such.

All these assertions passed for this documentation release. See [verification.json](research/generated/verification.json). The archived five-backbone `tools/reproduce_epoch4.py --metrics-only` was also run separately; each backbone's complete `brief` dictionary matched its archived report. That command generated `/tmp/deroi_documentation_epoch4_recheck_20261023/` in this session.

## Archived five-backbone metrics

Use an entirely new output directory on each run:

```bash
cd 'Phase2(addbackbone_with_S4_minusFusion)'
python tools/reproduce_epoch4.py --metrics-only --output /tmp/deroi_epoch4_metrics_new
```

This uses `shared/labels/test.jsonl`, archived predictions and the original evaluator. It does not need a GPU, original video features or checkpoint inference. Add `--backbone qd_detr` to restrict the run. Omitting `--metrics-only` also redraws figures into the new output directory; it is unnecessary to validate the documentation tables.

## Public literature metadata

```bash
python docs/research/collect_literature.py
```

This invokes public Crossref/OpenAlex APIs. It saves exact URLs, returned data and failures into `docs/research/retrieval/`. It sends only public research titles and identifiers. An HTTP 429, missing metadata, or an empty citation graph is retained as a limitation, never converted into a successful validation. Search results require relevance and title/version checks before entering evidence cards.

The original broad queries were executed with the supplied `ResearchPilot-Skills/scripts/api/openalex_search.py`. The focused collector removes the broad script's `type:article` restriction and uses `title.search`. The existing expansion helper silently suppresses individual fetch failures, so the collector records graph calls explicitly and uses `referenced_works` IDs rather than relying on an unverified `/references` endpoint. In this session the graph returned no usable links; verified GMR references supplied the backward expansion instead.

The original broad-search snapshots are retained for provenance; the collector refreshes focused search/validation inputs. The independent citation report is a separate review artifact and is not recreated by a metadata download.

## Model inference and fresh training require external features

Configure the root FlashVTG API with real asset paths:

```bash
python code/configure_paths.py \
  --assets-dir /path/to/Soccergmr \
  --baseline-checkpoint /path/to/E0_s2026/best.ckpt \
  --s4-fusion-checkpoint /path/to/S4_noFusion/best.ckpt
python code/preflight.py --full-dataset
python code/evaluate_checkpoint.py --model E0 --device cuda:0 --output runs/eval_E0_new
python code/evaluate_checkpoint.py --model S4Fusion --device cuda:0 --output runs/eval_S4_new
```

The API validates the archived checkpoint identities; new output directories prevent accidental replacement. The root API supports exactly E0 and S4Fusion. It is not the five-backbone training entry point.

If a new root training experiment is required, the existing commands are:

```bash
python code/train.py --model E0 --run-name E0_new --device cuda:0
python code/train.py --model S4Fusion --run-name S4_new --device cuda:0
```

These root entries train on Train and select on Val. The Phase 2 six-epoch trainer is a source snapshot with historical absolute paths and explicit epoch 3–6 Test diagnostics. Port its paths and preserve its initialization/feature/adapter contracts before attempting new runs; documentation-table reproduction does not validate its portability.

A3 has an additional matching CLIP projection prerequisite:

```bash
python code/test_qcd.py
python code/build_qcd_cache.py --clip-model /path/to/ViT-B-32.pt
python code/train_qcd.py --run-name A3_new --device cuda:0 --epochs 10
```

Its default entry uses Train/Val. `--test-from 4` explicitly enables repeated Test diagnostics and changes the evaluation status of the resulting study. It is not needed for a new Train/Val run.

## What was verified in this session

The work read the local reference PDF, active code, configurations, annotations, predictions and provenance; retrieved public literature; generated documentation and analysis tables; recomputed prediction-based metrics; and independently checked 13 citations. It did not execute training, inference from binary checkpoints, large feature-bank preflight, QCD cache construction or runtime benchmarking. Earlier preflight reports remain historical evidence, not newly executed checks.

No commit, push, publication, external document upload or change to training/model behavior is part of this release. Source hashes are in [source_manifest.json](research/source_manifest.json); the supplied reference PDF remains at its original local path and is not redistributed in this documentation.
