# Repository review and reproduction verification

This refactor establishes a SoccerGMR Standard → Standard/Full Test pipeline for five DETR backbones. It does not launch a new full training campaign or claim that bounded integration scores reproduce previous benchmark numbers. Previous results and their exact README are archived under the original protocol.

The implementation is published separately on `soccergmr-standard-geq-full`. The original `gmr-mr-detr-multiroute-20261030` branch is preserved.

## Findings and corrections

| Finding in previous layout | Correction in the supported pipeline |
|---|---|
| Dispatch paths referred to missing deployment files; some GEQ adapters imported a separate checkout and machine-specific training scripts | One owned CLI, explicit builders and transitive native source snapshots; no imports from archive or another project |
| Several families share `models` / `utils` names | One family per process with namespace checks; each verification phase uses a fresh process |
| EaTR had a transitive root-level `misc.py` dependency | Included the file and its source hash |
| GEQ history stopped after six non-improving validations, incompatible with the requested rule | Strict Val three-threshold mean, `stale >= 10`, ties count, persisted selector state; old numbers remain historical |
| Native positive-only localization losses can fail on null queries or all-null batches | Per-backbone null-safe loss adapter; native existence loss on all genuine queries; Sim keeps native null-aware losses |
| CG attention BCE can include padded probabilities | Native loss evaluated over each positive sample's valid video length |
| Bool masks fail in CG arithmetic and Sim CUDA argmax | Common 0/1 float masks, explicit bool conversion inside attention consumers |
| Native QD / CG / Sim temporary tensors used `.cuda()` with no device | Small recorded patches allocate on the input device; original and patched hashes remain in the source manifest |
| Full text qids can collide with Standard qids | Separate roots for all Full features, including text |
| Some Full videos exceed the old 75-token limit | Explicit 128-token capacity, no silent truncation; full asset audit includes long videos |
| Historical data/export conventions differ | One actual-duration target/export contract, documented deterministic saliency targets and native feature ordering; historical numerical equivalence is not asserted |
| Stale or incomplete evaluation output could be mistaken for completion | Exact qid coverage, checkpoint/labels/features/source hashes and checked output markers |
| Duplicate dispatch / restoring best as training state | Directory lock; legal atomic `last.pt` with optimizer, RNG, loader state, history and selector; resumable epoch boundary |
| Test could be mixed into model selection across scattered scripts | Trainer loads only Standard Train/Val; separate Test command requires a completed Val-selected best |

GEQ Full retains a native GMR loss and adds a GEQ existence BCE. GEQ reads projected text followed by each encoder layer's video evidence; QD/CG skip the global token. Sim uses the first encoder pass, consistent with the archived attachment point. Both native and GEQ parameters are trainable. Sim candidate confidence retains native foreground × sigmoid(IoU quality).

## Verification and evidence

Machine-readable evidence is in [`reports/verification`](../reports/verification/). The checks use real precomputed SoccerGMR assets. Each bounded experiment uses two epochs on at most eight Train / eight Val samples, with multiple-window and null queries. Test subsets include long Full videos. These scores are deliberately not promoted to a result table.

- Protocol regression tests cover the exact ten-epoch boundary, ties, counter restoration, mean-versus-max selection, official metric gating and complete qid coverage.
- Data regression tests cover Full `.mp4` stems, separate text roots, a 104-token video, true-duration span/mask targets and rejection of silent truncation or corrupt features.
- Checkpoint regression tests compare optimizer continuation and next-epoch order and reject incompatible identities/stages.
- Real-data checks exercise positive/null and all-null backward passes, GEQ native parameter updates, graceful interruption, resumed versus uninterrupted tensor/selector equality, and both Test settings for both stages.
- A separate clean source copy has no historical archive. Imported project module paths are checked against that copy. Standard Python/site-packages dependencies and external data assets are allowed.
- Full data audit loads the arrays for all 4,138 Train, 465 Val, 1,036 Standard Test and 2,970 Full-Decodable Test queries, hashes 14,169 referenced assets and checks normalized video disjointness. It computes no Test metric.
- Distribution build checks include native source and official evaluator files, rather than relying on paths from an editable checkout.

## Practical limits

Functional reproducibility is verified locally; convergence or numerical reproduction of the archived strong baselines still requires full formal training. Common target coordinates, deterministic saliency labels, shared optimizer defaults and the ten-epoch GEQ rule are explicit new protocol choices. They may change results. Further choices must use Train / Val only; never improve Test scores by changing these settings after evaluation.

Existing Full features cover 2,970/2,986 queries. No result from that manifest can be called a complete 2,986-query Full Test. Raw feature extraction itself is outside this repository's supported trainer and was not rerun. Asset roots and labels must be supplied explicitly; a Git clone does not contain licensed videos or large NPZ files.

Exact resume equality is checked on the same local software/GPU setup. This is not a promise of bitwise equality across PyTorch versions, drivers or GPU architectures. CUDA deterministic mode is requested with warnings for unsupported native operations; the measured equality is the evidence for the tested setup.

Old archived scripts retain historical paths and are not the supported reproduction entry. The current CLI intentionally rejects direct loading of unconverted legacy checkpoints as legal training resumes.
