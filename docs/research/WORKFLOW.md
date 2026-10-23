# ResearchPilot execution record and lifecycle handoff

## Scope

The requested endpoint is repository data and algorithm documentation, informed by the supplied reference, README experiments, literature and source. The delivered prose is English. A supported abstract and manuscript sections are included; no full Introduction or complete submission package is represented as finished. Evidence gaps remain explicit in working files rather than appearing as placeholders in paper prose.

## Skills actually applied

| Stage | Component read and applied | Result |
| --- | --- | --- |
| Routing | `ResearchPilot-Skills/skills/thesis-research-router/SKILL.md` | Tier 1 public APIs and local drafting selected |
| Evidence search | `thesis-openalex-search` and its bundled Python helper | Four broad searches, then three focused title-filter searches |
| Citation metadata | `thesis-crossref-validate` and helper | Title searches plus direct DOI calls; failures retained |
| Literature expansion | `thesis-openalex-expand` | Explicit OpenAlex seed/forward/backward attempt; no usable graph links; GMR paper reference list used for backward expansion |
| Paper explanation | `thesis-scispace-explain` local-text fallback | Supplied PDF explained directly; SciSpace not used |
| Evidence-based drafting | `thesis-jenni-draft` local fallback | Accepted cards feed manuscript sections; Jenni not used |
| Scientific prose/evidence gate | `/home/guoxiangyu/.codex/skills/paper-writer/SKILL.md` | Evidence map, chapter plan, grounded prose and independent citation check |

The local ResearchPilot SKILL.md files and their quality-check references were read. Their tool-specific `Skill(...)` examples were translated to filesystem instructions and available tools; no nonexistent Skill tool was claimed. No platform account, paid action or private text upload was needed. The independent citation sub-agent was used because `paper-writer` explicitly requires it for three or more citations.

## Investigation and evidence decisions

Research questions were narrowed to the implemented readout/refinement distinction, positive retention, multi-moment behavior, source effects, and reproducibility. The supplied paper was identified as generalized **image segmentation**. The source-level review found that the active repository variant bypasses local candidate-to-presence fusion, but retains a legacy FlashVTG output-score scaling step. It also confirmed disabled plurality supervision and the absence of the native VMR loss in adaptation.

The four broad OpenAlex searches returned 32 off-topic hits, which were excluded rather than presented as relevant evidence. Focused searches removed `type:article` and used `title.search`, finding the benchmark and AAAI reference. Metadata-only adjacent work remains in `maybe_papers`. Exact title/author/version checks excluded near matches, including a later differently titled CG-related journal paper and unrelated top Crossref hits for GMR/CVA. The final 13 kept references came from primary-source checks, including backward references in the GMR paper.

Crossref title searches and DOI calls partially encountered HTTP 429. Successful metadata and failures are archived. arXiv-issued DOIs are registered through DataCite, so absence in Crossref is not treated as invalidity. The independent check used primary papers/publisher material and resolved the relevant identity/version issues. Citation counts are retained only where returned and are not used as proof of a claim, endorsement, or methodological quality. `supporting` and `contrasting` counts are null; Scite was not used.

The OpenAlex seed metadata exposed no usable reference/forward links in this run. The graph attempt is preserved; there is no fabricated citation network. Reference-list expansion independently identified generalized VMR, open-set VMR, negative-aware VMR and multi-moment retrieval precedents. The review is focused and cannot establish exhaustive novelty of Gaussian temporal refinement.

## Analysis and drafting artifacts

The offline audit derived split/source/cardinality counts and overlap diagnostics from the original JSONL files. It recomputed frozen-prediction metrics and emitted CSV/JSON tables with file hashes. No new annotations, feature values, predictions or experimental runs were synthesized.

`evidence_cards.json` and `thesis_memory.json` follow ResearchPilot's evidence-card/paper/memory schemas. Accepted source claims are limited by the independent review. `references.bib` is produced using the bundled exporter, then assigned preprint/conference entry types according to the cited source form. `working/evidence_map.json` maps local assertions to code and results; `working/chapter_blueprint.json` records section roles; `draft_handoff.json` provides the local equivalent of the evidence-to-draft contract without pretending that Jenni was opened.

The prose separates implemented mechanisms, recorded results, cross-task motivation and untested causal claims. The revised abstract reports the positive-acceptance tradeoff and nonuniform AUROC. A3 remains separate and exploratory. The original README's .4/.5 narrative ambiguity is resolved in the new tables without rewriting historical results.

## Evidence gates for the next research cycle

| Lifecycle stage | Existing artifact to update | Evidence required before strengthening claims |
| --- | --- | --- |
| Literature update | Evidence cards and raw retrievals | Primary text for each new mechanism/comparison; rerun independent citation review after substantive changes |
| Data preparation | Generated statistics and source manifest | Full feature availability/shape/order preflight; preserve all empty GT and full multi-GT metadata |
| Hypothesis testing | Experiment registry and archived config | Controlled query/refiner/LoRA/fusion and uniform-pooling comparisons with matched budgets/selectors |
| Confirmatory evaluation | New frozen protocol | Multiple seeds, uncertainty, predeclared thresholds and a fresh untouched holdout after Test diagnostics |
| Error analysis | Source/cardinality metrics and paired predictions | Positive/multi-target end-to-end recall, within-WorldCup discrimination and matched-specificity/recall comparisons |
| Efficiency evaluation | Measured resource report | Training/adaptation cost, inference latency, memory and FLOPs on specified hardware |
| Final paper | Manuscript sections and reference library | Final supported experimental scope, figures/tables, venue formatting and complete submission review |

These rows are a handoff for new evidence collection. They are not claims that missing experiments have already been executed. The current documentation objective is complete when the listed files, generated tables, citation review and reproducibility checks are available and consistent.
