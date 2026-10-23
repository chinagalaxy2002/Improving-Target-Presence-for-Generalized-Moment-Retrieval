# DeRoI-GMR research documentation

Publication note: this is the documentation copy for the GitHub experiment archive. It describes the complete original source workspace; checkpoint binaries and the historical root FlashVTG/A3 inputs are not all included in this publication. Links to available code and labels are mapped to the published Phase 2 folders. Documentation-audit scripts accept `GMR_DOCUMENTATION_SOURCE_ROOT` pointing to the full source workspace; the published five-backbone results can be reproduced using the archive's existing tools.

This package documents the implementation archived as **S4-Fusion / S4_noFusion / S4_minus_Fusion** under the paper name **DeRoI-GMR**. The name mapping is editorial; existing APIs, checkpoints and experiment identities retain their original names. Documentation was prepared on 23 October 2026 from source, annotations, frozen predictions and verified literature.

| Document | Purpose |
| --- | --- |
| [DATA.md](DATA.md) | Annotation schema, feature contract, split statistics, augmentation and data limitations |
| [ALGORITHM.md](ALGORITHM.md) | Active forward path, equations, losses, LoRA, backbone interfaces and implementation qualifications |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Recomputed results, matched baselines, checkpoint protocols and claim limits |
| [RESEARCH_GAP.md](RESEARCH_GAP.md) | What the supplied AAAI paper establishes, prior temporal work and the defensible research gap |
| [LITERATURE.md](LITERATURE.md) | Focused literature review and independently checked references |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Commands, prerequisites and verification scope |
| [PAPER_SECTIONS.md](PAPER_SECTIONS.md) | Evidence-grounded abstract, methods, experimental results and limitations prose |
| [research/WORKFLOW.md](research/WORKFLOW.md) | ResearchPilot lifecycle execution, fallback disclosures and future evidence gates |

The [generated tables](research/generated/) are rebuilt by [audit_repository.py](research/audit_repository.py). The [evidence cards](research/evidence_cards.json), [research memory](research/thesis_memory.json), [BibTeX](research/references.bib), [claim map](research/working/evidence_map.json) and [independent citation review](research/citation_verification.json) keep prose and supporting evidence separate.

The evidence supports a particular result: at the archived τ=0.4 operating point, the five implementations improve generalized retrieval and ungated localization mAP relative to their local GMR baselines, while positive acceptance falls. It does not establish uniform improvement in rejection ranking, causal benefits of individual components, statistical robustness, match-disjoint generalization or lower runtime.

This is a documentation and analysis release. No training, checkpoint inference, dataset relabeling or model changes were performed for it. Existing experiments and archival figures retain their identities. The paper sections are a supported partial manuscript draft, not a submission-ready full paper.
