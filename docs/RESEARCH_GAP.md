# Research problem and defensible gap

## What the supplied AAAI paper actually studies

The local reference `/home/guoxiangyu/10554-AAAI26.KimN-CV.pdf` is **Improving Target Presence and Plurality Recognition for Generalized Referring Image Segmentation**, by Namyup Kim, Jinsung Lee and Suha Kwak, AAAI 2026, pp. 5700–5708. It studies image segmentation, with zero, one or multiple referred objects. The [publisher record](https://ojs.aaai.org/index.php/AAAI/article/view/37490) independently confirms its identity.

Its Introduction argues that a presence head receiving mask-prediction features forces those features to serve segmentation and presence classification, potentially limiting presence recognition. Its proposed target query reads intermediate pixel-decoder features. A text-conditioned plurality head supplies an additional training signal, and synthetic no-target/multi-target data address the paper's dataset imbalance. Sections 3 and 4 evaluate these choices in generalized RIS. These are findings and motivations within image segmentation; they do not prove the same failure mechanism in GMR.

The present repository adopts the dedicated global-reader idea in a temporal setting and adds Gaussian temporal RoI refinement with LoRA. It does **not** activate the reference paper's plurality objective or reproduce its image/text stitching and within-batch synthetic no-target augmentation. A3 is a separate background-replacement adaptation inspired by CVA.

## What temporal prior work already covers

[Qin et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/7ac19fdcdf4f311f3e3ef2e7ef4784d7-Paper-Conference.pdf) already study no-target and one-to-many temporal retrieval with NExT-VMR and BCANet. [OpenVMR](https://dl.acm.org/doi/10.1145/3664647.3680947), published at ACM MM 2024, studies rejection of video-irrelevant queries; its 2026 arXiv upload is not its original publication date. [Moment of Untruth](https://arxiv.org/abs/2502.08544) studies negative-aware retrieval with in-domain and out-of-domain negatives. [FlashMMR](https://arxiv.org/abs/2510.17218) addresses multiple moments. Absent-query handling, multiple occurrences, and learned queries are therefore not new by themselves.

[Ding et al.](https://arxiv.org/abs/2605.02623) provide Soccer-GMR and a unified evaluation protocol. Their §4.1 presence adapter pools query-conditioned cross-modal localization embeddings and applies a two-layer MLP. For backbones without decoder slots, it uses an equivalent cross-modal representation. It does not generally require explicit proposed endpoints as presence inputs. The root FlashVTG E0 also uses its specific configured pooling head; one should not describe every baseline as an identical proposal-level classifier.

## Research question supported by the implementation

**Can a dedicated global event query over text and intermediate video features improve generalized temporal retrieval when temporal proposals are refined through a separate query-conditioned path?**

The architectural hypothesis is that a dedicated readout gives existence prediction a representation explicitly trained for that role, while a separate RoI path can adjust temporal boundaries. This is motivated by the reference paper and by the baseline's representation reuse. It remains a hypothesis about the cause of performance; the current complete-system comparisons do not isolate representation conflict, poor proposals, or gradient competition.

The precise design distinction is **no direct candidate evidence in the global presence logit**, together with **proposal-conditioned refinement of localization**. Both branches share backbone features and LoRA parameters. The system is not statistically independent, gradient-isolated, or proposal-free. The FlashVTG exporter also retains a legacy uniform score multiplier, so the claim must not expand to absence of all existence-related candidate-score processing.

## Contribution boundaries

| Proposed claim | Evidence-grounded wording |
| --- | --- |
| “We introduce generalized/no-target/multi-moment retrieval.” | We study the established GMR setting on Soccer-GMR. |
| “All previous existence heads depend on unreliable proposals.” | The studied GMR adapter reuses localization-oriented cross-modal embeddings; the proposed global readout excludes explicit candidate inputs. |
| “This feature sharing causes GMR errors.” | This motivates our design; a causal diagnosis remains to be tested. |
| “A novel learnable global query.” | A temporal adaptation of dedicated target-query reasoning; distinguish it from Kim, learned moment queries and event-aware queries. |
| “Fully decoupled verification and localization.” | Separate forward readouts sharing backbone features and LoRA updates. |
| “Backbone-agnostic and seamless.” | Implemented with backbone-specific interfaces on five representative architectures. |
| “Consistently improves rejection.” | Improves fixed-.4 rejection and G-mIoU in these runs; AUROC does not improve for every backbone. |
| “Maintains excellent temporal localization.” | Ungated mAP increases in the five archived comparisons; positive acceptance falls. |
| “Lightweight / efficient.” | Fewer trainable parameters for FlashVTG adaptation; no measured latency or memory claim. |
| “Solves plurality recognition.” | Supports multi-moment outputs; explicit plurality supervision is disabled. |

The literature investigation is focused rather than exhaustive. No “first,” “unique,” or universal Gaussian-RoI novelty claim is established. Further comparison against temporal Gaussian weighting and refinement methods is required before claiming the operation itself is novel. The strongest present contribution is the specific architecture, implementation across five backbones and transparently reported operating-point behavior.
