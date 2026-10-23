# DeRoI-GMR: Decoupled Event Verification and Gaussian Temporal RoI Refinement for Generalized Moment Retrieval

## Abstract

Generalized Moment Retrieval (GMR) localizes all temporal segments matching a natural-language query while allowing an empty output when the event is absent. Existing GMR adapters can estimate existence from representations also used for localization, motivating a dedicated verification readout. We present DeRoI-GMR, a framework combining global event verification with a separate temporal refinement path. A learnable event query reads textual representations and intermediate video features to predict existence without directly consuming candidate proposals. In parallel, a Gaussian temporal Region-of-Interest module pools proposal-centered video features and performs text-conditioned boundary and confidence refinement. Rank-eight Low-Rank Adaptation updates the video encoder while retaining frozen backbone weights. We implement the framework with Moment-DETR, QD-DETR, CG-DETR, EaTR and FlashVTG on Soccer-GMR. In the archived single-seed comparisons at an existence threshold of 0.4, G-mIoU@1 improves by 10.84–46.09 percentage points and ungated localization mAP increases by 0.41–1.56 points relative to the corresponding local GMR baselines. These gains accompany lower positive-query acceptance, and AUROC does not improve for every backbone. The results demonstrate the framework's applicability across the evaluated architectures while identifying positive-event retention as an unresolved limitation.

## Method

### Task and architecture

Given a video and a text query, GMR returns the set of corresponding temporal intervals, including an empty set for an absent event [1]. We implement separate forward readouts for existence and temporal refinement. The existence readout is inspired by dedicated target-query reasoning for generalized referring image segmentation [2], adapted here to text and intermediate temporal representations. The two paths share backbone features and low-rank updates; the separation concerns the inputs to the existence logit rather than independent optimization.

A learnable 256-dimensional event token first attends to projected text and then sequentially reads the intermediate video encoder layers. Each read uses masked cross-attention with residual normalization and a feed-forward update. A linear head maps the final token to a scalar logit, whose sigmoid gives the existence probability. Candidate coordinates, candidate confidences and pooled RoI representations are excluded from this readout. The number of video reads follows the backbone's encoder depth.

### Gaussian temporal RoI refinement

For each of up to eight high-confidence native proposals, we pool final video features using a Gaussian centered at the interval midpoint. Its standard deviation is half the proposal duration, with a minimum duration of two seconds. Padded clips receive zero weight, and the remaining weights are normalized. A geometry embedding encodes normalized start, end, duration and confidence and is added to the pooled feature.

The resulting moment token attends to the text representation. Geometry and six local saliency descriptors provide additional cues for predicting start, end and confidence residuals. A tanh output bounds temporal changes to 1.5 seconds and confidence changes to 0.3. The output layer is initialized to zero. The refined candidates replace their selected native counterparts, while the remaining candidates are retained. Backbone-specific timestamp processing and ranking are preserved.

The global existence logit does not fuse candidate-level evidence. The archived FlashVTG exporter separately retains its original uniform existence-based confidence scaling; this export operation does not supply candidate evidence to the existence reader.

### Adaptation and objective

We apply LoRA [3] with rank eight and scaling parameter 16 to both feed-forward linear layers in each video encoder layer. Original backbone weights remain frozen. The event reader, RoI encoder, refiner and low-rank matrices are trainable, and captured video features retain gradients. Thus both losses can update the shared low-rank representation.

The adaptation objective is binary cross-entropy for existence on all query–video pairs plus a refinement loss weighted by 0.5. For positive queries, each annotated interval is assigned its highest-overlap refined candidate. The refinement loss combines normalized Smooth L1 endpoint losses, temporal generalized IoU and an IoU-quality confidence target. The assignment is not constrained to be one-to-one. Negative queries do not incur refinement loss. The active objective does not include a plurality loss, candidate-presence fusion or an additional native localization criterion.

### Optional context diversification

We separately investigate an adaptation of CVA's Query-aware Context Diversification [4]. This A3 variant replaces query-screened background features in WorldCup Train positive examples while protecting the complete GT union and adjacent clips. SlowFast and CLIP features are replaced synchronously, whereas the recipient's temporal encoding, query and labels are retained. The main five-backbone results do not use this augmentation.

## Experimental setting

We use the Soccer-GMR Standard partition: 4,138 training queries, 465 validation queries and 1,036 test queries. The test set contains 492 no-target, 384 single-target and 160 multi-target queries. The model consumes fixed SlowFast and CLIP video features, temporal endpoint features and CLIP text representations. The adaptation seed is 2026.

The FlashVTG baseline is selected by native pre-NMS validation mAP, while its adaptation is selected by the archived validation composite criterion. The additional Moment-DETR, QD-DETR, CG-DETR and EaTR runs retain all six adaptation epochs and report fixed epoch four. Those runs include Test diagnostics at epochs three through six. They therefore provide single-seed implementation evidence rather than a fresh confirmatory holdout study.

We report threshold-free AUROC, rejection F1, negative rejection and positive acceptance, alongside generalized retrieval and localization. The main operating threshold is 0.4. Positive-query localization mAP uses at most ten predictions and tIoU thresholds from 0.50 to 0.95 in increments of 0.05, without hard existence gating. Generalized retrieval applies the existence decision before set-level overlap evaluation on all queries. This distinction is necessary because rejecting a positive query can improve aggregate rejection behavior while removing a useful localization.

## Results

At threshold 0.4, the five archived adaptations improve G-mIoU@1 and ungated localization mAP relative to their corresponding local GMR baselines (Table 1). These comparisons concern the complete adapted systems and do not isolate individual components.

Table 1. Changes relative to the corresponding local GMR baseline at threshold 0.4. All entries are percentage-point differences; each adaptation uses seed 2026 and reports epoch four.

| Backbone | G-mIoU@1 | Ungated mAP | AUROC | Positive acceptance |
| --- | ---: | ---: | ---: | ---: |
| Moment-DETR | +46.09 | +0.91 | +0.06 | −60.29 |
| QD-DETR | +10.84 | +1.56 | −0.80 | −17.46 |
| CG-DETR | +27.68 | +0.68 | +0.32 | −41.18 |
| EaTR | +15.26 | +0.81 | −0.53 | −26.47 |
| FlashVTG | +23.44 | +0.41 | +0.85 | −35.66 |

The gains accompany a shift toward rejection. Positive acceptance decreases for all five backbones, and AUROC decreases by 0.80 points for QD-DETR and 0.53 points for EaTR. For FlashVTG, G-mIoU@1 rises from 28.10% to 51.54% and negative rejection rises from 32.72% to 89.43%, while positive acceptance falls from 86.76% to 51.10%. The higher fixed-threshold score therefore does not imply uniformly improved discrimination or positive retrieval.

FlashVTG's adaptation trains 2,496,452 parameters compared with 10,600,985 for E0. This establishes a smaller trainable parameter set for adaptation, without measuring end-to-end training cost or inference efficiency. At threshold 0.5, its ungated mAP increases from 25.43% to 25.84%, whereas hard-gated mAP decreases from 16.07% to 11.16%, reflecting the loss of accepted positives.

The exploratory A3 checkpoint yields mAP of 26.37% and AUROC of 75.77%, compared with 25.84% and 75.15% without augmentation. At threshold 0.4, G-mIoU@1 decreases from 51.54% to 50.99%. Because multiple A3 epochs were inspected on Test, this result is reported as exploratory evidence and does not establish that context diversification resolves the positive-retention limitation.

## Limitations

The current results cover one dataset and one adaptation seed. Backbone-specific initialization, training schedules and checkpoint policies differ, and controlled component ablations are not available in the documented evidence. Accordingly, the experiments do not establish that representation reuse causes the baseline's errors or that any single module explains the observed changes.

The dataset also couples source and target regime: all SportsMoments examples are positive singles, while WorldCup2022 supplies all negatives and multi-target examples. Query IDs and video IDs are disjoint across partitions, but exact match metadata overlaps. The results therefore do not establish match-disjoint or unseen-action generalization. Positive-event retention, especially in the multi-target and WorldCup subsets, remains a substantive limitation that should accompany claims based on aggregate G-mIoU.

## References

1. Yiming Ding et al. 2026. *Retrieving Any Relevant Moments: Benchmark and Models for Generalized Moment Retrieval.* [arXiv:2605.02623](https://arxiv.org/abs/2605.02623).
2. Namyup Kim, Jinsung Lee, and Suha Kwak. 2026. *Improving Target Presence and Plurality Recognition for Generalized Referring Image Segmentation.* Proceedings of AAAI 40(7), 5700–5708. [DOI:10.1609/aaai.v40i7.37490](https://doi.org/10.1609/aaai.v40i7.37490).
3. Edward J. Hu et al. 2021. *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
4. Sungho Moon, Seunghun Lee, Jiwan Seo, and Sunghoon Im. 2026. *CVA: Context-aware Video-text Alignment for Video Temporal Grounding.* [arXiv:2603.24934](https://arxiv.org/abs/2603.24934).
