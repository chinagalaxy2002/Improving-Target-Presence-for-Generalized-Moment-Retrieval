# 基于当前仓库的论文段落

下文只描述当前仓库的主方法、原始数据和五主干结果，不引入新的文献判断或实验。论文方法名暂沿用仓库实验名 `S4_minus_Fusion`。其中 .4 工作点的改善、正例保留下降和单 seed 限制均应保留。

## Abstract

Generalized Moment Retrieval requires localizing all temporal intervals matching a natural-language query while returning an empty set when the event is absent. We study S4_minus_Fusion, a framework combining a global event-query readout with a separate proposal-refinement path. A learnable query reads projected text and intermediate video representations to estimate event existence without directly consuming candidate coordinates, scores, or RoI features. In parallel, Gaussian temporal Region-of-Interest pooling produces proposal-centered video representations for text-conditioned boundary and confidence refinement. Rank-eight low-rank adaptation updates the video encoder while retaining frozen backbone weights. We evaluate implementations with Moment-DETR, QD-DETR, CG-DETR, EaTR, and FlashVTG using the original Soccer-GMR Standard data. At an existence threshold of 0.4, the archived single-seed adaptations improve G-mIoU@1 by 10.84–46.09 percentage points and ungated localization mAP by 0.41–1.56 points relative to their corresponding local GMR baselines. These improvements accompany lower positive-query acceptance, and AUROC decreases for two backbones. The results establish the framework's applicability to the evaluated architectures while identifying positive-event retention and threshold sensitivity as unresolved limitations.

## Method description

The model separates the forward readout for event existence from candidate-conditioned temporal refinement. An event token first attends to projected text and then sequentially reads intermediate video encoder memories. A linear head maps the final state to an existence logit. Explicit temporal proposals and their RoI representations are excluded from this readout. The paths share backbone representations and trainable low-rank updates, so their separation does not imply independent optimization.

For localization, the model selects up to eight high-confidence native proposals and pools final video features using proposal-centered Gaussian weights. A geometry embedding represents normalized endpoints, duration, and confidence. Each moment representation attends to text and incorporates local saliency cues to predict bounded endpoint and confidence residuals. The selected candidates are replaced, while the remaining native candidates and backbone-specific output processing are retained.

Adaptation trains the event reader, RoI encoder, refiner, and rank-eight low-rank matrices in the encoder feed-forward layers. The objective combines existence binary cross-entropy on all queries with a positive-only refinement loss weighted by 0.5. The refinement term includes normalized Smooth L1 endpoints, temporal generalized IoU, and an IoU-quality confidence objective. Each ground-truth interval selects its highest-overlap candidate; this assignment is not constrained to be one-to-one. The active objective does not include explicit plurality supervision.

## Experimental statement

The repository contains 14 local configurations: Baseline, GMR, and GMR with S4_minus_Fusion for four DETR backbones, plus E0/GMR and its S4 adaptation for FlashVTG. All published S4 rows report epoch four. The four DETR adaptations are fixed-epoch runs with Test diagnostics recorded during training; FlashVTG uses its archived reference checkpoint. These runs do not provide a new untouched confirmatory holdout.

Table 1 reports changes relative to the corresponding local GMR baseline at threshold 0.4. Values are percentage-point differences from the repository's frozen predictions.

| Backbone | G-mIoU@1 | Ungated mAP | AUROC | Positive acceptance |
| --- | ---: | ---: | ---: | ---: |
| Moment-DETR | +46.09 | +0.91 | +0.06 | −60.29 |
| QD-DETR | +10.84 | +1.56 | −0.80 | −17.46 |
| CG-DETR | +27.68 | +0.68 | +0.32 | −41.18 |
| EaTR | +15.26 | +0.81 | −0.53 | −26.47 |
| FlashVTG | +23.44 | +0.41 | +0.85 | −35.66 |

The gains are specific to the reported operating point. For example, Moment-DETR's G-mIoU@1 difference changes from +46.09 points at 0.4 to −0.10 points at 0.6. Localization mAP is evaluated without a hard existence gate, so its improvement does not establish preservation of accepted positive retrievals. The single-dataset, single-seed results and differing native training protocols do not isolate the effects of individual components or establish runtime savings.
