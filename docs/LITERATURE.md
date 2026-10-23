# Focused literature review

## Retrieval scope

This review supports the repository's data and algorithm documentation. It covers the established generalized task, rejection, multiple moments, the five backbone families, the supplied image-segmentation inspiration, parameter adaptation and optional context augmentation. It is not a systematic review or an exhaustive novelty assessment. Thirteen references received an independent primary-source citation check; raw metadata retrievals and screened records are retained in [research](research/).

## Established generalized retrieval

The temporal problem already has several formulations. OpenVMR considers rejection of video-irrelevant queries, while Moment of Untruth distinguishes in-domain and out-of-domain negatives in negative-aware retrieval. Qin et al. explicitly include no-target and one-to-many cases in Generalized Video Moment Retrieval and introduce NExT-VMR and BCANet. These precedents rule out a claim that empty-set temporal retrieval or multiple matching moments are first introduced here. [OpenVMR](https://dl.acm.org/doi/10.1145/3664647.3680947), [Moment of Untruth](https://arxiv.org/abs/2502.08544), [Qin et al.](https://proceedings.iclr.cc/paper_files/paper/2025/file/7ac19fdcdf4f311f3e3ef2e7ef4784d7-Paper-Conference.pdf).

Ding et al. study retrieval of the complete relevant temporal set or an empty set, provide Soccer-GMR and evaluate rejection, positive localization and generalized retrieval separately. Their discriminative adapter max-pools cross-modal localization representations and applies a two-layer presence MLP. This provides a concrete comparison for representation reuse, but does not show that presence directly consumes candidate coordinates. The current method instead reads intermediate video memories with a dedicated event token; whether this readout remedies a specific feature conflict requires controlled evidence. [Soccer-GMR paper](https://arxiv.org/abs/2605.02623).

## Dedicated target queries as cross-task inspiration

Kim, Lee and Kwak introduce a target query reading intermediate pixel-decoder features for presence recognition in generalized referring image segmentation. Their framework also uses text-conditioned plurality supervision and no-target/multi-target data augmentation. That work motivates a dedicated presence readout, but concerns spatial segmentation. It does not establish GMR performance, and its plurality loss and synthetic-image construction are not active components of this repository. The architectural relationship should be acknowledged directly rather than presenting the query concept as entirely new. [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/37490).

## Temporal grounding and query conditioning

QVHighlights introduces a query-based moment/highlight benchmark and Moment-DETR, and already allows multiple disjoint relevant moments. QD-DETR emphasizes query-dependent video representations and includes training with negative video-query pairs; that negative-pair objective should not be confused with a full generalized-rejection protocol. CG-DETR calibrates query dependence through correlation-guided modeling, and EaTR forms event-aware dynamic moment queries. These works show that query dependence and event-aware queries are established ideas. The present global event query has a different output role: a video-query existence score rather than a temporal slot. [QVHighlights](https://arxiv.org/abs/2107.09609), [QD-DETR](https://arxiv.org/abs/2303.13874), [CG-DETR](https://arxiv.org/abs/2311.08835), [EaTR](https://arxiv.org/abs/2308.06947).

FlashVTG uses temporal feature layering and adaptive score refinement for grounding and highlight detection, providing the root implementation's native localizer. FlashMMR separately introduces multi-moment retrieval with QV-M² and Multi-moment Post-verification. These methods place both proposal refinement and cross-moment processing in an existing research context; their original results are not evidence for this repository's absent-query behavior. The present refiner's proposal-centered Gaussian pooling must be assessed against relevant temporal weighting/refinement methods before claiming operation-level novelty. [FlashVTG](https://arxiv.org/abs/2412.13441), [FlashMMR](https://arxiv.org/abs/2510.17218).

## Adaptation and data augmentation

LoRA freezes pretrained weights and learns low-rank updates. Here it is applied to video encoder FFN layers, with the global reader and refiner trained alongside it. This supports describing the trainable parameterization; the original language-model paper's speed and memory findings are not measurements of the GMR implementation. [LoRA](https://arxiv.org/abs/2106.09685).

CVA introduces Query-aware Context Diversification to screen replacement clips by query relevance. It motivates A3's background replacement, while A3's WorldCup-only donor scope, full multi-GT protection and synchronized SlowFast/CLIP replacement are repository-specific adaptations. CVA also has other architectural and loss components that are not included here. The five-backbone main archive has QCD disabled, and A3 remains an exploratory FlashVTG experiment. [CVA](https://arxiv.org/abs/2603.24934).

## Reference handling

The reference list below cites the exact retrieved form. A preprint's year can differ from its conference year: FlashVTG is a 2024 preprint with a WACV 2025 publication; LoRA is a 2021 preprint; CG-DETR's listed preprint began in 2023. OpenVMR is cited as ACM MM 2024 despite a 2026 arXiv upload. Crossref metadata and citation counts are not semantic endorsement, and a missing Crossref match for a DataCite arXiv DOI is not evidence that the paper does not exist. Full author metadata, version qualifications and verification locations are in [evidence_cards.json](research/evidence_cards.json) and [citation_verification.json](research/citation_verification.json).

## References

1. Yiming Ding et al. 2026. *Retrieving Any Relevant Moments: Benchmark and Models for Generalized Moment Retrieval.* [arXiv:2605.02623](https://arxiv.org/abs/2605.02623).
2. Namyup Kim, Jinsung Lee, and Suha Kwak. 2026. *Improving Target Presence and Plurality Recognition for Generalized Referring Image Segmentation.* Proceedings of AAAI 40(7), 5700–5708. [DOI:10.1609/aaai.v40i7.37490](https://doi.org/10.1609/aaai.v40i7.37490).
3. Jie Lei, Tamara L. Berg, and Mohit Bansal. 2021. *QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries.* [arXiv:2107.09609](https://arxiv.org/abs/2107.09609).
4. WonJun Moon, Sangeek Hyun, SangUk Park, Dongchan Park, and Jae-Pil Heo. 2023. *Query-Dependent Video Representation for Moment Retrieval and Highlight Detection.* [arXiv:2303.13874](https://arxiv.org/abs/2303.13874).
5. WonJun Moon, Sangeek Hyun, SuBeen Lee, and Jae-Pil Heo. 2023. *Correlation-Guided Query-Dependency Calibration for Video Temporal Grounding.* [arXiv:2311.08835](https://arxiv.org/abs/2311.08835).
6. Jinhyun Jang, Jungin Park, Jin Kim, Hyeongjun Kwon, and Kwanghoon Sohn. 2023. *Knowing Where to Focus: Event-aware Transformer for Video Grounding.* [arXiv:2308.06947](https://arxiv.org/abs/2308.06947).
7. Zhuo Cao, Bingqing Zhang, Heming Du, Xin Yu, Xue Li, and Sen Wang. 2024. *FlashVTG: Feature Layering and Adaptive Score Handling Network for Video Temporal Grounding.* [arXiv:2412.13441](https://arxiv.org/abs/2412.13441).
8. Edward J. Hu et al. 2021. *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
9. Sungho Moon, Seunghun Lee, Jiwan Seo, and Sunghoon Im. 2026. *CVA: Context-aware Video-text Alignment for Video Temporal Grounding.* [arXiv:2603.24934](https://arxiv.org/abs/2603.24934).
10. You Qin, Qilong Wu, Yicong Li, Wei Ji, Shawn Li, Pengcheng Cai, Lina Wei, and Roger Zimmermann. 2025. *Generalized Video Moment Retrieval.* ICLR. [Official paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/7ac19fdcdf4f311f3e3ef2e7ef4784d7-Paper-Conference.pdf).
11. Kevin Flanagan, Dima Damen, and Michael Wray. 2025. *Moment of Untruth: Dealing with Negative Queries in Video Moment Retrieval.* [arXiv:2502.08544](https://arxiv.org/abs/2502.08544).
12. Xiang Fang et al. 2024. *Not All Inputs Are Valid: Towards Open-Set Video Moment Retrieval using Language.* ACM Multimedia, 28–37. [DOI:10.1145/3664647.3680947](https://doi.org/10.1145/3664647.3680947).
13. Zhuo Cao, Heming Du, Bingqing Zhang, Xin Yu, Xue Li, and Sen Wang. 2025. *When One Moment Isn't Enough: Multi-Moment Retrieval with Cross-Moment Interactions.* [arXiv:2510.17218](https://arxiv.org/abs/2510.17218).
