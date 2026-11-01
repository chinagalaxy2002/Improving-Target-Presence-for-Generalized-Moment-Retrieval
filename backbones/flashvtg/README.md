# FlashVTG-GMR + Multi-TQAux (Charades structural adaptation)

FlashVTG retains its native dense temporal localization head and losses. The `multi` configuration adds two training-only temporal query routes, each with24 queries, a shared two-layer width256/eight-head SA/CA decoder, one separate FFN route and one common FFN route with four instruction tokens. Fixed temporal O2M K=2 uses classification/L1/tGIoU supervision, weighted0.15 per route. The auxiliary decoder does not execute in eval; inference uses the native dense head.

This is a structural adaptation, not a claim that FlashVTG has the same native query decoder as Moment-DETR, EaTR or QD-DETR. Frozen keeps the trunk in eval; native Flash eval batch1 is respected even when GEQ training averages an eight-sample batch. Positive examples with no assigned dense regression point receive a differentiable zero regression term while keeping native classification/existence supervision.

[Shared code and exact protocol](../../shared/charades_multi_geq/README.md) · [Charades results](../../results/charades_multi_geq/flashvtg/README.md)

Use `bash charades_multi_geq.sh --method multi --mode train` for a single worker after running the shared preflights; use the shared persistent queue to enforce dependencies and two tasks per physical GPU.
