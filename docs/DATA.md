# SoccerGMR data contract

The primary pipeline consumes published/precomputed features. Feature extraction and raw-video decoding are outside this package. Only labels, NPZ assets and optional external weights may come from another directory; no external project Python modules are imported.

## Splits and coverage

Configure Standard Train / Val / Test and Full Test explicitly. Canonical Standard counts are 4,138 / 465 / 1,036 queries. Standard Train has 2,136 positive queries; Val has 255; Test has 544 positive and 492 null queries. Full raw Test has 2,986 queries (1,638 positive / 1,348 null). The currently available Full-Decodable manifest has 2,970 (1,633 positive / 1,337 null). Its 16 excluded queries must not silently disappear from a claimed 2,986-query result.

The local validation uses Full-Decodable 2970/2986; this limitation is recorded in the audit. To evaluate all 2,986 queries, provide all required NPZ files and the original Full manifest. The code fails on missing assets. Standard Test queries are also present in Full Test; these are two evaluation settings, not independent training datasets.

Each JSONL row has `qid` (unique within split), `vid`, positive `duration` in seconds and `relevant_windows` (a list of `[start,end]` seconds). An empty list is a genuine null query. Multiple windows are preserved. Each window must satisfy `0 <= start < end <= duration`. Video identities must not overlap Train / Val / Test; Standard Test and Full Test may overlap.

`audit` validates arrays and hashes every referenced asset across all four splits. Training itself reads only Train / Val and checks their video disjointness. Run the audit before any formal experiment to verify disjointness against both Test sets, without using Test scores.

## NPZ schema

| Directory | Filename | Array | Shape |
|---|---|---|---|
| `clip` | `<video-stem>.npz` | `features` | T × 512 |
| `slowfast` | `<video-stem>.npz` | `features` | T × 2304 |
| `clip_text` | `qid<qid>.npz` | `last_hidden_state` | L × 512 |
| `clip_text` | same | `attention_mask` (required for Sim-DETR) | L |

Video stems omit `.mp4`. Standard and Full use independent feature roots, including independent text qid namespaces. Pointing Full to Standard text features can load the wrong query even if filenames exist.

Video streams align to the shorter available stream and are each L2 normalized with epsilon `1e-5`. Text is truncated at 32 tokens and L2 normalized. Sim-DETR applies the stored text attention mask; the other archived native profiles used all stored tokens up to 32. Feature order is declared by each registry spec: Moment-DETR and Sim-DETR use CLIP then SlowFast; EaTR, QD-DETR and CG-DETR use SlowFast then CLIP. Two TEF channels follow the concatenated video channels.

## Coordinates and targets

The current common contract normalizes span targets by **actual annotated duration**, then exports predictions with that same duration. TEF covers the full loaded sequence. Targets include all windows, union saliency, mask targets and real query existence. EaTR retains its original SoccerGMR profile without saliency-pair supervision. Other saliency pair indices use deterministic first/last positive and negative frames. Sim-DETR retains native null-aware losses; the other backbones apply native localization losses to positive queries and existence BCE to every real query.

The default video limit is 128 tokens. No video is silently truncated: increasing capacity requires explicit configuration. The audited Full assets include 104- and 94-token videos, which exceeded an old 75-token limit. This correction and the common timestamp/target contract can change numerical results relative to historical scripts. Therefore historical checkpoints/scores are not asserted to be numerically reproduced by the new training protocol.

## Recorded identities

Training checkpoints bind the resolved config, source-tree hash and Train / Val labels plus feature hashes. Test artifacts bind the selected checkpoint, source, Test labels and features. Moving assets changes the current path-sensitive identity and requires a new run directory. Mutating a file in place also invalidates identity. Keep asset snapshots immutable throughout a run.

Full feature extraction provenance should accompany a formal result (extractor/checkpoint, clip stride, preprocessing and exclusions). The current assets used the recorded `soccer_full_lighthouse_standard_bn45_v1` extraction protocol; an independent extractor is not certified by the integration test.
