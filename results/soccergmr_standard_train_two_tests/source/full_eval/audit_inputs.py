"""Read and validate every unique Full Test feature, without modifying input files."""
import collections
import math
import numpy as np
from common import *

def audit():
    hashes = {}
    for entry in CHECKPOINTS['checkpoints']:
        p = entry['checkpoint']; hashes[p] = sha(p)
        assert hashes[p] == entry['sha256'], p
    for key in ('original_test','available_decodable_test'):
        item = DATA[key]; hashes[item['path']] = sha(item['path'])
        assert hashes[item['path']] == item['sha256']
    for key in ('feature_manifest','feature_verification','feature_exclusions'):
        p = DATA[key]; hashes[p] = sha(p)
        assert hashes[p] == DATA[key+'_sha256']
    manifest = json.loads(Path(DATA['feature_manifest']).read_text())
    verified = json.loads(Path(DATA['feature_verification']).read_text())
    exclusions = json.loads(Path(DATA['feature_exclusions']).read_text())
    assert verified['passed'] and manifest['protocol'] == DATA['feature_protocol']
    assert verified['manifest_sha256'] == DATA['feature_manifest_sha256']
    gt, original = rows(GT), rows(DATA['original_test']['path'])
    assert len(gt) == len({d['qid'] for d in gt}) == 2970
    assert sum(bool(d['relevant_windows']) for d in gt) == 1633
    original_index = {d['qid']: d for d in original}
    assert all(d == original_index[d['qid']] for d in gt)
    missing = [d for d in original if d['qid'] not in {x['qid'] for x in gt}]
    assert missing == DATA['excluded_test_rows'] and len(missing) == 16
    assert {d['qid'] for d in missing} == {d['qid'] for d in exclusions['excluded_queries'] if d['split']=='test'}
    jsonl(HERE/'EXCLUDED_ORIGINAL_FULL_TEST_16.jsonl', missing)
    video_lengths, text_lengths, feature_files = {}, {}, {}
    timing = []
    for d in gt:
        duration = float(d['duration'])
        assert math.isfinite(duration) and duration > 0
        assert isinstance(d['relevant_windows'], list)
        for s,e in d['relevant_windows']: assert 0 <= s < e <= duration, d
        vid = d['vid']; stem = Path(vid).stem
        if vid not in video_lengths:
            lengths = []
            for kind, dim in [('clip',512),('slowfast',2304)]:
                p = FEATURES/kind/f'{stem}.npz'
                with np.load(p) as z: x = z['features']
                assert x.ndim==2 and x.shape[1]==dim and len(x)>0 and np.isfinite(x).all(), p
                lengths.append(len(x)); feature_files[str(p)] = sha(p)
            assert lengths[0] == lengths[1], (vid,lengths)
            video_lengths[vid] = lengths[0]
            if abs(lengths[0]*2-duration) > 2:
                timing.append(dict(vid=vid, feature_steps=lengths[0], feature_seconds=lengths[0]*2, label_duration=duration))
        p = FEATURES/'clip_text'/f'qid{d["qid"]}.npz'
        with np.load(p) as z: x=z['last_hidden_state']
        assert x.ndim==2 and x.shape[1]==512 and len(x)>0 and np.isfinite(x).all(),p
        text_lengths[d['qid']]=len(x); feature_files[str(p)] = sha(p)
    manifest_queries = {d['qid']:d for d in manifest['queries']}
    assert all(all(d[k]==manifest_queries[d['qid']][k] for k in ('vid','query','duration')) for d in gt)
    overlap={}
    labels=PROJECT/'soccergmr_standard/d_gmr_two_stage/dataset/data/label/Standard'
    for split in ('train','val','test'):
        standard=rows(labels/f'{split}.jsonl')
        hashes[str(labels/f'{split}.jsonl')]=sha(labels/f'{split}.jsonl')
        pairs={(Path(d['vid']).stem,d['query']) for d in original}
        overlap[split]=dict(videos=len({Path(d['vid']).stem for d in original}&{Path(d['vid']).stem for d in standard}),
                            video_query_pairs=len(pairs & {(Path(d['vid']).stem,d['query']) for d in standard}))
    assert overlap['train']['videos']==overlap['val']['videos']==0
    assert overlap['test']['video_query_pairs']==1036
    dump(HERE/'FEATURE_SHA256.json',feature_files)
    obj=dict(passed=True,title=TITLE, input_sha256=hashes,
             handoff_sha256={str(p):sha(p) for p in HANDOFF.glob('*') if p.is_file()},
             gt_rows=2970,positive=1633,negative=1337,unique_videos=len(video_lengths),
             multi_gt=sum(len(d['relevant_windows'])>=2 for d in gt),
             original_rows=2986,excluded_rows=16,excluded_positive=5,excluded_negative=11,
             feature_files=len(feature_files),feature_sha256=sha(HERE/'FEATURE_SHA256.json'),
             video_length_histogram=dict(sorted(collections.Counter(video_lengths.values()).items())),
             text_token_histogram=dict(sorted(collections.Counter(text_lengths.values()).items())),
             over_75_steps=[dict(vid=v,steps=n,qids=[d['qid'] for d in gt if d['vid']==v]) for v,n in video_lengths.items() if n>75],
             native_max_video_steps=75,native_max_text_tokens=32,
             native_truncation='Preserve original maximum 75 video steps; truncate each stream before concatenation; retain original duration scaling and native postprocessing. No query is dropped.',
             temporal_mismatches_gt_vs_features=timing,
             overlap_original_full_vs_standard=overlap,
             source_feature_protocol=DATA['feature_protocol'],
             feature_manifest_parameters=manifest['parameters'],
             code_sha256=code_files())
    dump(HERE/'INPUT_AUDIT.json',obj)
    print(json.dumps({k:obj[k] for k in ('passed','gt_rows','feature_files','over_75_steps','temporal_mismatches_gt_vs_features')}),flush=True)
    return obj

if __name__=='__main__': audit()
