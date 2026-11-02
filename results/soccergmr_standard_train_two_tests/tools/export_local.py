#!/usr/bin/env python3
"""One-time export from verified local runs. Read-only toward all original data."""
import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path('/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval')
CORE=Path('/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr')
STANDARD=Path('/home/guoxiangyu/charades-multi-geq-worktree')
FULL=PROJECT/'soccergmr_full_test_20261102'
KIT=PROJECT/'gmr_geq_protocol_kit'
DATA_ROOT=Path('/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval')
BACKBONES=['moment_detr','eatr','qd_detr','cg_detr']
METHODS=['gmr','gmr_geq_frozen','gmr_geq_full','multi','multi_geq_frozen','multi_geq_full']
COMMON=['AUROC','mAP','mR@1','mR@5','mR+@5']
PER=['N-acc','Rej-F1','G-mIoU@1','G-mIoU@3']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n')

def main():
    origins={};copies={}
    def copy(p,relative):
        p=Path(p);dest=ROOT/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest);assert sha(p)==sha(dest)
        origins[str(p)]=sha(p);copies[str(relative)]=dict(source=str(p),sha256=sha(p))
        return str(relative)
    cp=PROJECT/'soccergmr_full_test_handoff_20261102/CHECKPOINTS.json'
    da=PROJECT/'soccergmr_full_test_handoff_20261102/DATA_AUDIT.json'
    entries=read(cp)['checkpoints'];assert len(entries)==24
    source_std=STANDARD/'results/test_all_thresholds/TEST_ALL_THRESHOLDS.json'
    std_rows=[r for r in read(source_std) if r['dataset']=='SoccerGMR' and r['protocol']=='current_val_selection' and r['method'] in METHODS]
    assert len(std_rows)==24
    std_index={(r['backbone'],r['method']):r for r in std_rows}
    full_index={(r['backbone'],r['method']):r for r in read(FULL/'RESULTS.json')}
    origins[str(source_std)]=sha(source_std)
    for p in (cp,da,FULL/'RESULTS.json',FULL/'INPUT_AUDIT.json',FULL/'AUDIT.json',FULL/'COMPLETE.json'):
        copy(p,Path('evidence/original')/p.name)
    data=read(da);datasets={}
    for split in ('train','val','test'):
        p=DATA_ROOT/'data/label/Standard'/f'{split}.jsonl';rr=rows(p)
        name=copy(p,Path('data/labels')/f'standard_{split}.jsonl')
        datasets['standard_'+split]=dict(labels=name,rows=len(rr),positive=sum(bool(d['relevant_windows']) for d in rr),negative=sum(not d['relevant_windows'] for d in rr),sha256=sha(p))
    for name,key in [('full_decodable','available_decodable_test'),('full_original','original_test')]:
        p=Path(data[key]['path']);rr=rows(p)
        label=copy(p,Path('data/labels')/f'{name}_test.jsonl')
        datasets[name]=dict(labels=label,rows=len(rr),positive=sum(bool(d['relevant_windows']) for d in rr),negative=sum(not d['relevant_windows'] for d in rr),sha256=sha(p),
                            raw_vid_strings=len({d['vid'] for d in rr}),normalized_video_stems=len({Path(d['vid']).stem for d in rr}))
    copy(FULL/'EXCLUDED_ORIGINAL_FULL_TEST_16.jsonl','data/labels/full_original_excluded_16.jsonl')
    for key in ('feature_manifest','feature_verification','feature_exclusions'):
        p=Path(data[key]);copy(p,Path('data/feature_protocol')/p.name)
    copy(FULL/'FEATURE_SHA256.json','data/FULL_FEATURE_SHA256.json')
    cg_features=PROJECT/'cg_detr_gmiou_handoff_20261102/FEATURE_SHA256.json'
    copy(cg_features,'data/REGISTERED_FEATURE_SHA256.json')
    datasets['full_feature_protocol']=data['feature_protocol']
    datasets['feature_manifest_sha256']=data['feature_manifest_sha256']
    datasets['overlap']=data['standard_overlap']
    datasets['full_temporal_caveats']={k:read(FULL/'INPUT_AUDIT.json')[k] for k in ('over_75_steps','temporal_mismatches_gt_vs_features','native_truncation')}
    dump(ROOT/'data/DATASETS.json',datasets)
    run_entries=[];checkpoint_entries=[];training=[]
    for entry in entries:
        b=entry['backbone'];m=entry['method'];key=(b,m);is_geq='_geq_' in m
        checkpoint=Path(entry['checkpoint']);assert sha(checkpoint)==entry['sha256']
        origins[str(checkpoint)]=entry['sha256']
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert len(state['model'])==entry['state_keys'] and state['epoch']==entry['epoch_raw']
        native_seed=None
        if not is_geq:
            config=read(checkpoint.parent/'experiment_config.json');native_seed=config.get('seed')
        else:config=read(checkpoint.parent/'CONFIG.json')
        evidence_dir=Path('evidence/training')/b/m
        metadata=[]
        for name in ('CONFIG.json','COMPLETE.json','PREFLIGHT.json','experiment_config.json','execution_plan.json','history.jsonl',
                     'train.log','val.log','frozen_checkpoint.json','frozen_selection.json','reused_baseline.json','source_effective_config.json',
                     'best_val_metrics.json','baseline_val_metrics.json','status.json','STATUS.json'):
            if (checkpoint.parent/name).is_file():metadata.append(copy(checkpoint.parent/name,evidence_dir/name))
        layout=dict(epoch_raw=state['epoch'],state_keys=len(state['model']),checkpoint_keys=list(state),
                    parameters={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in state['model'].items()})
        dump(ROOT/evidence_dir/'CHECKPOINT_LAYOUT.json',layout)
        del state
        model_id=b+'/'+m
        checkpoint_entries.append(dict(entry,id=model_id,checkpoint_uploaded=False,training_evidence=str(evidence_dir),
                                       config=config,seed=config.get('seed'),best_epoch_human=entry['epoch_raw'] if is_geq else entry['epoch_raw']+1))
        if not is_geq:
            native=dict(backbone=b,method=m,seed=native_seed,config=config,checkpoint_sha256=entry['sha256'],evidence=metadata)
            if (checkpoint.parent/'train.log').is_file():native['last_logged_epoch']=rows(checkpoint.parent/'train.log')[-1].get('epoch')
            elif (checkpoint.parent/'history.jsonl').is_file():native['last_logged_epoch']=rows(checkpoint.parent/'history.jsonl')[-1].get('epoch')
            else:native['last_logged_epoch']=None
            training.append(native)
        std=std_index[key];summary_path=STANDARD/std['source'];summary=read(summary_path)
        summary_rel=copy(summary_path,Path('evidence/standard_summaries')/b/f'{m}.json')
        pred=Path(summary['source_predictions'])
        assert sha(pred)==summary['signature']['prediction_sha256']
        assert abs(std['common']['mAP']-entry['standard_test_mAP'])<1e-12
        if is_geq:
            complete=read(checkpoint.parent/'COMPLETE.json');assert complete['checkpoint_sha256']==entry['sha256']
        for test,prediction,expected in [('standard',pred,dict(std['common'],**{f'{k}@tau={t}':std['thresholds'][str(t)][k] for t in (.4,.5,.6) for k in PER})),
                                          ('full_decodable',FULL/'runs'/b/m/'predictions.jsonl',full_index[key]['values'])]:
            raw=prediction.read_bytes();pred_records=[json.loads(l) for l in raw.decode().splitlines() if l.strip()]
            assert len(pred_records)==datasets['standard_test' if test=='standard' else 'full_decodable']['rows']
            relative=Path('data/predictions')/test/b/f'{m}.jsonl.gz';dest=ROOT/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('wb') as f:
                with gzip.GzipFile(fileobj=f,mode='wb',filename='',mtime=0) as g:g.write(raw)
            origins[str(prediction)]=sha(prediction)
            evidence={}
            if test=='full_decodable':
                source_dir=prediction.parent;complete=read(source_dir/'COMPLETE.json')
                assert complete['identity']['checkpoint_sha256']==entry['sha256'] and complete['strict_load']
                assert complete['predictions_sha256']==sha(prediction)
                for name in ('COMPLETE.json','PREFLIGHT.json','metrics.json'):
                    evidence[name]=copy(source_dir/name,Path('evidence/full')/b/m/name)
            else:
                evidence['summary']=summary_rel
                metric=Path(summary['source_metrics'])
                assert sha(metric)==summary['signature']['metric_sha256']
                evidence['metrics']=copy(metric,Path('evidence/standard')/b/m/'metrics.json')
            run_entries.append(dict(test=test,backbone=b,method=m,label=entry['label'],model_id=model_id,
                                    checkpoint_sha256=entry['sha256'],predictions=str(relative),prediction_source=str(prediction),
                                    predictions_uncompressed_sha256=sha(prediction),predictions_gzip_sha256=sha(dest),
                                    gt=datasets['standard_test' if test=='standard' else 'full_decodable']['labels'],
                                    expected_published=expected,evidence=evidence))
        print('EXPORTED',b,m,flush=True)
    dump(ROOT/'data/RUNS.json',run_entries)
    dump(ROOT/'data/CHECKPOINTS.json',checkpoint_entries)
    dump(ROOT/'data/NATIVE_TRAINING.json',training)
    # Capture actual runtime sources; large models, features and unrelated runs
    # are deliberately excluded. These are historical-path source snapshots.
    trees=[(KIT,'source/gmr_geq_protocol_kit'),(PROJECT/'gmr_multiroute_kit','source/gmr_multiroute_kit'),
           (PROJECT/'generalized-moment-retrieval/training/moment_detr_gmr','source/moment_project/training/moment_detr_gmr'),
           (PROJECT/'generalized-moment-retrieval/models/moment_detr_gmr','source/moment_project/models/moment_detr_gmr'),
           (PROJECT/'generalized-moment-retrieval/configs/moment_detr_gmr','source/moment_project/configs/moment_detr_gmr'),
           (CORE.parent/'eatr/models','source/backbonefolder/eatr/models'),(CORE.parent/'eatr/utils','source/backbonefolder/eatr/utils'),
           (CORE.parent/'qd_detr/gmr_deployment','source/backbonefolder/qd_detr/gmr_deployment'),
           (CORE.parent/'cg_detr/gmr_source','source/backbonefolder/cg_detr/gmr_source')]
    for source,prefix in trees:
        for p in source.rglob('*'):
            if p.is_file() and p.suffix in ('.py','.yaml','.yml','.json') and '__pycache__' not in p.parts:
                copy(p,Path(prefix)/p.relative_to(source))
    for p in CORE.glob('*.py'):copy(p,Path('source/backbonefolder/multiroute_gmr')/p.name)
    copy(CORE/'execution_plan.json','source/backbonefolder/multiroute_gmr/execution_plan.json')
    for p,relative in [(CORE.parent/'eatr/deploy.py','source/backbonefolder/eatr/deploy.py'),
                       (CORE.parent/'eatr/config.py','source/backbonefolder/eatr/config.py'),
                       (CORE.parent/'qd_detr/train_gmr.py','source/backbonefolder/qd_detr/train_gmr.py'),
                       (CORE.parent/'cg_detr/train_gmr.py','source/backbonefolder/cg_detr/train_gmr.py')]:copy(p,relative)
    for p in FULL.glob('*.py'):copy(p,Path('source/full_eval')/p.name)
    official=Path('/home/guoxiangyu/GMR_Unseen/experiments/table2_repro_20261022/shared/official_eval')
    for p in official.glob('*.py'):copy(p,Path('lib/official_eval')/p.name)
    copy(PROJECT/'cg_detr_gmiou_handoff_20261102.zip','handoff/cg_detr_gmiou_handoff_20261102.zip')
    dump(ROOT/'PROVENANCE.json',dict(scope='24 current Standard-trained checkpoints, two Test domains, 48 formal records',
         origin_sha256=origins,copied_files=copies,source_report_commit='f697dcb2e41825430c85796e8b58cb2bd9d58aaf',
         repository_base_commit='39ab0eb74427709851476b2bb86b698c8a3f4186',training_started=False,inference_started=False,
         weights_uploaded=False,features_uploaded=False,original_results_modified=False))
    print('EXPORT_COMPLETE',len(run_entries),'records;',len(origins),'original inputs hashed',flush=True)

if __name__=='__main__':main()
