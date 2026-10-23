"""Requirement-level audit: predictions, counts, checkpoint, metrics, controls."""
import csv,hashlib,json,math
from pathlib import Path
from official_eval.eval_main import evaluate_gmr
from official_eval.metrics import _clean_pred_windows
ROOT=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def main():
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text());checks=[];missing=[];degenerate_windows=[]
    for backbone in protocol['backbones']:
        initial=[]
        for row in protocol['execution_rows']:
            run=ROOT/'runs'/backbone/f'row_{row:02d}';key=f'{backbone}/row_{row:02d}'
            if not (run/'COMPLETE.json').exists():missing.append(key);continue
            config=json.loads((run/'CONFIG.json').read_text());complete=json.loads((run/'COMPLETE.json').read_text())
            assert json.loads((run/'PREFLIGHT.json').read_text())['passed']
            assert complete['frozen_unchanged'] and complete['epochs']==(0 if row==1 else 4)
            assert not config['test_selection'] and config['job']['seed']==2026
            initial.append(config['initialization']['sha256'])
            assert sha(Path(config['initialization']['snapshot']))==config['initialization']['sha256']
            if row>1:
                ck=json.loads((run/'CHECKPOINT.json').read_text());assert sha(Path(ck['path']))==ck['sha256']
                history=read(run/'history.jsonl');assert [h['epoch'] for h in history]==[1,2,3,4]
            for split,count in [('val',465),('test',1036)]:
                pred=read(run/f'{split}_predictions.jsonl');gt=read(ROOT/'data'/f'{split}.jsonl')
                assert len(pred)==len(gt)==count and len({p['qid'] for p in pred})==count
                assert {p['qid'] for p in pred}=={g['qid'] for g in gt}
                if row==2:
                    baseline=read(ROOT/'runs'/backbone/'row_01'/f'{split}_predictions.jsonl')
                    key_for_invariance='pred_relevant_windows_pre_exist' if backbone=='flashvtg' else 'pred_relevant_windows'
                    assert {p['qid']:p[key_for_invariance] for p in baseline}=={p['qid']:p[key_for_invariance] for p in pred},(backbone,split,'decoupling must preserve raw localizer')
                reports=json.loads((run/f'{split}_metrics.json').read_text())
                for p in pred:
                    assert math.isfinite(p['pred_exist_score']) and 0<=p['pred_exist_score']<=1
                    assert all(all(math.isfinite(v) for v in w) and w[1]>=w[0] for w in p['pred_relevant_windows'])
                    zero=[dict(rank=i+1,window=w) for i,w in enumerate(p['pred_relevant_windows']) if w[1]==w[0]]
                    if zero:
                        assert backbone=='flashvtg','Unexpected degenerate windows outside the native FlashVTG exporter'
                        degenerate_windows.append(dict(run=key,scope='formal' if row<=6 else 'additional_diagnostic',split=split,qid=p['qid'],windows=zero))
                    # G-mIoU cleans degenerate windows; mAP/mR retain their own
                    # official raw-top-K semantics. Do not rewrite native exports.
                    assert all(w[1]>w[0] for w in _clean_pred_windows(p['pred_relevant_windows'],max_pred_windows=10))
                for tau in [.4,.5,.6]:
                    report=reports[str(tau)]
                    actual=evaluate_gmr(pred,gt,cls_thresholds=(tau,),gmiou_cls_threshold=tau,map_num_workers=0,verbose=False)
                    for k in ['mAP','mR+@5','G-mIoU@1','Rej-F1@'+str(tau)]:assert actual['brief'][k]==report['brief'][k],(key,split,tau,k)
                    assert abs(report['brief']['N-acc']-actual['GMR-CLS']['per_threshold'][str(tau)]['Rej-R'])<.006
                checks.append(dict(run=key,scope='formal' if row<=6 else 'additional_diagnostic',split=split,count=count,prediction_sha256=sha(run/f'{split}_predictions.jsonl'),official_recomputed=True))
        assert len(set(initial))<=1,(backbone,initial)
        component=ROOT/'reports'/f'{backbone}_COMPONENT_CHECK.json'
        state_checks=json.loads(component.read_text())['checks'] if component.exists() else []
        if sum(c['row']<=6 for c in state_checks)<6:missing.append(backbone+'/formal_state_verification')
        if not any(c['row']==7 for c in state_checks):missing.append(backbone+'/additional_diagnostic_state_verification')
        fusion=ROOT/'reports'/f'{backbone}_FUSION_BEHAVIOR.json'
        if not fusion.exists() or not json.loads(fusion.read_text())['passed']:missing.append(backbone+'/additional_fusion_behavior_verification')
    for relative,expected in protocol['source_sha256'].items():assert sha(ROOT/relative)==expected,('protocol source changed',relative)
    formal_checks=[c for c in checks if c['scope']=='formal']
    extra_checks=[c for c in checks if c['scope']=='additional_diagnostic']
    formal_missing=[m for m in missing if m.endswith('formal_state_verification') or ('/row_' in m and int(m[-2:])<=6)]
    extra_missing=[m for m in missing if m not in formal_missing]
    result=dict(complete=not missing,
        formal=dict(required_experiments=30,backbones=5,configurations=6,checked_split_pairs=len(formal_checks),checked_threshold_evaluations=3*len(formal_checks),complete=not formal_missing and len(formal_checks)==60,missing=formal_missing),
        additional_diagnostics=dict(required_experiments=5,checked_split_pairs=len(extra_checks),checked_threshold_evaluations=3*len(extra_checks),complete=not extra_missing and len(extra_checks)==10,missing=extra_missing),
        missing=missing,checks=checks)
    (ROOT/'reports/NATIVE_DEGENERATE_WINDOWS.json').write_text(json.dumps(dict(note='Native FlashVTG zero-width candidates are preserved. G-mIoU cleans them before top-K gating; localization uses the official raw-top-K evaluator unchanged. No predictions or metrics are repaired for this audit.',entries=degenerate_windows),indent=2)+'\n')
    path=ROOT/'reports/AUDIT.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
if __name__=='__main__':main()
