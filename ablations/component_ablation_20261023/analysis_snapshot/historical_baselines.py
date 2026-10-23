"""Document export-level differences from the archived historical baselines."""
import gzip,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
def read(p):
    data=gzip.open(p,'rt').read() if p.suffix=='.gz' else p.read_text()
    return [json.loads(x) for x in data.splitlines() if x.strip()]
def write():
    result=[]
    for b in ['moment_detr','qd_detr','cg_detr','eatr','flashvtg']:
        old_report=REPO/'backbonefolder'/b/'gmr/summary_table_metrics_tau_0.4.json'
        report=json.loads(old_report.read_text());new=json.loads((ROOT/'runs'/b/'row_01/test_metrics.json').read_text())['0.4']
        old_prediction=REPO/'backbonefolder'/b/'gmr/test_predictions.jsonl'
        if b=='cg_detr':old_prediction=old_prediction.with_name('final_test_predictions.jsonl')
        if b=='flashvtg':old_prediction=REPO/'results/E0/test_predictions.jsonl.gz'
        old={p['qid']:p for p in read(old_prediction)};current={p['qid']:p for p in read(ROOT/'runs'/b/'row_01/test_predictions.jsonl')}
        assert set(old)==set(current)
        changed=[]
        for qid in old:
            a=[w[:2] for w in old[qid]['pred_relevant_windows'][:10]];c=[w[:2] for w in current[qid]['pred_relevant_windows'][:10]]
            if a!=c:changed.append(qid)
        keys=['mAP','mR+@5','G-mIoU@1','Rej-F1@0.4']
        result.append(dict(backbone=b,archived_metrics={k:report['brief'][k] for k in keys},new_metrics={k:new['brief'][k] for k in keys},delta_pp={k:round(new['brief'][k]-report['brief'][k],8) for k in keys},top10_coordinate_differences_qids=changed,old_prediction_sha256=hashlib.sha256(old_prediction.read_bytes()).hexdigest(),note='New paired rows use the same exporter throughout. This records historical differences; it does not identify a unique numerical cause.'))
    out=ROOT/'reports/HISTORICAL_BASELINE_COMPARISON.json';out.write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':print(json.dumps(write(),indent=2))
