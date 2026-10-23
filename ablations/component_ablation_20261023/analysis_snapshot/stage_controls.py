"""Match gating across all three archived inference stages (no new training)."""
import copy,csv,json
from table4 import ROOT,REPO,SOURCE,read,evaluate

gt=read(REPO/'data/val.jsonl');pre=read(SOURCE/'validation_predictions_pre_nms.jsonl');post=read(SOURCE/'validation_predictions.jsonl')
refined=copy.deepcopy(pre)
for p in refined:p['pred_relevant_windows']=p['pred_relevant_windows_pre_exist']
truth={g['qid']:bool(g['relevant_windows']) for g in gt}
records=[]
for stage,data in [('Native Pre-NMS (LoRA checkpoint)',pre),('Refined Pre-NMS',refined),('Refined Post-NMS',post)]:
    for gate in ['noPH','learned_tau05','oracle']:
        pred=copy.deepcopy(data)
        if gate!='learned_tau05':
            for p in pred:p['pred_exist_score']=1. if gate=='noPH' else float(truth[p['qid']])
        row,report=evaluate(stage,pred,gt);row['presence']=gate;records.append(row)
out=ROOT/'table4'
with (out/'matched_inference_stages.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
(out/'matched_inference_stages.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
