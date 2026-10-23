"""Recompute Table 4 validation cells from the actual selected FlashVTG archives."""
import copy,json,hashlib,csv
from pathlib import Path
from official_eval.eval_main import evaluate_gmr
from official_eval.metrics import compute_G_mIoU
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
SOURCE=Path('/home/guoxiangyu/GMR_Unseen/experiments/kim_gmr_autoresearch_20261019/runs/P1_A_S4_noFusion_correctinit_s2026')
def read(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def evaluate(name,pred,gt):
    m=evaluate_gmr(pred,gt,cls_thresholds=(.5,),gmiou_cls_threshold=.5,map_num_workers=0,verbose=False)
    r={'setting':name,**{k:m['brief'][k] for k in ['mAP','mR@1','mR@5','mR+@5','G-mIoU@1']}}
    if not any(len(g['relevant_windows'])>=2 for g in gt):r['mR+@5']=None
    return r,m

def main():
    gt=read(REPO/'data/val.jsonl');pre=read(SOURCE/'validation_predictions_pre_nms.jsonl');post=read(SOURCE/'validation_predictions.jsonl')
    assert len(gt)==len(pre)==len(post)==465
    native=copy.deepcopy(pre)
    for p in native:p['pred_exist_score']=1.
    refined=copy.deepcopy(pre)
    for p in refined:p['pred_relevant_windows']=p['pred_relevant_windows_pre_exist']
    oracle=copy.deepcopy(post);lookup={g['qid']:bool(g['relevant_windows']) for g in gt}
    for p in oracle:p['pred_exist_score']=float(lookup[p['qid']])
    configs=[('Native Pre-NMS (noPH)',native,gt),('Gaussian-RoI Pre-NMS (learned gate)',refined,gt),('DeRoI Post-NMS',post,gt)]
    for name,n in [('Single',1),('Multi',2)]:
        sub=[g for g in gt if (len(g['relevant_windows'])==1 if n==1 else len(g['relevant_windows'])>=2)];ids={g['qid'] for g in sub}
        configs.append((name,[p for p in post if p['qid'] in ids],sub))
    configs.append(('Oracle Post-NMS',oracle,gt))
    output=[];full={}
    for name,p,g in configs:
        row,m=evaluate(name,p,g);output.append(row);full[name]=m
    neg=[g for g in gt if not g['relevant_windows']];ids={g['qid'] for g in neg}
    rejection=100*sum(p['pred_exist_score']<=.5 for p in post if p['qid'] in ids)/len(neg)
    output.insert(5,dict(setting='No-Moment',mAP=None,**{'mR@1':None,'mR@5':None,'mR+@5':None,'G-mIoU@1':round(rejection,2)}))
    # These are provenance assertions against the actual manuscript's recorded metrics.
    assert [r['mAP'] for r in output[:3]]==[26.55,26.81,28.63],output
    print(json.dumps(output,indent=2),flush=True)
    assert output[0]['G-mIoU@1']==16.47 and output[2]['G-mIoU@1']==51.78
    assert output[5]['G-mIoU@1']==95.24 and output[6]['G-mIoU@1']==61.65
    out=ROOT/'table4';out.mkdir(exist_ok=True)
    with (out/'validation.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=output[0]);w.writeheader();w.writerows(output)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/'validation_predictions_pre_nms.jsonl',SOURCE/'validation_predictions.jsonl',REPO/'data/val.jsonl']}
    (out/'EVIDENCE.json').write_text(json.dumps(dict(rows=output,official_reports=full,source_sha256=hashes,checkpoint=json.loads((REPO/'provenance/EXTERNAL_CHECKPOINTS.json').read_text())['S4Fusion'],correction='Native pre-NMS noPH G-mIoU@1 is 16.47; old 16.49 belonged to post-NMS always-accept, a different prediction stage.',note='Same trained checkpoint throughout; native proposals include trained LoRA. These rows isolate inference stages, not training components. No-Moment localization undefined; retain dashes.'),indent=2)+'\n')
    tex=REPO/'write/DeRoI_GMR_Paper.tex';text=tex.read_text();backup=out/'DeRoI_GMR_Paper.before.tex'
    if not backup.exists():backup.write_text(text)
    labels=['Native Pre-NMS Proposals', '+ Gaussian-RoI Refiner', '+ DeRoI-GMR Post-NMS',r'\quad -- Single-Moment Subset',r'\quad -- Multi-Moment Subset',r'\quad -- No-Moment Subset','Oracle Presence Ceiling']
    lines=text.splitlines()
    for i,line in enumerate(lines):
        for label,row in zip(labels,output):
            if line.startswith(label):
                fields=line.split(' & ')
                for k,key in enumerate(['mAP','mR@1','mR@5','mR+@5','G-mIoU@1'],1):
                    value=row[key]
                    if value is not None:
                        rendered=f'{value:.2f}'
                        if k==5:rendered+=r' \\'
                        fields[k]=rendered
                lines[i]=' & '.join(fields)
    text='\n'.join(lines)+'\n'
    text=text.replace('progressive refinement and oracle presence ceiling on Validation', 'inference-stage refinement and oracle presence ceiling on Validation')
    text=text.replace('Test set compared to \\texttt{FlashVTG + GMR Adapter}.}', 'Test set compared to \\texttt{FlashVTG + GMR Adapter}. All validation stages share the same trained checkpoint, including LoRA. Localization metrics are ungated; G-mIoU uses the indicated presence rule. Dashes denote metrics undefined for the subset.}')
    text=text.replace('+ Gaussian-RoI Refiner (Pre-NMS) &', '+ Gaussian-RoI Refiner (Pre-NMS, $\\tau=0.5$) &')
    tex.write_text(text)
    (out/'README.md').write_text('Table 4 validation values recomputed with the official evaluator from the selected FlashVTG epoch-4 archives. See EVIDENCE.json for hashes and all metric details. Same checkpoint and LoRA for all inference stages. Dashes on negative-only localization remain because those metrics have no positive ground truth. Native noPH uses no gate; refined pre-NMS and post-NMS use tau=0.5.\n')
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()
