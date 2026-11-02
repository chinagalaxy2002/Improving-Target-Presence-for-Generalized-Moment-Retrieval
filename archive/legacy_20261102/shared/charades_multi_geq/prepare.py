from paths import *
from collections import defaultdict,Counter
import numpy as np

def main():
 plan=PROJECT/'charades/multi_geq_plan_20261101'
 expected=json.loads((plan/'data_audit.json').read_text())
 proposed=json.loads((plan/'train_conflict_exclusions.proposed.json').read_text())['exclusions']
 train=rows(SPLITS/'train.jsonl');groups=defaultdict(list)
 for i,r in enumerate(train):groups[(r['vid'],' '.join(r['query'].lower().split()))].append(i)
 excluded={i for ids in groups.values() if any(train[j]['relevant_windows'] for j in ids) for i in ids if not train[i]['relevant_windows']}
 assert excluded=={x['row_index'] for x in proposed} and len(excluded)==21
 for x in proposed:
  r=train[x['row_index']];assert all(r[k]==x[k] for k in ('qid','vid','query','kind'))
  assert not r['relevant_windows']
  for e in x['positive_evidence']:assert train[e['row_index']]['relevant_windows']==e['relevant_windows']
 audit={};allrows=[]
 for split in ('train','val','test_id','test_ood'):
  p=SPLITS/f'{split}.jsonl';assert sha(p)==expected['splits'][split]['source_file_sha256']
  raw=rows(p);view=[]
  for i,r in enumerate(raw):
   assert len(r['relevant_windows'])<=1
   if split=='train' and i in excluded:continue
   m=dict(r,original_qid=r['qid'],source_row_index=i,source_line_number=i+1,evaluation_id=f'{split}:{i:06d}')
   m['qid']=m['evaluation_id'];m['sample_id']=m['evaluation_id'];view.append(m)
  target=OUT/'data'/f'{split}.jsonl'
  if target.exists():assert rows(target)==view
  else:jsonl(target,view)
  audit[split]=dict(rows=len(view),positive=sum(bool(r['relevant_windows']) for r in view),sha256=sha(target),source_sha256=sha(p),exact_duplicate_extra_rows=expected['splits'][split]['exact_duplicate_extra_rows'])
  allrows+=view
 assert (audit['train']['rows'],audit['train']['positive'])==(27788,11199)
 videos=sorted({r['vid'] for r in allrows});qids=sorted({str(r['original_qid']) for r in allrows})
 lengths={};payloads=0
 for vid in videos:
  arrays=[]
  for folder,dim in [('clip',512),('slowfast',2304)]:
   with np.load(FEATURES/folder/f'{vid}.npz') as z:x=z['features'];assert x.ndim==2 and x.shape[1]==dim and np.isfinite(x).all();arrays.append(len(x));payloads+=1
  assert arrays[0]==arrays[1] and arrays[0]>0,(vid,arrays)
  lengths[vid]=arrays[0]
 for qid in qids:
  with np.load(FEATURES/'clip_text'/f'qid{qid}.npz') as z:
   x=z['last_hidden_state'];mask=z['attention_mask'];assert x.shape==(77,512) and mask.shape==(77,) and mask.sum()>0 and np.isfinite(x).all();payloads+=1
 for r in allrows:
  assert abs(lengths[r['vid']]*2-float(r['duration']))<=1.01
  for a,b in r['relevant_windows']:assert 0<=a<b<=float(r['duration'])+1e-5
 maximum=max(lengths.values())
 audit.update(passed=True,excluded_negative_rows=sorted(excluded),max_video_length=maximum,max_v_l=max(128,maximum),full_finite_payloads_checked=payloads,video_lengths=lengths,text_qids=len(qids),mR_plus='N/A',multi_gt_coverage='N/A')
 dump(OUT/'data/audit.json',audit);dump(OUT/'data/exclusions.json',proposed)
 dump(OUT/'protocol.json',dict(seed=2023,backbones=BACKBONES,methods=METHODS,stage_a=dict(max_epochs=200,patience=20,lr=3e-5,selection='positive Val mAP'),stage_b=dict(max_epochs=20,patience=6,lr_base=1e-5,lr_geq=3e-5,selection='mean Val G-mIoU@1 0.4/0.5/0.6'),batch_sizes=dict(moment_detr=8,eatr=32,qd_detr=32,flashvtg=8),amp=False,num_workers=2,weight_decay=1e-4,gradient_clip=.1,train_sha256=audit['train']['sha256'],test_used_for_selection=False,threshold_operator='>',flash_multi='Multi-TQAux structural adaptation',o2m_k=2,aux_queries=24,instruction_tokens=4,route_weights=[.15,.15],max_v_l=audit['max_v_l']))
 print(json.dumps({k:v for k,v in audit.items() if k!='video_lengths'},indent=2))
if __name__=='__main__':main()
