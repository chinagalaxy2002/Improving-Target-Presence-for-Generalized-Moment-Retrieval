"""Reuse completed Moment E3 results, never relabel them as newly trained."""
import gzip,json,shutil
from pathlib import Path
from soccergmr.io import dump_json,sha256,read_jsonl
R=Path(__file__).resolve().parents[1];OUT=R/'runs/e3_eleven/formal/moment_detr'
V2=Path('/home/guoxiangyu/SoccerGMR-GEQ-clip-v2/runs/clip_v2/E3');JOINT=Path('/home/guoxiangyu/SoccerGMR-GEQ-clip-v2-joint/runs/clip_joint/E3')
records={}
for stage,source in [('gmr',V2),('geq_full',JOINT)]:
 p=source/stage;m=json.loads((p/'complete.json').read_text());assert sha256(p/'best.pt')==m['best_sha256'] and m['status']=='complete'
 records[stage]=dict(checkpoint_path=str(p/'best.pt'),checkpoint_sha256=m['best_sha256'],training_source=m['source'],original_directory=str(source),retrained=False)
 d=OUT/stage;d.mkdir(parents=True,exist_ok=True)
 for f in ['history.jsonl','complete.json','best_val_metrics.json']:shutil.copy2(p/f,d/f)
 for split,count in [('test_standard',1036),('test_full',2970)]:
  src=source/(stage+'_test')/split;marker=json.loads((src/'complete.json').read_text());assert marker['rows']==count
  for f,k in [('predictions.jsonl','predictions_sha256'),('metrics.json','metrics_sha256')]:assert sha256(src/f)==marker[k]
  dst=OUT/(stage+'_test')/split;dst.mkdir(parents=True,exist_ok=True)
  for f in ['predictions.jsonl','metrics.json','complete.json']:shutil.copy2(src/f,dst/f)
if (JOINT/'diagnostics').exists():shutil.copytree(JOINT/'diagnostics',OUT/'diagnostics',dirs_exist_ok=True)
dump_json(OUT/'REUSE.json',dict(method='Moment-DETR E3 GEQ+aux',retrained=False,source_policy='Historical source identities retained; these results are not attributed to the new common commit.',stages=records))
print('Moment E3: existing GMR and GEQ plus both Tests reused; zero retraining.')
