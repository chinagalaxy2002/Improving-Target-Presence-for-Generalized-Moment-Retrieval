"""Audited identity migration, NOT retraining or unchecked checkpoint loading."""
import copy,json
from pathlib import Path
import torch
from soccergmr import checkpoint as ck
from soccergmr.config import stage_config_identity
from soccergmr.io import fingerprint,sha256,dump_json

def migrate(parent,config,target):
 parent=Path(parent);target=Path(target);old=ck.load(parent,stage='gmr');m=json.loads((parent.parent/'complete.json').read_text())
 assert m['status']=='complete' and m['best_sha256']==sha256(parent) and m['identity']==old['identity']
 expected=fingerprint(dict(config=stage_config_identity(old['config'],'gmr'),source=old['source'],train_val_assets=old['train_val_assets'],stage='gmr'))
 assert old['identity']==expected
 previous={k:v for k,v in old['config'].items() if k!='identity'};current=copy.deepcopy({k:v for k,v in config.items() if k!='identity'})
 assert current['clip'].pop('geq_weight')==.1
 assert previous==current, 'Only the new Stage-2 auxiliary weight may change'
 # Stage-1 model/data/native criterion code must remain byte-identical.
 root=Path(__file__).resolve().parents[1];base=Path('/home/guoxiangyu/SoccerGMR-GEQ-clip-v2')
 frozen=json.loads((base/'runs/clip_v2/FREEZE.json').read_text())
 assert frozen['source']==old['source']
 for file,digest in frozen['files'].items():assert sha256(base/file)==digest
 for file in ['src/soccergmr/data.py','src/soccergmr/geq.py','src/soccergmr/backbones/loss.py','src/soccergmr/backbones/registry.py']:
  assert sha256(root/file)==sha256(base/file)
 source=ck.source_identity();identity=fingerprint(dict(config=stage_config_identity(config,'gmr'),source=source,train_val_assets=old['train_val_assets'],stage='gmr'))
 migration=dict(schema='joint-geq-parent-v1',original_path=str(parent.resolve()),original_sha256=sha256(parent),trained_source=old['source'],compatible_source=source,original_identity=old['identity'],new_identity=identity,allowed_difference='Stage2 auxiliary coefficient .1; GMR weights unchanged; native/head strict state loading preserved',not_retrained=True)
 out=dict(old);out.pop('format');out.update(config=config,source=source,identity=identity,migration=migration)
 target.mkdir(parents=True,exist_ok=True)
 if (target/'best.pt').exists():
  saved=ck.load(target/'best.pt',identity,'gmr');assert saved['migration']==migration
 else:ck.save(target/'best.pt',out)
 saved=ck.load(target/'best.pt',identity,'gmr');assert set(saved['model'])==set(old['model']) and all(torch.equal(v,saved['model'][k]) for k,v in old['model'].items())
 result=dict(m,identity=identity,source=source,best_sha256=sha256(target/'best.pt'),migration=migration)
 dump_json(target/'complete.json',result);dump_json(target/'MIGRATION.json',migration)
 return target/'best.pt'
