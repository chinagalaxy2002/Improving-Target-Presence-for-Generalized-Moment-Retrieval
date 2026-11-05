"""Post-training evaluation of the same frozen best on both Test splits."""
import json
from pathlib import Path
from . import checkpoint as ck
from .backbones.registry import build,get_spec
from .data import SoccerDataset
from .evaluation import evaluate
from .geq import GEQFull
from .io import dump_json,dump_jsonl,fingerprint,sha256,read_jsonl
from .predict import predict


def test(config,checkpoint,output,device='cuda:0'):
    checkpoint=Path(checkpoint).resolve();output=Path(output).resolve()
    with ck.run_lock(output):
        saved=ck.load(checkpoint)
        completion=json.loads((checkpoint.parent/'complete.json').read_text())
        checkpoint_hash=sha256(checkpoint)
        if completion['best_sha256']!=checkpoint_hash or completion['identity']!=saved['identity']:
            raise ValueError('Only finalized validation-selected best.pt can be tested')
        if saved['config']['identity']!=config['identity']:raise ValueError('Test configuration differs from the training configuration')
        if saved['source']!=ck.source_identity():raise ValueError('Source code differs from the training checkpoint')
        model,_=build(config,device)
        if saved['stage']=='geq_full':model=GEQFull(model,get_spec(config['backbone'])).to(device)
        model.load_state_dict(saved['model'],strict=True)
        results={}
        for split in ('test_standard','test_full'):
            dataset=SoccerDataset(config,split,get_spec(config['backbone']))
            assets=ck.asset_identity([dataset])
            identity=fingerprint(dict(checkpoint_sha256=checkpoint_hash,assets=assets,source=saved['source'],split=split))
            target=output/split;target.mkdir(exist_ok=True)
            marker=target/'complete.json'
            valid=False
            if marker.exists():
                prior=json.loads(marker.read_text())
                valid=prior['identity']==identity and all((target/file).is_file() and sha256(target/file)==prior[key]
                    for file,key in [('predictions.jsonl','predictions_sha256'),('metrics.json','metrics_sha256')])
            if valid:
                # Also enforce row identity even for a structurally valid cached artifact.
                predictions=read_jsonl(target/'predictions.jsonl')
                if len(predictions)!=len(dataset):raise ValueError('Cached prediction coverage mismatch')
                results[split]=json.loads((target/'metrics.json').read_text())
                continue
            predictions=predict(model,dataset,config,device)
            metrics=evaluate(predictions,dataset.data)
            metrics['provenance']=dict(split=split,train_split='Standard',checkpoint_sha256=checkpoint_hash,
                selected_epoch=saved['selector']['best_epoch'],selection_metric=saved['selector']['metric'],
                label_sha256=sha256(dataset.path),feature_asset_identity=assets,
                full_label=config['data']['full_label'] if split=='test_full' else None,
                test_used_for_selection=False,source=saved['source'])
            dump_jsonl(target/'predictions.jsonl',predictions);dump_json(target/'metrics.json',metrics)
            dump_json(marker,dict(identity=identity,predictions_sha256=sha256(target/'predictions.jsonl'),
                                 metrics_sha256=sha256(target/'metrics.json'),rows=len(predictions)))
            results[split]=metrics
        return results
