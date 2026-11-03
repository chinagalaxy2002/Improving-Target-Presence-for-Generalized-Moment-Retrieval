"""Verify archived artifact hashes and optionally recompute metrics with labels."""
import argparse,csv,gzip,hashlib,json
from pathlib import Path
from soccergmr.evaluation import evaluate
ROOT=Path(__file__).resolve().parents[1]
def digest(data):return hashlib.sha256(data).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--label-root',type=Path);a=p.parse_args();root=ROOT/'results/clipseg'
    manifest=json.loads((root/'PUBLICATION_MANIFEST.json').read_text())
    for file,h in manifest['files'].items():assert digest((ROOT/file).read_bytes())==h,file
    selection=json.loads((root/'selection.json').read_text());assert not selection['test_used'];assert selection['chosen']==max(selection['candidates'],key=lambda c:(c['val_g1'],-c['weight']))
    count=0
    for folder in sorted((root/'execution').glob('*/g*_test/test_*')):
        split=folder.name;marker=json.loads((folder/'complete.json').read_text());raw=gzip.decompress((folder/'predictions.jsonl.gz').read_bytes());rows=[json.loads(x) for x in raw.decode().splitlines() if x]
        assert digest(raw)==marker['predictions_sha256'];assert digest((folder/'metrics.json').read_bytes())==marker['metrics_sha256']
        assert len(rows)==marker['rows']==(1036 if split=='test_standard' else 2970)
        assert len({r['qid'] for r in rows})==len(rows)
        metrics=json.loads((folder/'metrics.json').read_text());assert not metrics['provenance']['test_used_for_selection']
        if a.label_root:
            label=a.label_root/('Standard' if split=='test_standard' else 'Full')/'test.jsonl';assert digest(label.read_bytes())==metrics['provenance']['label_sha256']
            gt=[json.loads(x) for x in label.read_text().splitlines() if x];m=evaluate(rows,gt)
            for key in ['brief','mAP_detail','gmiou']:assert m[key]==metrics[key]
        count+=1
    assert count==12
    suffix=selection['chosen']['name'][2:];names={'A':'A','B':'B_'+suffix,'C':'C_'+suffix,'D':'D_'+suffix}
    for row in csv.DictReader((root/'METRICS.csv').open()):
        m=json.loads((root/'execution'/names[row['group']]/'geq_full_test'/row['split']/'metrics.json').read_text())
        assert float(row['mAP'])==m['brief']['mAP']
        assert abs(float(row['G1_mean'])-sum(m['gmiou'][str(t)]['G-mIoU@1'] for t in [.4,.5,.6])/3)<1e-9
    native=json.loads((ROOT/'src/soccergmr/backbones/SOURCE_MANIFEST.json').read_text())['backbones']['moment_detr']['sources']
    for file,h in native.items():assert digest((ROOT/'src/soccergmr/backbones/vendor/moment_detr'/file).read_bytes())==h
    print(f'PASS: publication hashes, native snapshot, Val lambda selection, 12 Test prediction coverages/hashes, result table'+('; official metrics recomputed' if a.label_root else '. Supply --label-root to recompute official metrics.'))
if __name__=='__main__':main()
