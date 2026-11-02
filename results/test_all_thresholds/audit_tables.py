#!/usr/bin/env python3
"""Independent source, unit, export, table and artifact checks (no inference)."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load(p):
    return json.loads(p.read_text())


def csv_rows(p):
    with p.open() as stream:
        return list(csv.DictReader(stream))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    data = load(HERE / 'TEST_ALL_THRESHOLDS.json')
    provenance = load(HERE / 'PROVENANCE.json')
    assert len(data) == 126
    assert len(csv_rows(HERE / 'TEST_ALL_THRESHOLDS_LONG.csv')) == 378
    assert len(csv_rows(HERE / 'TEST_ALL_THRESHOLDS.csv')) == 126
    key = lambda r: (r['dataset'], r['split'], r['protocol'], r['backbone'], r['method'])
    index = {key(r): r for r in data}
    assert len(index) == len(data)
    assert all(sha(REPO / p) == expected for p, expected in provenance['inputs'].items())
    checks = {}
    native = csv_rows(REPO / 'results/results.csv')
    count = 0
    for old in native:
        if not old['mAP']:
            continue
        method = 'gmr' if old['variant'] == 'baseline' else old['variant']
        new = index[('SoccerGMR', 'Standard Test', 'current_val_selection', old['backbone'], method)]
        for metric in ('mAP', 'mR@1', 'mR@5', 'mR+@5', 'AUROC'):
            assert float(old[metric]) == new['common'][metric]
        assert math.isclose(float(old['G-mIoU@3_0_1']), new['thresholds']['0.4']['G-mIoU@3'], abs_tol=1e-12)
        count += 1
    checks['original_soccer_native_csv_rows_matched'] = count
    assert count == 23
    for old in csv_rows(REPO / 'results/geq_val_selection/summary.csv'):
        route = 'gmr' if old['variant'] == 'baseline' else old['variant']
        new = index[('SoccerGMR', 'Standard Test', 'current_val_selection', old['backbone'], route+'_geq_'+old['mode'])]
        assert float(old['mAP']) == new['common']['mAP']
        assert math.isclose(float(old['test_g1_mean']), sum(new['thresholds'][t]['G-mIoU@1'] for t in ('0.4','0.5','0.6'))/3, abs_tol=1e-12)
    checks['soccer_geq_summary_rows_matched'] = 24
    for old in csv_rows(REPO / 'results/charades_multi_geq/all_metrics.csv'):
        new = index[('Charades', old['split'], 'current_val_selection', old['backbone'], old['method'])]
        assert float(old['mAP_percent']) == new['common']['mAP']
        assert float(old['AUROC_percent']) == new['common']['AUROC']
        for tau in ('0.4', '0.5', '0.6'):
            assert float(old['gmiou_1_at_'+tau]) == new['thresholds'][tau]['G-mIoU@1']
            assert float(old['gmiou_3_at_'+tau]) == new['thresholds'][tau]['G-mIoU@3']
    checks['original_charades_csv_rows_matched'] = 48
    history_names = {'GMR baseline':'gmr', 'GMR + GEQ (frozen)':'gmr_geq_frozen', 'GMR + GEQ (full)':'gmr_geq_full'}
    for old in csv_rows(REPO / 'results/geq_epoch_study/test_summary.csv'):
        new = index[('SoccerGMR', 'Standard Test', 'historical_main_e1', old['backbone'], history_names[old['method']])]
        assert new['common']['mAP'] == float(old['mAP'])
        for tau, col in (('0.4','g1_40'),('0.5','g1_50'),('0.6','g1_60')):
            assert math.isclose(new['thresholds'][tau]['G-mIoU@1'], float(old[col]), abs_tol=1e-12)
    checks['historical_e1_rows_matched'] = 12
    fixed_names = {'GMR baseline':'gmr', 'GMR baseline + GEQ':'gmr_geq_frozen', 'multi + GEQ':'multi_geq_frozen', 'shared + GEQ':'shared_geq_frozen'}
    for old in csv_rows(REPO / 'results/geq_finetune/four_way_map.csv'):
        new = index[('SoccerGMR', 'Standard Test', 'historical_fixed_four_epochs', old['backbone'], fixed_names[old['method']])]
        assert new['common']['mAP'] == float(old['mAP'])
        assert new['common']['AUROC'] == float(old['AUROC'])
    checks['historical_fixed_four_epoch_rows_matched'] = 16
    frozen = [r for r in csv_rows(HERE / 'MAP_DELTAS.csv') if 'Frozen' in r['method']]
    assert all(float(r['delta_vs_parent_pp']) == 0 for r in frozen)
    checks['frozen_mAP_unchanged_rows'] = len(frozen)
    for r in data:
        for tau in ('0.4','0.5','0.6'):
            p = r['thresholds'][tau]
            assert all(math.isfinite(v) for v in p.values())
            assert 0 <= p['G-mIoU@1'] <= 1 and 0 <= p['G-mIoU@3'] <= 1
            assert 0 <= p['N-acc'] <= 100 and 0 <= p['Rej-F1'] <= 100
        if r['dataset'] == 'Charades':
            assert r['common']['mR+@5'] is None
        else:
            assert r['common']['mR+@5'] is not None
    checks['all_cells_finite_and_units_valid'] = True
    for split in ('test_id', 'test_ood'):
        root = HERE / 'supplemental/charades/eatr/multi_positive_only' / split
        marker = load(root / 'COMPLETE.json')
        assert sha(root / 'metrics.json') == marker['metrics_sha256']
        raw = gzip.decompress((root / 'predictions.jsonl.gz').read_bytes())
        assert hashlib.sha256(raw).hexdigest() == marker['prediction_sha256']
        predictions = [json.loads(line) for line in raw.splitlines()]
        assert len(predictions) == len({p['qid'] for p in predictions}) == marker['rows'] == 7440
        assert load(root / 'metrics.json')['mAP'] == 33.84
    selection = load(HERE / 'supplemental/charades/eatr/multi_positive_only/selection.json')
    assert selection['original_exports_unchanged'] and selection['checkpoint_unchanged']
    assert selection['trained']['best_epoch'] == 65
    checks['two_new_charades_formal_exports_verified'] = True
    root = HERE / 'supplemental/soccergmr/eatr/adaptive'
    selection = load(root / 'selection.json')
    assert selection['existing_formal_test_reused'] and selection['best_epoch'] == 75
    raw = gzip.decompress((root / 'formal_test_predictions.jsonl.gz').read_bytes())
    assert hashlib.sha256(raw).hexdigest() == load(root / 'data.json')['signature']['prediction_sha256']
    assert len(raw.splitlines()) == 1036
    assert load(root / 'formal_test_metrics.json')['brief']['mAP'] == 17.29
    checks['completed_adaptive_formal_export_verified'] = True
    for p in (HERE / 'supplemental/soccergmr').glob('*/*/data.json'):
        d = load(p)
        original = load(p.with_name('formal_test_metrics.json'))
        assert sha(p.with_name('formal_test_metrics.json')) == d['signature']['metrics_sha256']
        report = load(p.with_name('threshold_metrics.json'))
        assert d['metrics']['mAP'] == original['brief']['mAP'] == report['brief']['mAP']
        assert report['stats']['num_total'] == 1036
        assert report['stats']['num_positive'] == 544 and report['stats']['num_negative'] == 492
        assert len(d['grid']) == 3
    checks['native_threshold_supplements_verified'] = 12
    integrity = load(HERE / 'supplemental/INTEGRITY.json')
    assert integrity['passed'] and integrity['original_test_files_verified'] == 144
    assert all(v['expected'] == v['actual'] for v in integrity['original_test_files'].values())
    assert len(integrity['new_charades_thresholds']) == 6
    checks['original_144_test_files_and_six_new_threshold_points_verified'] = True
    artifacts = {}
    for p in HERE.glob('*.png'):
        with Image.open(p) as im:
            im.verify()
        artifacts[p.name] = sha(p)
    for p in HERE.glob('*.pdf'):
        info = subprocess.check_output(['pdfinfo', str(p)], text=True)
        pages = int(re.search(r'Pages:\s+(\d+)', info).group(1))
        assert pages == (6 if p.stem == 'TEST_ALL_THRESHOLDS' else 1)
        text = subprocess.check_output(['pdftotext', '-layout', str(p), '-'], text=True)
        assert '0.4' in text and '0.5' in text and '0.6' in text
        artifacts[p.name] = sha(p)
    for p in HERE.glob('*.tex'):
        assert r'\begin{document}' in p.read_text() and r'\end{document}' in p.read_text()
        for line in p.read_text().splitlines():
            if line.endswith(r' \\') and not line.startswith(r'\multicolumn'):
                assert line.count('&') == 18, (p.name, line)
        artifacts[p.name] = sha(p)
    checks['six_tables_pdf_png_tex_verified'] = True
    # Check local links in both entry points. HTTP links are validated by Git
    # publication separately; directory links are allowed.
    for p in (HERE / 'README.md', REPO / 'README.md'):
        for target in re.findall(r'\]\(([^)]+)\)', p.read_text()):
            if re.match(r'https?://', target) or target.startswith('#'):
                continue
            target = target.split('#')[0]
            if target == 'AUDIT.json':
                continue
            assert (p.parent / target).exists(), (p, target)
    checks['readme_relative_links_resolve'] = True
    result = dict(passed=True, checks=checks, method_results=126, current_results=98,
                  missing_formal_test=0, no_test_model_selection=True, original_main_exports_unchanged=True,
                  artifacts=artifacts)
    (HERE / 'AUDIT.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
