"""Audit annotations against feature archives without extracting or dropping rows."""
import argparse
import collections
import json
import pathlib
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=pathlib.Path, required=True)
    parser.add_argument('--repo', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    args = parser.parse_args()
    labels = args.data_root / 'd_gmr_two_stage/dataset/data/label/Standard'
    features = args.data_root / 'd_gmr_two_stage/dataset/Soccergmr/standard'
    inventory = {}
    for kind in ('clip', 'slowfast', 'clip_text'):
        with tarfile.open(features / f'{kind}.tar') as archive:
            inventory[kind] = {pathlib.PurePosixPath(m.name).name for m in archive if m.isfile()}
    report = {'repo': str(args.repo.resolve()), 'commit': subprocess.check_output(
        ['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'seed': 2023, 'splits': {}, 'errors': [], 'feature_archives': str(features.resolve())}
    seen = set()
    for split in ('train', 'val', 'test'):
        path = labels / f'{split}.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        counts = collections.Counter()
        missing = []
        for row in rows:
            for key in ('qid', 'query', 'vid', 'duration', 'relevant_windows'):
                if key not in row:
                    raise ValueError(f'{path}: missing {key}')
            qid = str(row['qid'])
            if qid in seen:
                report['errors'].append(f'duplicate qid: {qid}')
            seen.add(qid)
            windows = row['relevant_windows']
            counts[len(windows)] += 1
            for start, end in windows:
                if not 0 <= start < end <= row['duration']:
                    report['errors'].append(f'invalid window: {split}/{qid}')
            if len(windows) > 8:
                report['errors'].append(f'GT exceeds max_windows: {split}/{qid}')
            for kind in inventory:
                name = f'qid{qid}.npz' if kind == 'clip_text' else f"{row['vid']}.npz"
                if name not in inventory[kind]:
                    missing.append({'qid': qid, 'kind': kind, 'expected': name})
        report['splits'][split] = {'path': str(path.resolve()), 'samples': len(rows),
            'gt_histogram': dict(counts), 'missing_features': missing}
        if missing:
            report['errors'].append(f'{split}: {len(missing)} missing feature references')
    report['passed'] = not report['errors']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({'passed': report['passed'], 'errors': report['errors'], 'report': str(args.out)}, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
