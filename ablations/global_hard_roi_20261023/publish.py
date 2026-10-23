"""Stage the reviewed two-component experiment into the existing publication repo.

Creates files only. It does not commit, push, upload checkpoints or change results.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.resolve()
    assert json.loads((ROOT / 'reports/COMPLETION.json').read_text())['passed']
    assert json.loads((ROOT / 'reports/METRIC_VERIFICATION.json').read_text())['cells'] == 20
    target = repository / 'ablations' / ROOT.name
    assert not target.exists(), target
    target.mkdir(parents=True)
    forbidden = {'.ckpt', '.pt', '.pth', '.npz', '.npy', '.mp4', '.pyc', '.lock'}

    def copy(source, relative):
        if source.suffix in forbidden or '__pycache__' in source.parts or source.name.startswith('.capture'):
            return
        # Vendored CG-DETR ships unrelated datasets; retain code and licenses,
        # but publish only our Standard Soccer-GMR labels.
        if 'source' in source.parts and 'data' in source.parts and source.suffix in ['.json', '.jsonl']:
            return
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for folder in ['backends', 'baseline_source', 'configs', 'data', 'code', 'official_eval', 'reports', 'tables']:
        for source in (ROOT / folder).rglob('*'):
            if source.is_file():
                copy(source, source.relative_to(ROOT))
    for source in ROOT.glob('*'):
        if source.is_file() and source.suffix in ['.py', '.json', '.md', '.txt']:
            copy(source, source.name)
    for source in (ROOT / 'runs').rglob('*'):
        if source.is_file():
            copy(source, Path('results') / source.relative_to(ROOT / 'runs'))
    for source in (ROOT / 'logs').glob('*.log'):
        copy(source, Path('logs') / source.name)
    # Keep a complete file-level source record, including the original result
    # paths; publication only relocates runs/ to tracked results/.
    manifest = []
    secret = re.compile(rb'(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|https://[^\s/@]+:[^\s/@]+@github\.com)')
    for path in sorted(target.rglob('*')):
        if path.is_file():
            assert path.suffix not in forbidden and path.stat().st_size < 100_000_000, path
            assert not secret.search(path.read_bytes()), f'Credential detected in {path.relative_to(target)}'
            manifest.append(dict(path=str(path.relative_to(target)), bytes=path.stat().st_size, sha256=digest(path)))
    (target / 'PUBLICATION_FILES.json').write_text(json.dumps(dict(
        source=str(ROOT), relocation='runs/ -> results/; tensors/features omitted',
        files=manifest, file_count=len(manifest), credential_scan_passed=True), indent=2) + '\n')
    relative = str(target.relative_to(repository))
    readme = repository / 'README.md'
    text = readme.read_text()
    marker = '## 最新：仅 Global Event Query + Hard-RoI（固定 epoch 4）'
    assert marker not in text
    section = f'''{marker}

精简方案关闭 **LoRA、Gaussian-RoI、6D Saliency 与候选存在性融合**。在同一预训练 GMR 上冻结原参数，仅微调开启的新增模块，采用 **GMR / +Global / +Hard / +Global+Hard** 的完整 2×2 对照，五个 backbone 共 **20 个单元**：10 组新训练、10 组核验复用；全部微调固定第 4 个 epoch、seed 2026。下降项完整保留，结果为单种子探索性比较。

[实验设计与复现]({relative}/README.md) · [完整 Val/Test 结果]({relative}/reports/RESULTS.md) · [九指标 CSV]({relative}/reports/all_metrics.csv) · [相对 GMR 增益]({relative}/reports/gains_vs_gmr_pp.csv) · [独立数值复核]({relative}/reports/METRIC_VERIFICATION.json)

![仅 Global Event Query + Hard-RoI，Test，阈值0.4]({relative}/tables/ABLATION_TEST_tau_0.4.png)

[τ=0.4 PDF]({relative}/tables/ABLATION_TEST_tau_0.4.pdf) · [τ=0.5 PDF]({relative}/tables/ABLATION_TEST_tau_0.5.pdf) · [τ=0.6 PDF]({relative}/tables/ABLATION_TEST_tau_0.6.pdf) · [四个条件效应及交互]({relative}/reports/factorial_effects_pp.csv)

'''
    first = text.index('\n\n') + 2
    readme.write_text(text[:first] + section + text[first:])
    scope_path = repository / 'PUBLICATION_SCOPE.json'
    scope = json.loads(scope_path.read_text())
    scope[ROOT.name] = dict(path=relative, backbones=5, configurations=4, cells=20, newly_trained=10, reused=10,
                           frozen_baselines=5, adapted_cells=15, adaptation_epochs=4, seed=2026,
                           checkpoints_uploaded=False, predictions_uploaded=True, thresholds=[.4, .5, .6])
    scope['updated_utc'] = datetime.now(timezone.utc).isoformat()
    scope_path.write_text(json.dumps(scope, indent=2) + '\n')
    # Refresh the repository's existing file manifest without adding unrelated
    # untracked files from the working tree.
    manifest_path = repository / 'PUBLICATION_MANIFEST.json'
    publication = json.loads(manifest_path.read_text())
    existing = subprocess.check_output(['git', '-C', str(repository), 'ls-files', '-z']).decode().split('\0')
    paths = {repository / p for p in existing if p}
    paths.update(p for p in target.rglob('*') if p.is_file())
    entries = [dict(path=str(p.relative_to(repository)), bytes=p.stat().st_size, sha256=digest(p))
               for p in sorted(paths) if p.is_file() and p != manifest_path]
    publication.update(scope='Historical main comparisons, prior 30-cell ablation plus 5 fusion diagnostics, and new 20-cell Global Event Query / Hard-RoI study; checkpoints, features, videos and credentials omitted.',
                       files=entries, file_count=len(entries), total_bytes=sum(e['bytes'] for e in entries),
                       updated_utc=datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(publication, indent=2) + '\n')
    print(f'Staged {len(manifest)} experiment files in {relative}; no commit or push performed')


if __name__ == '__main__':
    main()
