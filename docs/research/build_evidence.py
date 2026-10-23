"""Build ResearchPilot evidence cards and research memory from reviewed sources.

Citation decisions come from citation_verification.json, not citation counts.
The bundled exporter supplies BibTeX; types are corrected for known source forms.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('GMR_DOCUMENTATION_SOURCE_ROOT', HERE.parents[1])).resolve()


def dump(name, obj):
    (HERE/name).parent.mkdir(parents=True,exist_ok=True)
    (HERE/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def main():
    if not (ROOT/'ResearchPilot-Skills/scripts/export_bibtex.py').exists():
        raise SystemExit('Set GMR_DOCUMENTATION_SOURCE_ROOT to the full source workspace containing ResearchPilot-Skills, data/, results/, and code/. Saved evidence cards remain readable without it.')
    review=json.loads((HERE/'citation_verification.json').read_text())
    accepted=[]
    for index,ref in enumerate(review['references'],1):
        paper={k:ref.get(k) for k in ['title','authors','year','doi','url']}
        paper.update(evidence_id=f'L{index:02d}',abstract=None,claims=[ref['resolved_wording']],methods=[],
                     limitations=ref.get('issues',[]),decision='keep',
                     notes='Accepted for the limited literature claim in the independent review; not evidence for repository results.',
                     citation_status={'counts_available':False,'supporting':None,'contrasting':None,'mentioning':None},
                     evidence_level=ref['evidence_level'],evidence_locations=ref['evidence_locations'],
                     doi_registry=ref.get('doi_registry'),bibliographic_form='preprint' if 'arxiv.org/abs/' in ref['url'] else 'conference')
        if ref.get('retrieved_primary_url'):paper['url']=ref['retrieved_primary_url']
        accepted.append(paper)
    rejected=[];seen=set()
    for path in sorted((HERE/'retrieval').glob('openalex_search_*.json')):
        for paper in json.loads(path.read_text()).get('papers',[]):
            key=(paper.get('doi') or paper['title']).lower()
            if key in seen:continue
            seen.add(key)
            paper.update(decision='reject',notes='Off-topic broad-query hit; excluded by title screening. Not used in prose.')
            rejected.append(paper)
    maybe=[]
    for path in sorted((HERE/'retrieval').glob('openalex_focused_*.json')):
        for r in json.loads(path.read_text()).get('response',{}).get('results',[]):
            title=r['title']
            if title.lower() in {p['title'].lower() for p in accepted}:continue
            if 'video' not in title.lower():continue
            maybe.append(dict(title=title,authors=[a['author']['display_name'] for a in r.get('authorships',[])],
                              year=r.get('publication_year'),doi=(r.get('doi') or '').removeprefix('https://doi.org/') or None,
                              url=r['id'],decision='maybe',claims=[],methods=[],limitations=[],
                              notes='Metadata-only adjacent work; requires primary-text screening; not cited as evidence.'))
    questions=[
        'Does dedicated global querying improve rejection at matched positive recall and matched negative recall?',
        'Which gains remain under matched optimization budgets, Val selectors and multiple seeds?',
        'Does Gaussian pooling outperform uniform pooling in this same active configuration?',
        'How much full-Test discrimination reflects source association versus within-WorldCup event verification?',
        'Can positive/multi-target end-to-end recall improve without sacrificing rejection?',
        'How do this operation and its full design differ from previous Gaussian temporal weighting/refinement methods?',
        'What are measured adaptation cost, inference latency, memory and FLOPs?',
    ]
    card=dict(query='Dedicated global existence verification and separate temporal RoI refinement for Soccer-GMR',
              platform='ResearchPilot Tier 1 APIs + primary-source retrieval + independent citation review',
              searched_at=None,search_date='2026-10-23',
              papers=accepted,summary='Focused evidence pool; not a systematic review or novelty proof.',gaps=questions)
    dump('evidence_cards.json',card)
    dump('thesis_memory.json',dict(topic=card['query'],accepted_papers=accepted,maybe_papers=maybe,rejected_papers=rejected,
                                useful_keywords=['generalized moment retrieval','negative-aware video moment retrieval','global event query','Gaussian temporal RoI','low-rank adaptation'],
                                authors_to_watch=[],open_questions=questions))
    command=[sys.executable,str(ROOT/'ResearchPilot-Skills/scripts/export_bibtex.py'),str(HERE/'evidence_cards.json')]
    exported=subprocess.run(command,check=True,capture_output=True,text=True).stdout
    blocks=exported.strip().split('\n\n')
    for block,paper in zip(blocks,accepted):
        kind='misc' if paper['bibliographic_form']=='preprint' else 'inproceedings'
        blocks[blocks.index(block)]=block.replace('@article{','@'+kind+'{',1)
    (HERE/'references.bib').write_text('\n\n'.join(blocks)+'\n')
    claims=[
        ('C01','The local task includes no/single/multiple intervals.',['data/train.jsonl','data/val.jsonl','data/test.jsonl'],'verified'),
        ('C02','Presence reads text and intermediate video memories without direct proposal evidence.',['code/model.py','code/runtime.py','configs/s4_fusion.json'],'verified implementation; causal benefit not isolated'),
        ('C03','The active objective is BCE existence plus 0.5 positive-only refinement; no plurality objective.',['code/model.py','code/train.py','configs/s4_fusion.json'],'verified'),
        ('C04','Five implementations improve G1 and ungated mAP at fixed .4 relative to archived local GMR baselines.',['docs/research/generated/cross_backbone_results.csv'],'single-seed complete-system comparisons; S4 recomputed; baseline curve summaries'),
        ('C05','Positive recall falls for all five backbones; AUROC falls for QD-DETR and EaTR.',['docs/research/generated/cross_backbone_results.csv'],'verified'),
        ('C06','All SportsMoments examples are positive; exact match metadata overlaps splits.',['docs/research/generated/dataset_statistics.json'],'verified association; not proof of model shortcut'),
        ('C07','A3 .4 and .5 differences must be labeled separately.',['docs/research/generated/fixed_threshold_results.csv','provenance/QCD_A3.md'],'verified'),
        ('C08','Root FlashVTG exporter retains uniform existence-based proposal-score scaling.',['baseline_source/training/flash_vtg_gmr/inference.py'],'verified; removed fusion concerns candidate-to-presence path'),
        ('C09','Root raw mAP is after refinement and before export score scaling/NMS.',['code/model.py','baseline_source/training/flash_vtg_gmr/inference.py'],'verified; not a component ablation'),
        ('C10','The supplied image-segmentation paper motivates but does not prove the GMR mechanism.',['L02'],'supported limited cross-task inference'),
    ]
    dump('working/evidence_map.json',dict(local_claims=[dict(id=i,claim=c,sources=s,status=t) for i,c,s,t in claims],literature_evidence=[p['evidence_id'] for p in accepted],unresolved_evidence=questions))
    dump('working/chapter_blueprint.json',dict(scope='Repository data/algorithm documentation and partial manuscript sections',sections=[
        dict(file='DATA.md',role='Reproduce data handling and identify split limitations',claims=['C01','C06']),
        dict(file='ALGORITHM.md',role='Define active computation and loss',claims=['C02','C03','C08','C09']),
        dict(file='EXPERIMENTS.md',role='Report observations and protocol limits',claims=['C04','C05','C07']),
        dict(file='RESEARCH_GAP.md',role='Separate established setting, inspiration and hypotheses',claims=['C10']),
        dict(file='PAPER_SECTIONS.md',role='Supported abstract, methods, results and limitations prose',claims=['C01','C02','C03','C04','C05','C06','C07'])]))
    draft=(HERE.parent/'PAPER_SECTIONS.md')
    dump('draft_handoff.json',dict(platform='Local evidence-grounded drafting; Jenni not used',section='abstract, methods, experimental results, limitations',
        draft_file='../PAPER_SECTIONS.md',draft=draft.read_text() if draft.exists() else '',
        source_map=[dict(claim=c,sources=s,status='supported') for i,c,s,t in claims],gaps=questions,
        revision_notes=['Partial manuscript sections, not a full submission-ready paper.','No new experiments; do not claim missing ablations or significance.']))
    files=[ROOT/'README.md',ROOT/'code/model.py',ROOT/'code/runtime.py',ROOT/'code/train.py',ROOT/'baseline_source/training/flash_vtg_gmr/dataset.py',ROOT/'baseline_source/training/flash_vtg_gmr/inference.py']
    files+=list((ROOT/'configs').glob('*.json'))+list((ROOT/'data').glob('*.jsonl'))
    files+=list((ROOT/'results').glob('*/test_metrics.json'))
    phase=ROOT/'Phase2(addbackbone_with_S4_minusFusion)'
    files+=[phase/'EPOCH4_TEST_RESULTS.csv',phase/'shared/train.py',phase/'shared/config.json',ROOT/'code/qcd.py',ROOT/'provenance/QCD_A3.md',ROOT/'reports/subsets_test_tau05.csv']
    files+=list((phase/'shared/plot_baselines').glob('*_gmr.json'))
    files+=list(phase.glob('*/model.py'))+list(phase.glob('*/configuration/CONFIG.json'))
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    external=Path('/home/guoxiangyu/10554-AAAI26.KimN-CV.pdf')
    dump('source_manifest.json',dict(repository_head=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,check=True,text=True,capture_output=True).stdout.strip(),
                                  local_source_sha256=manifest,external_pdf={'path':str(external),'sha256':hashlib.sha256(external.read_bytes()).hexdigest() if external.exists() else None},
                                  note='Some archives are untracked at audit time; hashes identify files independently of Git status.'))
    print(json.dumps(dict(accepted=len(accepted),maybe=len(maybe),rejected=len(rejected),claims=len(claims))))


if __name__=='__main__':main()
