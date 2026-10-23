"""Collect public scholarly metadata; never uploads project text or datasets.

Run from any directory. Uses the bundled ResearchPilot normalizer/validator.
Raw API responses are retained, and failed calls are recorded explicitly.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'retrieval'


def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'DeRoI-literature-audit/1.0'})
    with urllib.request.urlopen(req, timeout=20) as response:
        raw = response.read().decode('utf-8')
    return json.loads(raw)


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def main():
    titles = [
        'Improving Target Presence and Plurality Recognition for Generalized Referring Image Segmentation',
        'Retrieving Any Relevant Moments: Benchmark and Models for Generalized Moment Retrieval',
        'QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries',
        'Query-Dependent Video Representation for Moment Retrieval and Highlight Detection',
        'Correlation-Guided Query-Dependency Calibration for Video Temporal Grounding',
        'Knowing Where to Focus: Event-aware Transformer for Video Grounding',
        'FlashVTG: Feature Layering and Adaptive Score Handling Network for Video Temporal Grounding',
        'LoRA: Low-Rank Adaptation of Large Language Models',
        'CVA: Context-aware Video-text Alignment for Video Temporal Grounding',
        'Generalized Video Moment Retrieval',
        'Moment of Untruth: Dealing with Negative Queries in Video Moment Retrieval',
        'Not All Inputs Are Valid: Towards Open-Set Video Moment Retrieval Using Language',
        'When One Moment Is Not Enough: Multi-Moment Retrieval with Cross-Moment Interactions',
    ]
    def lookup(pair):
        index, title = pair
        record = {'title_query': title, 'retrieved_at': datetime.now(timezone.utc).isoformat()}
        url = 'https://api.crossref.org/works?' + urllib.parse.urlencode({'query.title': title, 'rows': 3})
        record['api_url'] = url
        try:
            record['response'] = fetch(url)
            record['status'] = 'completed'
        except Exception as error:
            record.update(status='failed', error=str(error))
        save(f'crossref_title_{index:02d}.json', record)
        return {'title': title, 'status': record['status']}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(lookup, enumerate(titles, 1)):
            print(json.dumps(result), flush=True)

    # Narrow title filters remove the broad search's type:article restriction.
    for index, term in enumerate(['generalized moment retrieval', 'target presence plurality', 'video temporal grounding'], 1):
        params = {'filter': 'title.search:' + term, 'per-page': 8}
        url = 'https://api.openalex.org/works?' + urllib.parse.urlencode(params)
        record = {'query': term, 'api_url': url, 'filter_change': 'title.search; no type filter'}
        try:
            record.update(status='completed', response=fetch(url))
        except Exception as error:
            record.update(status='failed', error=str(error))
        save(f'openalex_focused_{index}.json', record)

    # Keep the actual graph call and its errors, rather than treating an empty
    # response from a helper that suppresses exceptions as successful expansion.
    seed_url = 'https://api.openalex.org/works/https://doi.org/10.1609/aaai.v40i7.37490'
    graph = {'seed_url': seed_url, 'seed_title': titles[0]}
    try:
        seed = fetch(seed_url)
        graph['seed'] = seed
        refs = seed.get('referenced_works', [])[:12]
        graph['backward_ids'] = refs
        if refs:
            ids = '|'.join(ref.rsplit('/', 1)[-1] for ref in refs)
            graph['backward_url'] = 'https://api.openalex.org/works?' + urllib.parse.urlencode({'filter': 'openalex_id:' + ids, 'per-page': 12})
            graph['backward'] = fetch(graph['backward_url'])
        graph['forward_url'] = 'https://api.openalex.org/works?' + urllib.parse.urlencode({'filter': 'cites:' + seed['id'].rsplit('/', 1)[-1], 'per-page': 8})
        graph['forward'] = fetch(graph['forward_url'])
        graph['status'] = 'completed'
    except Exception as error:
        graph.update(status='failed', error=str(error))
    save('openalex_expansion.json', graph)
    print('Public metadata collection finished.', flush=True)


if __name__ == '__main__':
    main()
