"""Write ignored local external-asset paths; no external files are modified."""
from pathlib import Path
import argparse,json
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--assets-dir',required=True);p.add_argument('--baseline-checkpoint',required=True);p.add_argument('--s4-fusion-checkpoint',required=True);a=p.parse_args()
obj={k:str(Path(v).expanduser().resolve()) for k,v in vars(a).items()}
assert Path(obj['assets_dir']).is_dir()
assert all(Path(obj[k]).is_file() for k in ['baseline_checkpoint','s4_fusion_checkpoint'])
(ROOT/'local_paths.json').write_text(json.dumps(obj,indent=2)+'\n');print('Wrote ignored local_paths.json; external assets unchanged')
