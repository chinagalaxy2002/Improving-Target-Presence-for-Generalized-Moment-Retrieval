"""Independent bounded Git publishing retries; never blocks the CUDA dispatcher."""
import argparse,fcntl,hashlib,io,json,os,re,signal,subprocess,tempfile,time,urllib.request,importlib
from PIL import Image
from pathlib import Path
from paths import *
import prepare_release
from prepare_release import WORKTREE
URL='https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval.git'
BRANCH='gmr-mr-detr-multiroute-20261030'
REPO_RAW='https://raw.githubusercontent.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/'
OWNED=['shared/charades_multi_geq','results/charades_multi_geq','README.md']+[f'backbones/{b}/charades_multi_geq.sh' for b in BACKBONES]+['backbones/flashvtg/README.md']

def command(args,env=None,check=True):
 process=subprocess.Popen(['git','-c','credential.helper=','-C',str(WORKTREE),*args],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
 try:stdout,stderr=process.communicate(timeout=120 if args[0]=='push' else 60)
 except subprocess.TimeoutExpired:
  os.killpg(process.pid,signal.SIGKILL);process.communicate()
  raise RuntimeError(f'git {args[0]} exceeded its bounded network timeout; child processes terminated')
 result=subprocess.CompletedProcess(process.args,process.returncode,stdout,stderr)
 if check and result.returncode:raise RuntimeError(f'git {args[0]} exited {result.returncode}: {result.stderr[-1600:]}')
 return result

def publish_once(push=True):
 subprocess.run([__import__('sys').executable,str(KIT/'plot_report.py')],check=True,timeout=180)
 subprocess.run([__import__('sys').executable,str(KIT/'completion_audit.py'),'--checkpoints','--recompute'],check=True,timeout=600,stdout=subprocess.DEVNULL)
 manifest=importlib.reload(prepare_release).prepare(WORKTREE)
 return publish_prepared(manifest,push)

def publish_prepared(manifest,push=True):
 """Publish the exact prepared manifest; permits bounded Git-only retries."""
 for path,digest in manifest.items():assert sha(WORKTREE/path)==digest,'Prepared release changed before publication: '+path
 staged=command(['diff','--cached','--name-only']).stdout.splitlines()
 assert all(any(p==prefix or p.startswith(prefix+'/') for prefix in OWNED) for p in staged),'Unrelated staged files: refusing to commit'
 if not push:return dict(state='PREPARED_ONLY',files=len(manifest))
 # Credentials are read only by Git's askpass child. They never enter a URL,
 # command argument, output log, tracked file, or Python parent variable.
 with tempfile.TemporaryDirectory(prefix='charades-git-auth-') as temporary:
  ask=Path(temporary)/'askpass.py'
  ask.write_text('''#!/usr/bin/env python3
import os,sys
from pathlib import Path
if 'Username' in sys.argv[1]:
 print('x-access-token')
else:
 value=Path(os.environ['CHARADES_GIT_AUTH_FILE']).read_text().strip()
 if '\\n' in value:raise SystemExit('Expected a single Git credential value')
 print(value)
''');ask.chmod(0o700)
  env=dict(os.environ,GIT_ASKPASS=str(ask),GIT_TERMINAL_PROMPT='0',CHARADES_GIT_AUTH_FILE=os.environ.get('CHARADES_GIT_AUTH_FILE','/home/guoxiangyu/token2git.txt'))
  remote=command(['ls-remote',URL,f'refs/heads/{BRANCH}'],env).stdout.split()[0]
  if command(['cat-file','-e',remote+'^{commit}'],check=False).returncode:
   command(['fetch','--no-tags',URL,BRANCH],env)
   assert command(['rev-parse','FETCH_HEAD']).stdout.strip()==remote
  command(['add','--',*OWNED])
  changed=command(['diff','--cached','--quiet'],check=False).returncode
  if changed:
   command(['commit','-m','Add Charades Multi GEQ seed2023 implementation and verified progress'],env)
  head=command(['rev-parse','HEAD']).stdout.strip()
  if command(['merge-base','--is-ancestor',remote,head],check=False).returncode:
   rebase=command(['rebase',remote],env,check=False)
   if rebase.returncode:
    command(['rebase','--abort'],env,check=False)
    raise RuntimeError('Remote branch advanced; automatic rebase conflicted. Candidate commit preserved for manual review; no force-push attempted.')
   head=command(['rev-parse','HEAD']).stdout.strip()
  # A rebase must not silently change any prepared Charades artifact.
  for path,digest in manifest.items():
   assert sha(WORKTREE/path)==digest,f'Prepared release changed during rebase: {path}'
  command(['push',URL,f'HEAD:refs/heads/{BRANCH}'],env)
  observed=command(['ls-remote',URL,f'refs/heads/{BRANCH}'],env).stdout.split()[0]
  assert observed==head,'Remote commit does not match pushed HEAD'
 # Git commits bind all reachable image blobs by content hash. This also
 # verifies private-repository images without using credentials outside Git.
 images=[]
 readmes=[WORKTREE/'README.md']+list((WORKTREE/'results/charades_multi_geq').rglob('README.md'))
 for readme in readmes:
  for target in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',readme.read_text()):
   local=(readme.parent/target).resolve()
   if 'charades_multi_geq' not in str(local):continue
   relative=local.relative_to(WORKTREE).as_posix();assert local.is_file()
   blob=subprocess.run(['git','-C',str(WORKTREE),'show',head+':'+relative],capture_output=True,check=True,timeout=20).stdout
   assert hashlib.sha256(blob).hexdigest()==sha(local),'Committed image differs from prepared image'
   if local.suffix=='.png':
    with Image.open(io.BytesIO(blob)) as picture:picture.verify()
   images.append(relative)
 status=dict(state='CODE_AND_AVAILABLE_RESULTS_PUBLISHED',time=time.time(),commit=head,branch=BRANCH,remote_commit=observed,verified_readme_images=sorted(set(images)),image_verification='Matching authenticated remote commit; reachable Git image blob hashes and PNG decoding. Private repository access is required.',release_manifest_sha256=sha(OUT/'reports/release_manifest.json'))
 dump(OUT/'logs/publication_status.json',status)
 audit=json.loads((OUT/'reports/completion_audit.json').read_text())
 if not audit['errors'] and audit['training_verified']==24 and audit['tests_verified']==48 and set(audit['pending'])<={'verified_final_GitHub_publication'}:
  status['state']='PUBLISHED';dump(OUT/'logs/PUBLISHED.json',status)
  dump(OUT/'logs/publication_status.json',status)
 return status

def main():
 p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');p.add_argument('--prepare-only',action='store_true');p.add_argument('--interval',type=int,default=1800);a=p.parse_args();(OUT/'logs').mkdir(parents=True,exist_ok=True)
 lock=(OUT/'logs/publisher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  try:
   status=publish_once(not a.prepare_only);print(json.dumps(status),flush=True)
   if a.once or a.prepare_only or status['state']=='PUBLISHED':return
  except Exception as e:
   failure=dict(time=time.time(),error=f'{type(e).__name__}: {e}');dump(OUT/'logs/publication_failure.json',failure);print(json.dumps(failure),flush=True)
   if a.once:raise
  time.sleep(a.interval)
if __name__=='__main__':main()
