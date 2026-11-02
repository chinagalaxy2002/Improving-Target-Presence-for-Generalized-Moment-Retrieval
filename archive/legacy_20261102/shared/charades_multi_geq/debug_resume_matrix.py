import sys,subprocess,time
from paths import *
b=sys.argv[1]
for m in METHODS:
 report=OUT/'reports/debug24'/b/m/'preflight.json'
 if report.exists() and json.loads(report.read_text()).get('real_process_exit_and_restart'):continue
 while not (OUT/'reports/debug24'/b/m/'preflight.json').exists():time.sleep(1)
 cmd=[sys.executable,str(KIT/'debug_resume_one.py'),b,m]
 r=subprocess.run(cmd)
 assert r.returncode==75,(b,m,r.returncode)
 subprocess.run(cmd,check=True)
print('ALL_SIX_INTERRUPTION_CHECKS_PASS',b,flush=True)
