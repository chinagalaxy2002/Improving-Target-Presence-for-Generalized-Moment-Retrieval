import sys,subprocess,time
from paths import *
b=sys.argv[1]
for m in METHODS:
 report=OUT/'reports/debug24'/b/m/'preflight.json'
 if report.exists() and json.loads(report.read_text()).get('checks',{}).get('native_export_and_metrics'):continue
 result=subprocess.run([sys.executable,str(KIT/'debug_one.py'),b,m])
 if result.returncode:sys.exit(result.returncode)
print('BACKBONE_DEBUG_COMPLETE',b,flush=True)
