"""Run all six real-feature smoke checks sequentially on the visible GPU."""
import pathlib
import subprocess
import sys
entry = pathlib.Path(__file__).with_name('run_train.py')
for variant in ('baseline','shared','sepffn','instruction','multi','adaptive'):
    subprocess.run([sys.executable,str(entry),'--variant',variant,'--smoke'],check=True)
