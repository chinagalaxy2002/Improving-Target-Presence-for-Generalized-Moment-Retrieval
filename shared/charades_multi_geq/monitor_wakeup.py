"""Systemd timer monitor: snapshot authoritative processes, then wake this Codex thread."""
import argparse,subprocess,os,time,json
from datetime import datetime
from zoneinfo import ZoneInfo
from paths import *
THREAD=os.environ.get('CHARADES_CODEX_THREAD','')
UNIT='charades-multi-geq-monitor-20261101'
def snapshot():
 queue=json.loads((OUT/'logs/QUEUE.json').read_text()) if (OUT/'logs/QUEUE.json').exists() else {}
 pid=queue.get('pid');cmd=Path(f'/proc/{pid}/cmdline') if pid else None
 live=bool(cmd and cmd.exists() and str(KIT/'queue_manager.py').encode() in cmd.read_bytes())
 state=dict(time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),queue_process_live=live,queue=queue,runs={})
 for b in BACKBONES:
  for m in METHODS:
   d=OUT/'runs'/b/m;p=d/'status.json'
   state['runs'][f'{b}/{m}']=dict(status=json.loads(p.read_text()) if p.exists() else 'PENDING',trained=(d/'TRAINED.json').exists(),tests={s:(d/s/'COMPLETE.json').exists() for s in ('test_id','test_ood')},last_checkpoint_mtime=(d/'last.ckpt').stat().st_mtime if (d/'last.ckpt').exists() else None)
 gpu=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
 state['gpu_compute_processes']=gpu.stdout;dump(OUT/'reports/monitor/latest.json',state)
 dump(OUT/'reports/monitor'/f'{int(time.time())}.json',state)
 return state

def main():
 p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');a=p.parse_args();state=snapshot()
 if a.dry_run:print(json.dumps(dict(snapshot_written=True,queue_live=state['queue_process_live'])));return
 message=f'''按用户授权的两小时定时检查触发：恢复检查 Charades Multi＋GEQ seed2023 实验进度。先读取 {OUT}/HANDOFF.md 和 reports/monitor/latest.json，核验真实队列进程、GPU、24任务/48测试及失败日志。发现故障则修复并从合法last恢复，防止重复派发；不得按Test调参。若尚在正常训练，简短汇报并结束本轮，systemd定时器每约2小时再次唤醒。同一会话由用户要求只交给你实施，不要启动其他代理。全部完成后继续原始绘图、结果整理与GitHub发布验收；只有全部验收通过才完成目标并停用此定时器。'''
 result=subprocess.run(['/home/guoxiangyu/.local/bin/codex','queue','--thread',THREAD,'--message',message,'--resume'],capture_output=True,text=True,timeout=90,cwd=str(PROJECT))
 record=dict(time=time.time(),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,thread=THREAD)
 dump(OUT/'logs/last_wakeup.json',record)
 print(json.dumps(record),flush=True)
 if result.returncode:raise SystemExit(result.returncode)
if __name__=='__main__':main()
