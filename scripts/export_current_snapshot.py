"""Portable result snapshot; excludes model weights, features and credentials."""
import csv,gzip,json,shutil,subprocess
from pathlib import Path
from soccergmr.io import dump_json,sha256,read_jsonl
from soccergmr import checkpoint as ck
R=Path(__file__).resolve().parents[1];OLD=Path('/home/guoxiangyu/SoccerGMR-GEQ-clip-v2');RUN=R/'runs/clip_joint';OUT=R/'results/clip_joint_snapshot';OUT.mkdir(parents=True,exist_ok=True)
if (OLD/'results/clip_v2').exists():shutil.copytree(OLD/'results/clip_v2',R/'results/clip_v2',dirs_exist_ok=True)
rows=[];artifacts={}
for i in range(1,7):
 n=f'E{i}';source=RUN/n/'geq_full';target=OUT/n/'geq_full';target.mkdir(parents=True,exist_ok=True)
 m=json.loads((source/'complete.json').read_text());assert m['best_sha256']==sha256(source/'best.pt');last=ck.load(source/'last.pt');assert last['epoch']==m['completed_epoch']
 for file in ['complete.json','history.jsonl','best_val_metrics.json']:
  shutil.copy2(source/file,target/file)
 with gzip.GzipFile(filename=str(target/'best_val_predictions.jsonl.gz'),mode='wb',mtime=0) as f:f.write((source/'best_val_predictions.jsonl').read_bytes())
 migration=RUN/n/'gmr/MIGRATION.json';shutil.copy2(migration,target/'PARENT_MIGRATION.json')
 v=json.loads((source/'best_val_metrics.json').read_text());rows.append(dict(experiment=n,status='training_complete',best_epoch=m['best_epoch'],completed_epoch=m['completed_epoch'],Val_mAP=v['brief']['mAP'],Val_G1=m['best_score'],checkpoint_sha256=m['best_sha256'],checkpoint_source=m['source'],best_Test_not_used_for_selection=True))
 diag=RUN/n/'diagnostics'
 if diag.exists():shutil.copytree(diag,OUT/n/'diagnostics',dirs_exist_ok=True)
 for split in ['test_standard','test_full']:
  part=RUN/n/'geq_full_test'/split
  if not (part/'complete.json').exists():continue
  marker=json.loads((part/'complete.json').read_text());assert all(sha256(part/f)==marker[k] for f,k in [('predictions.jsonl','predictions_sha256'),('metrics.json','metrics_sha256')])
  dest=OUT/n/'geq_full_test'/split;dest.mkdir(parents=True,exist_ok=True)
  with gzip.GzipFile(filename=str(dest/'predictions.jsonl.gz'),mode='wb',mtime=0) as f:f.write((part/'predictions.jsonl').read_bytes())
  for file in ['metrics.json','complete.json']:shutil.copy2(part/file,dest/file)
for file in ['FREEZE.json','VAL_FROZEN.json','PAUSE_SNAPSHOT.json','queue_state.json']:
 if (RUN/file).exists():shutil.copy2(RUN/file,OUT/file)
shutil.copy2(RUN/'acceptance/complete.json',OUT/'ACCEPTANCE.json')
with (OUT/'VAL_SUMMARY.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Snapshot the requested common chart after it reflects paused/finalized state.
for ext in ['png','svg','pdf']:
 p=OLD/f'runs/clip_v2/monitor/training_val_panel.{ext}'
 if p.exists():shutil.copy2(p,OUT/p.name)
for p in sorted(OUT.rglob('*')):
 if p.is_file() and p.name!='ARTIFACTS.json':artifacts[str(p.relative_to(OUT))]=sha256(p)
dump_json(OUT/'ARTIFACTS.json',dict(passed=True,files=artifacts,training_paused=True,training_source='d782d85eca792df1d54b9086b55f9f10a1ded581',weights_uploaded=False))
lines=['# 当前结果快照：训练已暂停','','六组 GEQ＋aux 的训练均已完成并按 Val 选定 best；用户要求暂停时，仅自动评测队列仍在运行。已停止队列，不再派发训练或评测。E3 的专门对比评测由用户另行授权执行。所有 best.pt/last.pt 均保存在本机，未上传模型权重或特征。','','| 方法 | 最优轮 | 完成轮 | Val mAP | Val G1 三点均值 |','|---|---:|---:|---:|---:|']
for x in rows:lines.append(f"| {x['experiment']} | {x['best_epoch']} | {x['completed_epoch']} | {x['Val_mAP']:.2f} | {x['Val_G1']:.2f} |")
lines+=['','![训练进度快照](training_val_panel.png)','','黑色：GMR；蓝色：上一轮无辅助 GEQ；橙色：本轮 GEQ＋aux。所有训练已结束，不再自动刷新。','', '此目录保留已完成的 Test 缓存，未评测的项目不填造数值。E3 的完整对比见 [专门报告](../E3_baseline_comparison/README.md)。上一轮 E1–E6 的完整24组固定 Test 与诊断见 [上一轮报告](../clip_v2/REPORT.md)。','', '每组父 GMR checkpoint 的源路径、原始哈希、训练源码身份及兼容迁移记录保存在 PARENT_MIGRATION.json；GMR 权重逐张量保持一致，未重训已完成实验。实际训练 commit 与当前仅新增报告的发布 commit 分开记录，原始冻结记录未修改。']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
# Portable histories for the previous phase, which was fully evaluated.
for i in range(1,7):
 for stage in ['gmr','geq_full']:
  src=OLD/f'runs/clip_v2/E{i}/{stage}';dest=R/f'results/clip_v2/histories/E{i}/{stage}';dest.mkdir(parents=True,exist_ok=True)
  for file in ['history.jsonl','best_val_metrics.json','complete.json']:shutil.copy2(src/file,dest/file)
for p in sorted(OUT.rglob('*')):
 if p.is_file() and p.name!='ARTIFACTS.json':artifacts[str(p.relative_to(OUT))]=sha256(p)
dump_json(OUT/'ARTIFACTS.json',dict(passed=True,files=artifacts,training_paused=True,training_source='d782d85eca792df1d54b9086b55f9f10a1ded581',weights_uploaded=False))
print('Snapshot exported; six completed joint finetunes; previous six-method results included.')
