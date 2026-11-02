"""Prospective, recorded validation cadence; patience counts training epochs."""
from paths import dump, rows, sha


def ensure_schedule(directory, stage_b, next_epoch):
 path=directory/'validation_schedule.json'
 if path.exists():
  import json
  schedule=json.loads(path.read_text())
  assert schedule['interval_epochs']==(1 if stage_b else 5)
  return schedule
 history=directory/'history.jsonl'
 prefix=rows(history) if history.exists() else []
 assert next_epoch==len(prefix)+1, 'Checkpoint and completed-epoch history disagree'
 last=directory/'last.ckpt'
 schedule=dict(version=1,authorized_by='user',instruction='按照你分阶段考虑方案。',
  interval_epochs=1 if stage_b else 5,patience_epochs=6 if stage_b else 20,
  effective_from_epoch=next_epoch,legacy_history_rows=len(prefix),
  legacy_history_sha256=sha(history) if prefix else None,
  resumed_checkpoint_sha256=sha(last) if last.exists() else None,
  selection='Val mean G-mIoU@1 at .4/.5/.6' if stage_b else 'Val positive mAP',
  terminal_validation=True,existing_best_retained=True)
 dump(path,schedule)
 return schedule


def validation_due(epoch, best_epoch, ceiling, schedule):
 return (epoch % schedule['interval_epochs']==0 or epoch==ceiling or
         best_epoch>0 and epoch-best_epoch>=schedule['patience_epochs'] and
         epoch>=schedule.get('defer_early_stop_until_epoch',0))


def advance_selection(epoch, score, above, best, best_epoch, best_above):
 improved=score is not None and score>best
 tie_better=score is not None and score==best and above and not best_above
 if improved or tie_better:
  best=score;best_epoch=epoch;best_above=above
 # A tie can select a different Stage B checkpoint but cannot reset patience.
 return best,best_epoch,best_above,improved,tie_better
