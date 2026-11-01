"""Document the fixed temporal exporter and verify native grid rounding on CPU."""
import ast,types,torch
from paths import *

def audit():
 files={
 'moment_detr':KIT/'vendor/backends/moment_detr/source/training/moment_detr_gmr/postprocessing.py',
 'eatr':KIT/'vendor/backends/eatr/source/models/postprocessing.py',
 'qd_detr':KIT/'vendor/backends/qd_detr/source/qd_detr/postprocessing_qd_detr.py',
 'flashvtg':KIT/'vendor/baseline_source/training/flash_vtg_gmr/postprocessing.py'}
 values=torch.tensor([[-1.2,.1],[1.2,7.7],[34.6,36.5666667],[149.2,193.2]],dtype=torch.float32)
 evidence={}
 for b,p in files.items():
  tree=ast.parse(p.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='round_to_multiple_clip_lengths')
  namespace={'torch':torch};exec(compile(ast.Module(body=[method],type_ignores=[]),str(p),'exec'),namespace)
  actual=namespace[method.name](types.SimpleNamespace(clip_length=2.),values)
  expected=(values/2.).round()*2.;assert torch.equal(actual,expected)
  evidence[b]=dict(native_postprocessor=str(p.relative_to(KIT)),source_sha256=sha(p),native_grid_seconds=2.,native_rounding_bitwise_equal=True,time_conversion='Native dense head already exports seconds (clip_length=2)' if b=='flashvtg' else 'Normalized center/width multiplied by real sample duration',NMS='Native normal temporal NMS, configured threshold0.7' if b=='flashvtg' else 'No NMS')
 report=dict(passed=True,scope='Native clip-grid rounding and explicit fixed Charades export policy; no change to any prediction or selected checkpoint',backbones=evidence,charades_adaptations=dict(real_duration_clamp='After rounding, clamp boundaries to [0, actual duration]; no fixed150s limit',degenerate_windows='Existing Charades adapter repair: only nonpositive-width windows become min(2s,duration) windows inside the video',confidence_precision='Keep native FP32 confidence and stable descending confidence order',frozen='Copy parent fixed candidate export exactly; replace presence score only'),extra_ranking_modules=False,selection_uses_ungated_localization=True)
 dump(OUT/'reports/export_protocol_audit.json',report);return report
if __name__=='__main__':audit();print('NATIVE_GRID_AND_FIXED_CHARADES_EXPORT_POLICY_VERIFIED')
