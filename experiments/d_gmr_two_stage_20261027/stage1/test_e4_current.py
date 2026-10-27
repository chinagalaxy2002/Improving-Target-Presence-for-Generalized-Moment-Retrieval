import os, sys, json, torch
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, "/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/ablations/end_to_end_geq_pv_20261026")
from scratch_adapter import ScratchAdapter
from model import JointGMR
from evaluation import predict, evaluate

base = Path("/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/ablations/end_to_end_geq_pv_20261026/runs/flashvtg")
e4_ckpt = sorted((base / "E4_scratch_flashmmr").glob("epoch_*.ckpt"))[-1]
name = f"E4_{e4_ckpt.stem}"
print(f"Evaluating {name} ({e4_ckpt})...")

adapter = ScratchAdapter('flashvtg', 'cuda:0')
model = JointGMR(adapter.base, 'flashvtg', pv=True).to('cuda:0')
ckpt = torch.load(str(e4_ckpt), map_location="cpu", weights_only=False)
state = ckpt["model"] if "model" in ckpt else ckpt
model.load_state_dict(state, strict=False)
model.eval()

pred, diag = predict(adapter, model, adapter.dataset("test"))
gt = [row for row in adapter.dataset("test").data]
out_dir = Path("/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/ablations/end_to_end_geq_pv_20261026/temp_eval") / name
out_dir.mkdir(parents=True, exist_ok=True)
reports = evaluate(pred, gt, out_dir, "test")
r04 = reports["0.4"]["brief"]

summary = {
    "name": name,
    "AUROC": round(float(r04["AUROC"]), 2),
    "AUPRC": round(float(r04["AUPRC"]), 2),
    "Rej-F1": round(float(r04["Rej-F1@0.4"]), 2),
    "Pos-Recall": round(float(r04["Pos-Recall"]), 2),
    "TNR": round(float(r04["TNR"]), 2),
    "mAP": round(float(r04["mAP"]), 2),
    "mR@1": round(float(r04.get("mR@1", 0.0)), 2),
    "mR@5": round(float(r04.get("mR@5", 0.0)), 2),
    "mR+@5": round(float(r04["mR+@5"]), 2),
    "G-mIoU@1": round(float(r04["G-mIoU@1"]), 2),
}
print(json.dumps(summary, indent=2))
with open("/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/ablations/end_to_end_geq_pv_20261026/temp_e4_latest.json", "w") as f:
    json.dump(summary, f, indent=2)
