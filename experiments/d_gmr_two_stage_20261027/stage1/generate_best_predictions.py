import os, sys, json, torch
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/ablations/end_to_end_geq_pv_20261026")
from scratch_adapter import ScratchAdapter
from model import JointGMR
from evaluation import predict, evaluate

adapter = ScratchAdapter('flashvtg', 'cuda:0')
model = JointGMR(adapter.base, 'flashvtg', pv=False).to('cuda:0')

for ep in [63, 70]:
    out_dir = Path(f"/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/ablations/end_to_end_geq_pv_20261026/runs/flashvtg/E3_extra50/epoch_{ep:03d}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = f"/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/ablations/end_to_end_geq_pv_20261026/runs/flashvtg/E3_extra50/epoch_{ep:03d}.ckpt"
    print(f"Generating full test predictions for epoch {ep}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    pred, diag = predict(adapter, model, adapter.dataset("test"))
    gt = [row for row in adapter.dataset("test").data]
    evaluate(pred, gt, out_dir, "test")
    # Also dump test_predictions.jsonl
    with (out_dir / "test_predictions.jsonl").open("w") as f:
        for p in pred:
            f.write(json.dumps(p) + "\n")
    print(f"Saved to {out_dir / 'test_predictions.jsonl'}")

print("Done generating predictions for Epoch 63 and 70!")
