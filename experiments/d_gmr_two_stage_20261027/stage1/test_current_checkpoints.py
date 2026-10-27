import os, sys, json, torch
from pathlib import Path

# Run on GPU 1 (free memory available)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.insert(0, "/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/ablations/end_to_end_geq_pv_20261026")
from scratch_adapter import ScratchAdapter
from model import JointGMR
from evaluation import predict, evaluate

def eval_checkpoint(ckpt_path, name, pv=False):
    print(f"\n================ Evaluating {name}: {ckpt_path} ================")
    adapter = ScratchAdapter('flashvtg', 'cuda:0')
    model = JointGMR(adapter.base, 'flashvtg', pv=pv).to('cuda:0')

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)

    model.eval()
    test_data = adapter.dataset("test")
    pred, diag = predict(adapter, model, test_data)

    gt = [row for row in adapter.dataset("test").data]
    out_dir = Path("/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/ablations/end_to_end_geq_pv_20261026/temp_eval") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = evaluate(pred, gt, out_dir, "test")
    r04 = reports["0.4"]["brief"]

    summary = {
        "name": name,
        "checkpoint": os.path.basename(ckpt_path),
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
    print(f"Summary for {name}:\n{json.dumps(summary, indent=2)}")
    return summary

if __name__ == "__main__":
    base = Path("/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/ablations/end_to_end_geq_pv_20261026/runs/flashvtg")
    e3_ckpt = sorted((base / "E3_extra50").glob("epoch_*.ckpt"))[-1]
    e4_ckpt = sorted((base / "E4_scratch_flashmmr").glob("epoch_*.ckpt"))[-1]

    print(f"Found latest E3 extra50 ckpt: {e3_ckpt}")
    print(f"Found latest E4 ckpt: {e4_ckpt}")

    res_e3 = eval_checkpoint(str(e3_ckpt), f"E3_extra50_{e3_ckpt.stem}", pv=False)
    res_e4 = eval_checkpoint(str(e4_ckpt), f"E4_{e4_ckpt.stem}", pv=True)

    out_file = Path("/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/ablations/end_to_end_geq_pv_20261026/temp_eval_summary.json")
    with open(out_file, "w") as f:
        json.dump({"E3_extra50": res_e3, "E4": res_e4}, f, indent=2)
    print(f"\nAll done! Output written to {out_file}")
