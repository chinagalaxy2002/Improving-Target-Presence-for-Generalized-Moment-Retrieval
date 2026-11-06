"""Model Runner for extracting all decoder layer outputs, scores, and candidate proposals.

Supports all 8 DETR backbones:
- moment_detr (2 decoder layers)
- eatr (3 decoder layers)
- qd_detr (2 decoder layers)
- cg_detr (3 decoder layers)
- sim_detr (4 decoder layers)
- bm_detr (3 decoder layers)
- bam_detr (2 decoder layers)
- uvcom (3 decoder layers)

Strictly preserves all native decoder queries without Top-10 truncation, NMS, or GEQ suppression.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from soccergmr.config import load_config
from soccergmr.backbones.registry import build, forward, get_spec
from soccergmr.data import SoccerDataset, collate, move
from .core import compute_tiou_matrix

CHECKPOINTS_M1 = {
    "moment_detr": "/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/moment_detr/gmr/best.ckpt",
    "eatr": "/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/eatr/gmr/best.ckpt",
    "qd_detr": "/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/qd_detr/gmr/best.ckpt",
    "cg_detr": "/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/cg_detr/gmr/best.pt",
    "sim_detr": "/home/guoxiangyu/VLMbasedIter_momentretrival/Sim-DETR/results_soccer_gmr_csc/native_nullaware_maskloss6_bsz8_seed2023/model_best.ckpt",
    "bm_detr": "/home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bm_detr/gmr/best.pt",
    "bam_detr": "/home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bam_detr/gmr/best.pt",
    "uvcom": "/home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/uvcom/gmr/best.pt",
}

CHECKPOINTS_M2 = {
    bb: f"/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/{bb}/gmr/best.pt"
    for bb in CHECKPOINTS_M1
}

CHECKPOINTS_M3 = {
    bb: f"/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/{bb}/geq_full/best.pt"
    for bb in CHECKPOINTS_M1
}


def detach_dq_cgp(model: torch.nn.Module) -> None:
    """Detaches DQ-CGP adapter from any backbone to restore pure native GMR Baseline."""
    for parent in [
        model,
        getattr(model, "native", None),
        getattr(getattr(model, "semantic_model", None), "base_model", None),
    ]:
        if parent is None:
            continue
        if hasattr(parent, "query_cgp"):
            del parent.query_cgp
            parent.query_cgp = None
        decoder = None
        if hasattr(parent, "transformer") and hasattr(parent.transformer, "decoder"):
            decoder = parent.transformer.decoder
        elif hasattr(parent, "CIM") and hasattr(parent.CIM, "decoder"):
            decoder = parent.CIM.decoder
        if decoder is not None and hasattr(decoder, "interlayer_adapter"):
            del decoder.interlayer_adapter
            decoder.interlayer_adapter = None


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1.0 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def load_model(backbone_name: str, method: str = "m1", device: str = "cuda:0") -> tuple[torch.nn.Module, dict]:
    """Loads a backbone model under the specified method configuration (m1, m2, or m3)."""
    cfg_path = Path("configs/experiments") / f"{backbone_name}.json"
    cfg = load_config(str(cfg_path))

    if method == "m1":
        cfg["architecture"]["use_query_cgp"] = False
        ckpt_path = CHECKPOINTS_M1[backbone_name]
    elif method == "m2":
        ckpt_path = CHECKPOINTS_M2[backbone_name]
    elif method == "m3":
        ckpt_path = CHECKPOINTS_M3[backbone_name]
    else:
        raise ValueError(f"Unknown method {method}")

    model, _ = build(cfg, device=device)

    if method == "m1":
        detach_dq_cgp(model)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else (ckpt["state_dict"] if "state_dict" in ckpt else ckpt)

    if method == "m3":
        from soccergmr.geq import GEQFull
        model = GEQFull(model, get_spec(backbone_name)).to(device)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if method == "m1" and (len(missing) > 0 or len(unexpected) > 0):
        # Strict verification for baseline
        print(f"[{backbone_name} m1] Warning: missing={len(missing)}, unexpected={len(unexpected)}")

    model.eval()
    return model, cfg


@torch.no_grad()
def run_inference_and_extract(
    model: torch.nn.Module,
    cfg: dict,
    backbone_name: str,
    split_name: str,
    device: str = "cuda:0",
    batch_size: int = 16,
    num_workers: int = 2
) -> List[Dict[str, Any]]:
    """Runs inference over the entire dataset split and extracts all layer predictions."""
    dataset = SoccerDataset(cfg, split_name, get_spec(backbone_name))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate
    )

    records = []

    for metas, inputs, _ in loader:
        b_size = len(metas)
        inputs_dev = move(inputs, device)

        if backbone_name == "bm_detr":
            # Direct multi-layer extraction for BM-DETR
            native = model.native if hasattr(model, "native") else model
            src_vid = native.input_vid_proj(inputs_dev["src_vid"])
            pos_vid = native.position_embed(src_vid, inputs_dev["src_vid_mask"])
            src_txt = native.input_txt_proj(inputs_dev["src_txt"])
            pos_txt = native.txt_position_embed(src_txt) if native.use_txt_pos else torch.zeros_like(src_txt)

            if hasattr(native.transformer.decoder, "interlayer_adapter") and getattr(native.transformer.decoder, "interlayer_adapter") is not None:
                if hasattr(model, "query_cgp") and model.query_cgp is not None:
                    model.query_cgp.clear_diagnostics()
                semantic_mask = inputs_dev["src_txt_mask"].bool()
                semantic_count = semantic_mask.sum(dim=1, keepdim=True).clamp_min(1)
                query_semantic = (src_txt * semantic_mask.unsqueeze(-1).to(src_txt.dtype)).sum(dim=1) / semantic_count.to(src_txt.dtype)
                native.transformer.decoder.interlayer_adapter_kwargs = {
                    "query_semantic": query_semantic,
                    "video_length": src_vid.shape[1],
                }

            hs, references, memory, frame_probs = native.calculate_probs(
                src_vid=src_vid,
                src_vid_mask=inputs_dev["src_vid_mask"],
                pos_vid=pos_vid,
                src_txt=src_txt,
                src_txt_mask=inputs_dev["src_txt_mask"],
                pos_txt=pos_txt,
                returns=True
            )
            # hs: (#layers, batch_size, #queries, hidden_dim)
            # references: (#layers, batch_size, #queries, 2)
            outputs_class = native.class_embed(hs)  # (#layers, bsz, K, 2)
            outputs_coord = (native.span_embed(hs) + inverse_sigmoid(references)).sigmoid()  # (#layers, bsz, K, 2)
            num_layers = outputs_class.shape[0]

            layer_dicts_batch = []
            for l in range(num_layers):
                layer_dicts_batch.append({
                    "pred_logits": outputs_class[l],
                    "pred_spans": outputs_coord[l]
                })

            pred_exist_score = None
            if hasattr(model, "exist_head"):
                logit = model.exist_head(hs[-1].amax(dim=1)).squeeze(-1)
                pred_exist_score = logit.sigmoid().cpu().numpy()

        else:
            outputs = forward(model, backbone_name, inputs_dev, metas=metas)
            aux_outputs = outputs.get("aux_outputs", []) or []
            # Layers 0 to L-2 are aux_outputs, final layer is in outputs
            num_layers = len(aux_outputs) + 1
            layer_dicts_batch = list(aux_outputs) + [outputs]

            pred_exist_score = None
            exist_tensor = outputs.get("pred_exist_logits", outputs.get("pred_exist_logit"))
            if exist_tensor is not None:
                pred_exist_score = exist_tensor.reshape(-1).sigmoid().cpu().numpy()

        # Extract per-sample data
        for b in range(b_size):
            meta = metas[b]
            qid = int(meta["qid"])
            vid = str(meta["vid"])
            duration = float(meta["duration"])
            gt_windows = meta.get("relevant_windows", [])
            gt_arr = np.array(gt_windows, dtype=np.float32).reshape(-1, 2)
            num_gt = len(gt_windows)

            layers_data = []

            for l_idx, l_dict in enumerate(layer_dicts_batch):
                spans = l_dict["pred_spans"][b]  # (K, 2): center, width in [0, 1]
                logits = l_dict.get("pred_logits")
                fg_prob = logits[b].softmax(-1)[:, 0] if logits is not None else torch.zeros(spans.shape[0], device=device)

                native_score = fg_prob.clone()
                if "pred_quality" in l_dict:
                    native_score = l_dict["pred_quality"][b].squeeze(-1)
                if "iou_scores" in l_dict:
                    native_score = native_score * l_dict["iou_scores"][b].squeeze(-1).sigmoid()

                # Convert spans to unnormalized seconds [start, end]
                center = spans[:, 0]
                width = spans[:, 1]
                start = (center - width / 2.0).clamp(0.0, 1.0) * duration
                end = (center + width / 2.0).clamp(0.0, 1.0) * duration
                pred_windows = torch.stack([start, end], dim=-1).cpu().numpy()  # (K, 2)

                # Pairwise IoU matrix (K, M)
                ious = compute_tiou_matrix(pred_windows, gt_arr) if num_gt > 0 else np.zeros((len(pred_windows), 0), dtype=np.float32)

                layers_data.append({
                    "layer_idx": l_idx,
                    "pred_windows": pred_windows.tolist(),
                    "native_scores": native_score.cpu().numpy().tolist(),
                    "foreground_probs": fg_prob.cpu().numpy().tolist(),
                    "ious": ious.tolist()
                })

            records.append({
                "qid": qid,
                "vid": vid,
                "duration": duration,
                "gt_windows": gt_windows,
                "num_gt": num_gt,
                "num_layers": num_layers,
                "layers": layers_data,
                "pred_exist_score": float(pred_exist_score[b]) if pred_exist_score is not None else None
            })

    return records
