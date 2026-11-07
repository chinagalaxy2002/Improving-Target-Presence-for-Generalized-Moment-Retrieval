#!/usr/bin/env python3
"""Gate-0 Real Batch & VRAM Peak Measurement on GPU.

Executes a full forward -> loss -> backward -> optimizer.step on real SoccerGMR batch,
verifies strict save/load, and measures peak CUDA memory for concurrency budgeting.
"""
import argparse
import os
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soccergmr.config import load_config
from soccergmr.backbones.registry import build, forward, get_spec
from soccergmr.data import SoccerDataset, collate, move
from soccergmr.checkpoint import save as ck_save, load as ck_load
from torch.utils.data import DataLoader

def run_real_batch_test(config_path: Path, device: str = "cuda:0"):
    print(f"\n==========================================")
    print(f"Testing real batch for: {config_path.name}")
    print(f"==========================================")
    
    cfg = load_config(config_path)
    backbone = cfg["backbone"]
    spec = get_spec(backbone)
    
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    print(f"[{backbone}] Building model on {device}...")
    model, criterion = build(cfg, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    
    train_dataset = SoccerDataset(cfg, "train", spec)
    loader = DataLoader(train_dataset, batch_size=cfg["protocol"]["batch_size"], shuffle=False, collate_fn=collate)
    
    # 1. Real batch forward + backward + step
    print(f"[{backbone}] Loading 1 batch of real training data...")
    for metas, inputs, targets in loader:
        inputs_dev = move(inputs, device)
        targets_dev = move(targets, device)
        
        model.train()
        optimizer.zero_grad()
        
        print(f"[{backbone}] Forward pass...")
        outputs = forward(model, backbone, inputs_dev, targets_dev, metas=metas, epoch=0)
        
        print(f"[{backbone}] Loss calculation...")
        loss, loss_dict = criterion(outputs.get("native_output", outputs), targets_dev, epoch=0)
        assert torch.isfinite(loss), f"Non-finite loss: {loss}"
        print(f"[{backbone}] Loss value: {loss.item():.4f}")
        for k, v in loss_dict.items():
            val = v.item() if torch.is_tensor(v) else v
            print(f"    {k}: {val:.4f}")
            
        print(f"[{backbone}] Backward pass...")
        loss.backward()
        
        # Verify gradients exist and are finite
        has_grad = False
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"
                if p.grad.abs().sum() > 0:
                    has_grad = True
        assert has_grad, "No parameters received non-zero gradient!"
        
        print(f"[{backbone}] Optimizer step...")
        optimizer.step()
        break
        
    peak_vram_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    peak_vram_gb = peak_vram_bytes / (1024 ** 3)
    print(f"[{backbone}] Peak VRAM during Train step: {peak_vram_gb:.2f} GB")
    
    # 2. Real batch evaluation
    print(f"[{backbone}] Running 1 batch of real validation...")
    val_dataset = SoccerDataset(cfg, "val", spec)
    val_loader = DataLoader(val_dataset, batch_size=cfg["protocol"]["eval_batch_size"], shuffle=False, collate_fn=collate)
    model.eval()
    with torch.no_grad():
        for metas, inputs, targets in val_loader:
            inputs_dev = move(inputs, device)
            outputs = forward(model, backbone, inputs_dev, metas=metas)
            assert "pred_spans" in outputs
            assert "pred_logits" in outputs
            break
            
    val_peak_vram_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    val_peak_vram_gb = val_peak_vram_bytes / (1024 ** 3)
    print(f"[{backbone}] Total Peak VRAM (Train + Val): {val_peak_vram_gb:.2f} GB")
    
    # 3. Checkpoint save & strict load test
    test_ckpt_path = ROOT / "tmp_test_ckpt.pt"
    try:
        payload = {
            "format": 1,
            "stage": "gmr",
            "epoch": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(payload, test_ckpt_path)
        
        # Fresh model to test strict reload
        print(f"[{backbone}] Testing strict reload into fresh model...")
        fresh_model, _ = build(cfg, device=device)
        loaded = torch.load(test_ckpt_path, map_location=device)
        fresh_model.load_state_dict(loaded["model"], strict=True)
        print(f"[{backbone}] Strict state_dict reload PASS!")
    finally:
        if test_ckpt_path.exists():
            test_ckpt_path.unlink()
            
    print(f"==========================================")
    print(f"[{backbone}] GATE-0 REAL BATCH TEST PASSED!")
    print(f"==========================================\n")
    return peak_vram_gb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    run_real_batch_test(args.config, args.device)
