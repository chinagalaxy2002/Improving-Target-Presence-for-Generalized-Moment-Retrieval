#!/usr/bin/env python3
"""Generate publication-quality figures for the Presence Salvage Study.
"""
import os, json, gzip
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'experiments/presence_salvage_study'))

from run_study import (
    BACKBONES, DISPLAY_NAMES, METHODS, GRID,
    load_gz, order_predictions, get_paths, eval_curves
)

# Colors matching project design system
COLORS = {
    'GMR baseline': '#888888',     # Muted Gray
    'GMR(aux)': '#8E44AD',         # Royal Purple
    'GEQ only': '#D35400',         # Strong Orange
    'GEQ+aux': '#2980B9'           # High-Contrast Blue
}

MARKERS = {
    'GMR baseline': 's',
    'GMR(aux)': 'd',
    'GEQ only': '^',
    'GEQ+aux': 'o'
}

COLLAPSE_5 = ['taskweave', 'tr_detr', 'moment_detr', 'sim_detr', 'cg_detr']

def main():
    ev_root = ROOT / 'results/e3_high_retention'
    ev = json.load(open(ev_root / 'EVIDENCE.json'))
    gt = {s: load_gz(ev_root / ev['ground_truth'][s]['path']) for s in ('val', 'standard', 'full')}
    
    data = {}
    for b in BACKBONES:
        data[b] = {}
        paths = get_paths(b)
        for m_idx, m_name in enumerate(METHODS):
            data[b][m_name] = {}
            for s in ('val', 'standard', 'full'):
                preds = load_gz(paths[s][m_idx])
                preds = order_predictions(preds, gt[s])
                g1, g3, tpr, fpr, scores, pos = eval_curves(preds, gt[s])
                data[b][m_name][s] = {
                    'g1': g1, 'g3': g3, 'tpr': tpr, 'fpr': fpr,
                    'scores': scores, 'pos': pos
                }

    out_dir = ROOT / 'experiments/presence_salvage_study'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0

    # =========================================================================
    # FIGURE 1: 4-Panel G-mIoU@1 vs. Threshold (Full 11 vs 5-Collapse Subset)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    
    panels = [
        (axes[0, 0], 'standard', BACKBONES, '(a) Standard Test (All 11 Backbones Mean)'),
        (axes[0, 1], 'full', BACKBONES, '(b) Full-Decodable Test (All 11 Backbones Mean)'),
        (axes[1, 0], 'standard', COLLAPSE_5, '(c) Standard Test (Severe Collapse 5-Backbone Subset)'),
        (axes[1, 1], 'full', COLLAPSE_5, '(d) Full-Decodable Test (Severe Collapse 5-Backbone Subset)')
    ]
    
    for ax, split, subset, title in panels:
        # Highlight collapse zone
        ax.axvspan(0.20, 0.50, color='#FADBD8', alpha=0.35, label='High False-Alarm Collapse Zone (τ=0.2~0.5)' if ax == axes[0, 0] else None)
        
        for m in METHODS:
            curve = np.mean([data[b][m][split]['g1'] for b in subset], axis=0)
            lw = 2.4 if 'GEQ' in m or 'aux' in m else 1.8
            ls = '-' if m != 'GMR baseline' else '--'
            ax.plot(GRID, curve, label=m, color=COLORS[m], linewidth=lw, linestyle=ls)
            
            # Add markers at key thresholds
            key_pts = [35, 40, 50]
            ax.scatter(GRID[key_pts], curve[key_pts], color=COLORS[m], marker=MARKERS[m], s=40, zorder=5)

        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Existence Decision Threshold (τ)', fontsize=11)
        ax.set_ylabel('G-mIoU@1 (%)', fontsize=11)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 60.0)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Annotate operational region
        ax.axvline(0.40, color='#7F8C8D', linestyle=':', alpha=0.7)
        ax.axvline(0.50, color='#7F8C8D', linestyle=':', alpha=0.7)

    # Common legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=5, fontsize=11, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    fig1_png = out_dir / 'FIGURE_1_G1_VS_THRESHOLD.png'
    fig1_pdf = out_dir / 'FIGURE_1_G1_VS_THRESHOLD.pdf'
    fig.savefig(fig1_png, dpi=300)
    fig.savefig(fig1_pdf)
    plt.close(fig)
    print(f'Generated {fig1_png} and {fig1_pdf}')

    # =========================================================================
    # FIGURE 2: Massive Salvage Bar Chart at Tau = 0.40 (The Critical Operating Point)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    
    sub_backbones = ['taskweave', 'tr_detr', 'moment_detr', 'sim_detr', 'cg_detr']
    x = np.arange(len(sub_backbones))
    width = 0.18
    
    for ax_idx, split in enumerate(['standard', 'full']):
        ax = axes[ax_idx]
        split_name = 'Standard Test' if split == 'standard' else 'Full-Decodable Test'
        
        for m_idx, m in enumerate(METHODS):
            vals = [data[b][m][split]['g1'][40] for b in sub_backbones]
            bars = ax.bar(x + m_idx * width - 1.5 * width, vals, width, label=m, color=COLORS[m], edgecolor='#2C3E50', linewidth=0.8)
            
            # Add text labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.8, f'{height:.1f}',
                        ha='center', va='bottom', fontsize=8, rotation=0, fontweight='bold' if 'GEQ' in m else 'normal')
        
        ax.set_title(f'Presence Salvage at Operational Threshold τ=0.40 ({split_name})', fontsize=12, fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_NAMES[b] for b in sub_backbones], fontsize=11, fontweight='bold')
        ax.set_ylabel('G-mIoU@1 (%)', fontsize=11)
        ax.set_ylim(0, 65)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        
        # Add annotation box showing average salvage gain
        b_mean = np.mean([data[b]['GMR baseline'][split]['g1'][40] for b in sub_backbones])
        aux_mean = np.mean([data[b]['GMR(aux)'][split]['g1'][40] for b in sub_backbones])
        geq_mean = np.mean([data[b]['GEQ only'][split]['g1'][40] for b in sub_backbones])
        both_mean = np.mean([data[b]['GEQ+aux'][split]['g1'][40] for b in sub_backbones])
        
        box_text = (
            f"5-Backbone Subset Mean (τ=0.40):\n"
            f"• Baseline : {b_mean:.1f}%\n"
            f"• Aux Gain : +{aux_mean - b_mean:.1f} pp\n"
            f"• GEQ Gain : +{geq_mean - b_mean:.1f} pp\n"
            f"• Both Gain: +{both_mean - b_mean:.1f} pp"
        )
        ax.text(0.03, 0.95, box_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#EAEDED', edgecolor='#BDC3C7', alpha=0.9))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=11, frameon=True)
    plt.tight_layout()
    
    fig2_png = out_dir / 'FIGURE_2_SALVAGE_BARS.png'
    fig2_pdf = out_dir / 'FIGURE_2_SALVAGE_BARS.pdf'
    fig.savefig(fig2_png, dpi=300, bbox_inches='tight')
    fig.savefig(fig2_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f'Generated {fig2_png} and {fig2_pdf}')

    # =========================================================================
    # FIGURE 3: Score Margin & False-Alarm Suppression (Why Baseline Collapses)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    # Left: Score Margin for all 11 backbones
    ax1 = axes[0]
    margins = {m: [] for m in METHODS}
    for b in BACKBONES:
        for m in METHODS:
            scores = data[b][m]['standard']['scores']
            pos = data[b][m]['standard']['pos']
            margins[m].append(scores[pos].mean() - scores[~pos].mean())
    
    x11 = np.arange(len(BACKBONES))
    for m_idx, m in enumerate(METHODS):
        ax1.plot(x11, margins[m], marker=MARKERS[m], label=m, color=COLORS[m], linewidth=2.0, markersize=7)
    
    ax1.set_title('(a) Positive vs Negative Score Margin (Standard Test)', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xticks(x11)
    ax1.set_xticklabels([DISPLAY_NAMES[b] for b in BACKBONES], rotation=40, ha='right', fontsize=9)
    ax1.set_ylabel('Score Margin (Pos Mean - Neg Mean)', fontsize=11)
    ax1.axhline(0.0, color='red', linestyle='--', alpha=0.6, label='Random Chance Boundary (Margin=0)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=9, loc='lower right')
    
    # Right: False Positive Rate (FPR) vs Threshold at Tau=0.40
    ax2 = axes[1]
    fprs = {m: [data[b][m]['standard']['fpr'][40] for b in BACKBONES] for m in METHODS}
    for m_idx, m in enumerate(METHODS):
        ax2.plot(x11, fprs[m], marker=MARKERS[m], label=f'{m} (Mean FPR={np.mean(fprs[m]):.1f}%)',
                 color=COLORS[m], linewidth=2.0, markersize=7)
    
    ax2.set_title('(b) False Positive Rate on Negative Queries (τ=0.40)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticks(x11)
    ax2.set_xticklabels([DISPLAY_NAMES[b] for b in BACKBONES], rotation=40, ha='right', fontsize=9)
    ax2.set_ylabel('False Positive Rate (%) [Lower is Better]', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=9, loc='upper right')
    
    plt.tight_layout()
    fig3_png = out_dir / 'FIGURE_3_SCORE_MARGIN_AND_FPR.png'
    fig3_pdf = out_dir / 'FIGURE_3_SCORE_MARGIN_AND_FPR.pdf'
    fig.savefig(fig3_png, dpi=300, bbox_inches='tight')
    fig.savefig(fig3_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f'Generated {fig3_png} and {fig3_pdf}')

if __name__ == '__main__':
    main()
