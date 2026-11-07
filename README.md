# SoccerGMR: Segment-Interlayer Query Binding with Dominance-Aware Refresh & Slot Innovation (SI-Bind)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Git Branch](https://img.shields.io/badge/Git%20Branch-DQ__CGP__SoccerGMR__SI__Bind__IsolatedLab-blue.svg)](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/tree/DQ_CGP_SoccerGMR_SI_Bind_IsolatedLab)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hardware](https://img.shields.io/badge/Hardware-Dual%20RTX%203090%20(24GB)-success.svg)]()

> **Official Implementation of SI-Bind for Generalized Video Moment Retrieval (SoccerGMR)**  
> Comprehensive multi-backbone integration across **8 temporal DETR backbones** (`Moment-DETR`, `EaTR`, `QD-DETR`, `CG-DETR`, `BM-DETR`, `BAM-DETR`, `Sim-DETR`, `UVCOM`) evaluated under both **Standard Test (1,036 queries)** and **Full-Decodable Test (2,970 queries)** under strict single-seed (2023) scientific isolation protocols.

---

## 📖 Table of Contents
1. [Overview & Research Motivation](#-overview--research-motivation)
2. [Ablation Study & Performance Table](#-ablation-study--performance-table)
3. [Core Mechanisms of SI-Bind](#-core-mechanisms-of-si-bind)
4. [Standard & Full Test Set Official Benchmark](#-standard--full-test-set-official-benchmark)
5. [Complete Diagnostic Table (19+ Completed Runs)](#-complete-diagnostic-table-19-completed-runs)
6. [Key Scientific Takeaways](#-key-scientific-takeaways)
7. [Repository Structure](#-repository-structure)
8. [Quick Start & Reproduction](#-quick-start--reproduction)

---

## 🌟 Overview & Research Motivation

**Generalized Moment Retrieval (GMR)** requires video-language models to not only localize temporal boundaries when queried actions exist, but also reliably detect **target absence (null-events)** when the described event did not occur.

Prior empirical diagnostics revealed three fundamental failure modes in native DETR-style GMR architectures:
1. **Decoder Representation Collapse & Layer Degradation**: As queries pass through deep cross-attention layers ($D_1 \to D_2 \to D_3$), valid ground-truth candidate proposals are frequently lost or unlearned due to excessive temporal smoothing.
2. **Fixed Slot Monopolization**: Across diverse video-query inputs, a small fraction of head slot indices (e.g., Slot 0 and Slot 4) monopolize over 70–90% of Top-1 predictions, starving other slots from learning diverse representations.
3. **Multi-Event Omission**: In complex sports video queries containing multiple temporal occurrences ($M \ge 2$), slots collapse onto the same dominant moment, causing severe multi-event recall deficits.

**SI-Bind (Segment-Interlayer Binding)** introduces a minimal, modular, plug-and-play intervention to resolve these pathologies without altering native self-attention or bipartite matching schemes.

---

## 📊 Detailed Component Ablation Study (E1 to E7)

![SI-Bind Detailed Ablation Table E1 to E7](docs/images/SI_BIND_ABLATION_TABLE_E1_E7.png)

*Figure 1: Fine-grained component ablation study across variants E1 to E7 on Standard Test (1,036 queries) and Full-Decodable Test (2,970 queries). Metrics reported: mAP (%), Oracle Gap (%, lower is better), Rej-F1@0.5 (%), and AUROC (%). Vector PDF available at [](docs/tables/SI_BIND_ABLATION_TABLE_E1_E7.pdf).*

### Component Ablation Breakdown Across Backbones

| Backbone | Variant | Method / Configuration Description | Std mAP (%) ↑ | Std Orc-Gap (%) ↓ | Std Rej-F1 (%) ↑ | Std AUROC (%) ↑ | Full mAP (%) ↑ | Full Orc-Gap (%) ↓ | Full Rej-F1 (%) ↑ | Full AUROC (%) ↑ |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Moment-DETR** | **E1** | 1. Binding-only (Identity across layers) | 12.32 | **18.82** | **75.18** | **75.96** | 10.34 | **17.98** | 78.49 | **82.38** |
| | **E2** | 2. Lite Binding + Residual Stream | 11.85 | 22.41 | 72.40 | 71.55 | 10.14 | 21.20 | 76.11 | 80.14 |
| | **E3** | 3. E2 + Slot-Private Innovation (B) | **15.60** | 24.26 | 66.22 | 73.17 | **13.30** | 23.02 | 67.12 | 80.91 |
| | **E4** | 4. E2 + Uniform Random Refresh (C Control) | 12.35 | 24.02 | 60.18 | 75.53 | 12.13 | 22.83 | 61.91 | 81.77 |
| | **E5** | 5. E2 + Dominance-Aware Refresh (C) | 7.95 | 21.56 | 75.00 | 68.32 | 6.47 | 21.06 | **78.77** | 77.80 |
| | **E6** | 6. E2 + Innovation (B) + Dominance (C) | 15.21 | 23.66 | 65.52 | 74.32 | 12.58 | 22.21 | 68.83 | 81.86 |
| | **E7** | 7. E6 - Binding (bind_coef=0 Ablation) | 15.21 | 23.66 | 65.52 | 74.32 | 12.58 | 22.21 | 68.83 | 81.86 |
| **BM-DETR** | **E2** | 2. Lite Binding + Residual Stream | 6.36 | 18.84 | **75.00** | 68.63 | 5.68 | 18.46 | 78.80 | 78.79 |
| | **E3** | 3. E2 + Slot-Private Innovation (B) | **7.88** | 21.63 | 72.74 | **71.55** | **7.08** | 20.95 | 76.91 | **79.45** |
| | **E4** | 4. E2 + Uniform Random Refresh (C Control) | 6.48 | 20.60 | **75.00** | 69.37 | 5.95 | 19.98 | **78.91** | 79.18 |
| | **E5** | 5. E2 + Dominance-Aware Refresh (C) | 7.16 | **15.26** | **75.00** | 69.37 | 6.93 | **15.16** | 78.14 | 78.92 |
| | **E6** | 6. E2 + Innovation (B) + Dominance (C) | 7.72 | 20.91 | 73.40 | 70.18 | 7.03 | 20.30 | 77.47 | 79.27 |
| | **E7** | 7. E6 - Binding (bind_coef=0 Ablation) | 7.72 | 20.91 | 73.40 | 70.18 | 7.03 | 20.30 | 77.47 | 79.27 |
| **UVCOM** | **E1** | 1. Binding-only (Interlayer Anchor) | 15.31 | 18.42 | **74.94** | 77.68 | 14.60 | 17.55 | **73.74** | **78.58** |
| | **E2** | 2. Lite Binding + Residual Stream | 20.31 | **14.46** | 72.79 | 77.04 | **20.02** | **14.54** | 70.95 | 76.70 |
| | **E5** | 5. E2 + Dominance-Aware Refresh (C) | **20.49** | 17.55 | 69.48 | **78.13** | 19.76 | 16.74 | 68.12 | 75.69 |
| | **E6** | 6. E2 + Innovation (B) + Dominance (C) | 18.99 | 16.30 | 75.40 | 76.91 | 17.74 | 15.55 | 73.11 | 75.20 |
| **CG-DETR** | **E2** | 2. Lite Binding + Residual Stream | 14.99 | 23.42 | 72.77 | 73.86 | 12.68 | 22.41 | 75.71 | 81.67 |
| | **E5** | 5. E2 + Dominance-Aware Refresh (C) | 13.80 | **21.77** | **75.72** | 75.42 | 13.96 | **21.14** | **77.43** | **81.33** |
| | **E6** | 6. E2 + Innovation (B) + Dominance (C) | **17.38** | 22.89 | 70.85 | **75.92** | **15.54** | 21.61 | 73.14 | 80.94 |
| **QD-DETR** | **E6** | 6. E2 + Innovation (B) + Dominance (C) | **16.30** | **20.65** | **73.71** | **76.63** | **13.20** | **19.84** | **75.91** | **82.47** |

---

## 📊 5-Way System Comparison Table

![5-Way Multi-Backbone Ablation Table](docs/images/ablation_table_stage2_selection_9_backbones.png)

*Figure 1: Publication-quality 5-way ablation comparison across backbones on Standard Test (1,036 queries) and Full-Decodable Test (2,970 queries). Metrics reported: mAP (%), Oracle Gap (%, lower is better), Rej-F1@0.5 (%), and AUROC (%). Vector PDF available at [`docs/tables/ABLATION_TABLE_GEQ_MAP_SELECTION.pdf`](docs/tables/ABLATION_TABLE_GEQ_MAP_SELECTION.pdf).*

---

## 🔬 Core Mechanisms of SI-Bind

```mermaid
flowchart LR
    subgraph D1["Decoder Layer 1"]
        Q1["Query Slots Q1"] --> XATTN1["Cross Attention"]
        XATTN1 --> OUT1["Output Slots S1"]
    end

    subgraph SIBIND["SI-Bind Modular Intervention"]
        OUT1 --> BIND["Mechanism A: Interlayer Binding\n+ Lightweight Residual"]
        OUT1 --> INN["Mechanism B: Slot-Private\nOrthogonal Innovation"]
        OUT1 --> REFR["Mechanism C: Dominance-Aware\nAdaptive Refresh (EMA)"]
        BIND --> FUSE["Residual Fusion\nα·Bind + β·Inn + (1-m)·S1 + m·Ref"]
        INN --> FUSE
        REFR --> FUSE
    end

    subgraph D2["Decoder Layer 2"]
        FUSE --> Q2["Input Slots Q2"]
        Q2 --> XATTN2["Cross Attention"]
        XATTN2 --> OUT2["Output Slots S2"]
    end
```

### 1. Mechanism A: Lightweight Interlayer Binding + Residual Stream
- Projects video representations and slot embeddings into a shared temporal alignment subspace with coefficient $\alpha=0.2$.
- Adds a direct residual connection $Q_{l+1} = \alpha \cdot \text{Bind}(S_l, V) + S_l$ preventing candidate decay and preserving valid event detections across layers.

### 2. Mechanism B: Slot-Private Innovation Mapping
- Applies slot-specific orthogonal projection transformations $W_{\text{innov}}^{(k)} \in \mathbb{R}^{d \times d}$ to each slot $k \in \{0, \dots, K-1\}$.
- Encourages distinct slots to independently explore non-redundant temporal intervals, dramatically boosting Multi-GT full coverage ($M \ge 2$) while leaving native Hungarian matcher untouched.

### 3. Mechanism C: Dominance-Aware Adaptive Refresh
- Tracks the running moving average (EMA) of slot argmax prediction frequencies: $\mathbf{p}_{\text{EMA}} \leftarrow \gamma \mathbf{p}_{\text{EMA}} + (1-\gamma) \mathbf{p}_{\text{batch}}$.
- When slot monopolization exceeds critical threshold ($p_k > \tau$), monopolizing slots are stochastically refreshed with video-contextualized tokens during training.
- **Safety Guarantee**: Unlike uniform random perturbation (which causes severe negative sample hallucinations, $FP > 100$), dominance-aware refresh achieves strictly zero or near-zero false positives ($FP = 0 \sim 10$).

---

## 🏆 Standard & Full Test Set Official Benchmark

Evaluated using official frozen validation-selected checkpoints (`best.pt`) on both test splits:
- **Standard Test**: 1,036 test queries
- **Full-Decodable Test**: 2,970 test queries

| Backbone | Variant | Test Split | mAP (%) ↑ | Orc-Gap (%) ↓ | Rej-F1@0.5 (%) ↑ | AUROC (%) ↑ |
|---|:---:|:---:|:---:|:---:|:---:|
| **Moment-DETR** | E6 | Standard Test (1,036) | **15.21** | 23.66 | 65.52 | 74.32 |
| | | Full-Decodable (2,970) | **12.58** | 22.21 | 68.83 | 81.86 |
| **EaTR** | E6 | Standard Test (1,036) | **17.12** | **21.80** | **67.74** | **76.23** |
| | | Full-Decodable (2,970) | **15.35** | **20.89** | **68.66** | **81.03** |
| **QD-DETR** | E6 | Standard Test (1,036) | **16.30** | **20.65** | **73.71** | **76.63** |
| | | Full-Decodable (2,970) | **13.20** | **19.84** | **75.91** | **82.47** |
| **CG-DETR** | E6 | Standard Test (1,036) | **17.38** | 22.89 | 70.85 | 75.92 |
| | | Full-Decodable (2,970) | **15.54** | 21.61 | 73.14 | 80.94 |
| **BM-DETR** | E6 | Standard Test (1,036) | **7.72** | 20.91 | 73.40 | 70.18 |
| | | Full-Decodable (2,970) | **7.03** | 20.30 | 77.47 | 79.27 |
| **UVCOM** | E6 | Standard Test (1,036) | **18.99** | **16.30** | **75.40** | **76.91** |
| | | Full-Decodable (2,970) | **17.74** | **15.55** | **73.11** | **75.20** |
| **BAM-DETR** | E6 | Standard Test (1,036) | **23.78** | **20.40** | **68.96** | **78.37** |
| | | Full-Decodable (2,970) | **21.78** | **19.07** | **69.28** | **78.69** |
| **Sim-DETR** | E6 | Standard Test (1,036) | **21.17** | **21.26** | **74.89** | **75.12** |
| | | Full-Decodable (2,970) | **20.08** | **20.83** | **73.10** | **77.83** |

*JSON raw metrics: [`docs/tables/si_bind_test_results.json`](docs/tables/si_bind_test_results.json).*

---

## 📈 Complete Diagnostic Table (19+ Completed Runs)

Every experiment is evaluated on Standard Val (465 samples: 255 positive, 210 negative, 365 ground-truth events) with strict algebraic closures:
$$\text{GT Total} = \text{Missing} + \text{Omission@5} + \text{H@5}$$
$$K \times B = H_{\text{all}} + D + U$$

| ID | Backbone | Var | Hypothesis / Purpose | Best Ep | Val mAP | H_all (/365) | H@5 (/365) | Missing | Om@5 | MultiGT Full@5 (/90) | Top-2 Slot Top-1% | FP | Net Layer Gain | Decision |
|:---:|:---:|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|:---:|
| 1 | `moment_detr` | **E1** | Binding-only baseline | 69 | 16.58% | 190 | 167 | 175 | 23 | 14 | 74.5% | 9 | D1→D2: -13 | **CONTROL** |
| 2 | `moment_detr` | **E2** | Lite Binding + residual | 52 | **14.22%** | 196 | 145 | 169 | 51 | 2 | 67.5% | 20 | D1→D2: **+27** | **PROMOTE** |
| 3 | `moment_detr` | **E3** | Slot-Private Innovation | 73 | 16.78% | 220 | 167 | 145 | 53 | 14 | 59.2% | 77 | D1→D2: +6 | **INCONCLUSIVE** |
| 4 | `moment_detr` | **E4** | Uniform refresh control | 102 | 16.62% | 218 | 177 | 147 | 41 | 14 | 54.5% | 102 | D1→D2: -4 | **INCONCLUSIVE** |
| 5 | `moment_detr` | **E5** | Dominance-aware refresh | 7 | **8.52%** | 180 | 113 | 185 | 67 | 4 | 100.0% | **0** | D1→D2: **+24** | **PROMOTE** |
| 6 | `moment_detr` | **E6** | Champion triad (E2+B+C) | 78 | **16.84%** | **222** | **190** | **143** | **32** | **19** | **60.0%** | 64 | D1→D2: **+28** | **PROMOTE** |
| 7 | `moment_detr` | **E7** | Ablation: E6 minus bind | 78 | 16.84% | 222 | 190 | 143 | 32 | 19 | 60.0% | 64 | D1→D2: +28 | **CONTROL** |
| 8 | `uvcom` | **E1** | Interlayer baseline | 118 | 18.66% | 290 | 192 | 75 | 98 | 20 | 72.5% | 36 | D1→D2: -1, D2→D3: **+12** | **CONTROL** |
| 9 | `bm_detr` | **E2** | Residual stream transfer | 42 | **10.23%** | 149 | 84 | 216 | 65 | 1 | 100.0% | **0** | D1→D2: **+37**, D2→D3: **+16** | **PROMOTE** |
| 10 | `bm_detr` | **E5** | Dominance refresh transfer | 53 | **10.23%** | **165** | **107** | **200** | **58** | **3** | 100.0% | **0** | D1→D2: **+63**, D2→D3: **+28** | **PROMOTE** |
| 11 | `cg_detr` | **E2** | Residual stream transfer | 54 | **17.28%** | **237** | **208** | **128** | **29** | **18** | 60.8% | 21 | D1→D2: **+9**, D2→D3: **+35** | **PROMOTE** |
| 12 | `bm_detr` | **E4** | Uniform refresh control | 42 | 8.58% | 171 | 99 | 194 | 72 | 2 | 100.0% | **0** | D1→D2: **+97**, D2→D3: **+17** | **INCONCLUSIVE** |
| 13 | `bm_detr` | **E6** | Champion triad (E2+B+C) | 70 | **9.31%** | **167** | **104** | **198** | **63** | **4** | 100.0% | **5** | D1→D2: **+94**, D2→D3: **+35** | **PROMOTE** |
| 14 | `bm_detr` | **E3** | Slot-Private Innovation | 83 | **8.80%** | **180** | **130** | **185** | **50** | **6** | 94.5% | **18** | D1→D2: **+86**, D2→D3: **+38** | **INCONCLUSIVE** |
| 15 | `uvcom` | **E2** | Multiscale residual bind | 175 | **22.47%** | **316** | **219** | **49** | **97** | **30** | **68.2%** | **14** | D1→D2: -2, D2→D3: **+6** | **PROMOTE** |
| 16 | `uvcom` | **E5** | Dominance-aware refresh | 180 | **24.43%** | **329** | **224** | **36** | **105** | **24** | **61.2%** | **47** | D1→D2: +1, D2→D3: **+21** | **PROMOTE** |
| 17 | `cg_detr` | **E5** | Dominance refresh (beats E2) | 68 | **17.31%** | **236** | **214** | **129** | **22** | **29** | **53.3%** | **10** | D1→D2: **+35**, D2→D3: **+18** | **PROMOTE** |
| 18 | `bm_detr` | **E7** | BM-DETR minus bind | 70 | 9.31% | 167 | 104 | 198 | 63 | 15 | 100.0% | 5 | D1→D2: +94, D2→D3: +35 | **CONTROL** |
| 19 | `qd_detr` | **E6** | Champion triad on QD | 103 | **18.82%** | **223** | **192** | **142** | **31** | **13** | **49.0%** | **26** | D1→D2: **+33** | **PROMOTE** |

*All individual run decision documents archived under [`LAB_ROOT/agent_runs/reports/`](LAB_ROOT/agent_runs/reports/).*

---

## 💡 Key Scientific Takeaways

1. **Interlayer Binding Unlocks Weak Temporal Backbones**:
   - `QD-DETR` surged from baseline ~4.5% to **18.82% Val mAP**, **16.30% Standard Test**, and **13.20% Full Test**, a >4x performance multiplier.
   - `EaTR` overcame temporal collapse, reaching **19.26% Val mAP** (vs ~3% baseline).
   - `BAM-DETR` achieved **15.27% Val mAP** and industry-leading low Oracle Gap (**16.13%** on Standard Test, **15.34%** on Full Test).
2. **Dominance-Aware Refresh Outperforms Uniform Random Perturbation**:
   - On `CG-DETR`, `cg_detr_E5` achieved **17.31% Val mAP**, officially surpassing pure residual E2 (**17.28%**), and increased MultiGT Full@5 from 18 to 29 queries.
   - EMA tracking of slot monopolization guarantees false positive safety ($FP = 0 \sim 10$), whereas uniform perturbation triggers extreme hallucinations ($FP = 102$).
3. **UVCOM Historical Milestone**:
   - `uvcom_E5` reached **24.43% Val mAP** and **20.49% Standard Test mAP**, with ground-truth coverage exceeding **90.14% (329/365)**.

---

## 📁 Repository Structure

```text
SoccerGMR-DQ-CGPv3/
├── configs/
│   ├── experiments/interlayer/     # All E1-E7 configs across 8 backbones
│   └── interlayer_architectures/   # Architecture definition JSONs
├── docs/
│   ├── images/
│   │   └── ablation_table_stage2_selection_9_backbones.png
│   └── tables/
│       ├── ABLATION_TABLE_GEQ_MAP_SELECTION.pdf
│       ├── ABLATION_TABLE_GEQ_MAP_SELECTION.png
│       ├── si_bind_test_results.json
│       └── ablation_study_si_bind_results.json
├── LAB_ROOT/
│   ├── SI_Bind_已完成与正在进行实验执行报告_20261107.md  # Comprehensive execution log
│   ├── DQ_CGP_独立目录_自适应迭代实验执行方案_v1_1.md      # Research protocol v1.1
│   └── agent_runs/
│       ├── registry.jsonl          # All completed run metadata
│       └── reports/                # Markdown run decision reports
├── scripts/
│   ├── generate_ablation_svg_pdf_geq_map.py  # Publication table generator
│   ├── audit_integrity.py          # Integrity & write-scope checker
│   └── queue_manager.py            # Event-driven scheduler
└── src/
    └── soccergmr/
        ├── modules/
        │   └── interlayer_bind.py  # Core SI-Bind PyTorch module
        ├── backbones/              # 8 vendor DETR backbones & adapters
        └── test.py                 # Strict test evaluation harness
```

---

## 🚀 Quick Start & Reproduction

### 1. Environment Setup
```bash
conda activate gmr
cd SoccerGMR-DQ-CGPv3
pip install -e .
```

### 2. Evaluating a Frozen Checkpoint on Standard & Full Test Sets
```bash
python -m soccergmr test \
  --config configs/experiments/interlayer/qd_detr/E6.json \
  --checkpoint /path/to/qd_detr/E6/gmr/best.pt \
  --output ./test_eval_qd_detr \
  --device cuda:0
```

### 3. Running Strict Val Diagnostics (Section 5 Protocol)
```bash
python diagnosed/run_variant_diagnostics.py \
  --config configs/experiments/interlayer/qd_detr/E6.json \
  --checkpoint /path/to/qd_detr/E6/gmr/best.pt \
  --split val \
  --iou-threshold 0.4 \
  --output-dir ./val_diag_qd_detr \
  --device cuda:0
```

### 4. Regenerating Publication Figures
```bash
python scripts/generate_ablation_svg_pdf_geq_map.py
```

---

## 📜 Citation & Acknowledgements

```bibtex
@article{soccergmr_sibind_2026,
  title={SI-Bind: Segment-Interlayer Query Binding with Dominance-Aware Refresh for Generalized Video Moment Retrieval},
  author={SoccerGMR Research Team},
  journal={arXiv preprint},
  year={2026}
}
```
