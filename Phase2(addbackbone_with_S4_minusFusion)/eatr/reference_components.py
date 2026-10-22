"""Verbatim components extracted from FlashVTG S4_minus_Fusion; see SOURCE_MANIFEST.json."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class EvidenceRead(nn.Module):
    def __init__(self, width=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, n_heads, dropout=dropout, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(width, width * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width * 2, width)
        )
        self.norm_attention = nn.LayerNorm(width)
        self.norm_feedforward = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, state, memory, valid=None):
        key_padding_mask = ~valid.bool() if valid is not None else None
        evidence, attention = self.attention(
            state, memory, memory, key_padding_mask=key_padding_mask,
            need_weights=True, average_attn_weights=False
        )
        state = self.norm_attention(state + self.dropout(evidence))
        state = self.norm_feedforward(state + self.dropout(self.feedforward(state)))
        attn_out = attention.mean(dim=1).squeeze(1) if attention.ndim == 4 else attention.mean(dim=1)
        return state, attn_out

class CandidateMomentExtractor(nn.Module):
    """Extracts top-K candidate moments from localizer proposals and encodes temporal RoI features."""
    def __init__(self, hidden_dim=256, top_k=8, clip_length=2.0, max_duration=150.0, pooling_type='gaussian'):
        super().__init__()
        self.top_k = top_k
        self.clip_length = clip_length
        self.max_duration = max_duration
        self.pooling_type = pooling_type
        self.geom_mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, final_video, video_mask, candidate_spans):
        """
        Args:
            final_video: (B, T, D), T <= 75
            video_mask: (B, T) bool
            candidate_spans: (B, K, 3) where each row is [st, ed, conf]
        Returns:
            moment_tokens: (B, K, D)
            moment_confs: (B, K)
            moment_spans: (B, K, 2) in seconds
        """
        bsz = candidate_spans.shape[0]
        K = candidate_spans.shape[1]
        device = candidate_spans.device
        
        st = candidate_spans[:, :, 0].clamp(min=0.0, max=self.max_duration)
        ed = candidate_spans[:, :, 1].clamp(min=0.0, max=self.max_duration)
        confs = candidate_spans[:, :, 2].clamp(min=0.0, max=1.0)
        spans_2d = torch.stack([st, ed], dim=-1) # (B, K, 2)
        
        # Temporal RoI pooling from final_video
        T = final_video.shape[1]
        frame_centers = (torch.arange(T, device=device, dtype=torch.float32) * self.clip_length + self.clip_length / 2.0) # (T,)
        
        cand_center = (st + ed).unsqueeze(-1) / 2.0 # (B, K, 1)
        cand_half_w = (ed - st).clamp(min=self.clip_length).unsqueeze(-1) / 2.0 # (B, K, 1)
        
        if self.pooling_type == 'uniform_box':
            # Hard rectangular window: 1.0 inside [st, ed], 0.0 outside
            in_window = (frame_centers.view(1, 1, T) >= (st.unsqueeze(-1) - 1.0)) & (frame_centers.view(1, 1, T) <= (ed.unsqueeze(-1) + 1.0))
            weights = in_window.float()
            # If window happens to fall between discrete grid points, fall back to nearest frame
            empty = (weights.sum(dim=-1, keepdim=True) == 0)
            if empty.any():
                nearest_idx = (cand_center / self.clip_length).long().clamp(0, T - 1)
                fallback = torch.zeros_like(weights).scatter_(-1, nearest_idx, 1.0)
                weights = torch.where(empty, fallback, weights)
        elif self.pooling_type == 'random_box':
            # Control experiment: sample random moments uncorrelated with proposals
            # Use candidate coordinates to deterministically generate uncorrelated pseudo-random moments
            rand_u = torch.sin(st * 12.9898 + ed * 78.233).abs().unsqueeze(-1)
            rand_st = rand_u * (self.max_duration - 25.0)
            rand_ed = (rand_st + 15.0).clamp(max=self.max_duration)
            r_center = (rand_st + rand_ed) / 2.0
            r_half_w = (rand_ed - rand_st) / 2.0
            dist_sq = (frame_centers.view(1, 1, T) - r_center).square()
            weights = torch.exp(-dist_sq / (2.0 * r_half_w.square() + 1e-5))
        else: # 'gaussian'
            dist_sq = (frame_centers.view(1, 1, T) - cand_center).square()
            weights = torch.exp(-dist_sq / (2.0 * cand_half_w.square() + 1e-5))
            
        weights = weights * video_mask.unsqueeze(1).float()
        weights_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-5)
        norm_weights = weights / weights_sum # (B, K, T)
        
        # RoI video feature: (B, K, D)
        roi_video = torch.bmm(
            norm_weights.view(bsz * K, 1, T),
            final_video.unsqueeze(1).expand(-1, K, -1, -1).reshape(bsz * K, T, -1)
        ).view(bsz, K, -1)
        
        # Geometry embedding: [st/150, ed/150, duration/150, conf]
        geom_in = torch.stack([
            st / self.max_duration,
            ed / self.max_duration,
            (ed - st) / self.max_duration,
            confs,
        ], dim=-1) # (B, K, 4)
        geom_emb = self.geom_mlp(geom_in) # (B, K, D)
        
        moment_tokens = self.norm(roi_video + geom_emb)
        return moment_tokens, confs, spans_2d

def extract_saliency_features(saliency, spans, clip_length=2.0):
    """
    Extracts frame-level saliency gradients and interior intensity for candidate spans.
    saliency: (B, T)
    spans: (B, K, 3) where [st, ed, conf]
    Returns:
        sal_feats: (B, K, 3) where [grad_st, grad_ed, sal_mid]
    """
    bsz, K, _ = spans.shape
    T = saliency.shape[1]
    device = saliency.device
    
    st = spans[:, :, 0]
    ed = spans[:, :, 1]
    
    idx_s = (st / clip_length).long().clamp(0, T - 1)
    idx_e = (ed / clip_length).long().clamp(0, T - 1)
    idx_mid = ((st + ed) / (2.0 * clip_length)).long().clamp(0, T - 1)
    
    b_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(-1, K)
    
    s_before = saliency[b_idx, (idx_s - 1).clamp(0, T - 1)]
    s_after  = saliency[b_idx, (idx_s + 1).clamp(0, T - 1)]
    s_before2 = saliency[b_idx, (idx_s - 2).clamp(0, T - 1)]
    s_after2  = saliency[b_idx, (idx_s + 2).clamp(0, T - 1)]
    
    e_before = saliency[b_idx, (idx_e - 1).clamp(0, T - 1)]
    e_after  = saliency[b_idx, (idx_e + 1).clamp(0, T - 1)]
    e_before2 = saliency[b_idx, (idx_e - 2).clamp(0, T - 1)]
    e_after2  = saliency[b_idx, (idx_e + 2).clamp(0, T - 1)]
    
    s_mid    = saliency[b_idx, idx_mid]
    
    grad_st = s_after - s_before
    grad_ed = e_before - e_after
    grad_st_wide = s_after2 - s_before2
    grad_ed_wide = e_before2 - e_after2
    s_contrast = s_mid - 0.5 * (s_before + e_after)
    
    return torch.stack([grad_st, grad_ed, s_mid, grad_st_wide, grad_ed_wide, s_contrast], dim=-1)

class TemporalBoundaryRefiner(nn.Module):
    """
    Refines candidate temporal boundaries [st, ed, conf] conditioned on text-moment cross-attention
    and continuous frame-level saliency gradients.
    Operates specifically on top-K proposals to improve temporal mAP on positive queries.
    """
    def __init__(self, hidden_dim=256, max_delta=8.0, use_cand_self_attn=False):
        super().__init__()
        self.max_delta = max_delta
        self.use_cand_self_attn = use_cand_self_attn
        self.cand_text_attn = nn.MultiheadAttention(hidden_dim, 8, dropout=0.1, batch_first=True)
        self.norm_attn = nn.LayerNorm(hidden_dim)
        if self.use_cand_self_attn:
            self.cand_self_attn = nn.MultiheadAttention(hidden_dim, 4, dropout=0.1, batch_first=True)
            self.norm_self = nn.LayerNorm(hidden_dim)
            nn.init.zeros_(self.cand_self_attn.out_proj.weight)
            nn.init.zeros_(self.cand_self_attn.out_proj.bias)
        self.geom_proj = nn.Linear(4, hidden_dim)
        nn.init.zeros_(self.geom_proj.weight)
        nn.init.zeros_(self.geom_proj.bias)
        self.sal_proj = nn.Linear(6, hidden_dim)
        nn.init.zeros_(self.sal_proj.weight)
        nn.init.zeros_(self.sal_proj.bias)
        
        self.refine_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3) # [delta_st, delta_ed, delta_conf]
        )
        # Initialize final layer near zero so training starts with identity refinement
        nn.init.zeros_(self.refine_mlp[-1].weight)
        nn.init.zeros_(self.refine_mlp[-1].bias)

    def forward(self, moment_tokens, text_memory, text_mask, candidate_spans, saliency_scores=None):
        """
        Args:
            moment_tokens: (B, K, D)
            text_memory: (B, L, D)
            text_mask: (B, L) bool
            candidate_spans: (B, K, 3) where [st, ed, conf]
            saliency_scores: (B, T) float optional
        Returns:
            refined_spans: (B, K, 3)
            deltas: (B, K, 3)
        """
        key_padding_mask = ~text_mask.bool() if text_mask is not None else None
        cand_text_feat, _ = self.cand_text_attn(
            moment_tokens, text_memory, text_memory,
            key_padding_mask=key_padding_mask
        ) # (B, K, D)
        # Residual connection: preserve visual moment features conditioned on text semantics
        cand_refined_feat = self.norm_attn(moment_tokens + cand_text_feat)
        
        if self.use_cand_self_attn:
            cand_self_feat, _ = self.cand_self_attn(cand_refined_feat, cand_refined_feat, cand_refined_feat)
            cand_refined_feat = self.norm_self(cand_refined_feat + cand_self_feat)
        
        # Inject normalized span geometry [st/150, ed/150, dur/150, conf]
        st_norm = candidate_spans[:, :, 0].clamp(0.0, 150.0) / 150.0
        ed_norm = candidate_spans[:, :, 1].clamp(0.0, 150.0) / 150.0
        dur_norm = (candidate_spans[:, :, 1] - candidate_spans[:, :, 0]).clamp(min=0.0) / 150.0
        conf_norm = candidate_spans[:, :, 2].clamp(0.0, 1.0)
        geom_in = torch.stack([st_norm, ed_norm, dur_norm, conf_norm], dim=-1)
        cand_refined_feat = cand_refined_feat + self.geom_proj(geom_in)
        
        if saliency_scores is not None and saliency_scores.shape[0] == moment_tokens.shape[0]:
            sal_feats = extract_saliency_features(saliency_scores, candidate_spans)
            cand_refined_feat = cand_refined_feat + self.sal_proj(sal_feats)
        
        deltas = torch.tanh(self.refine_mlp(cand_refined_feat)) # in [-1, 1]
        delta_st = deltas[:, :, 0] * self.max_delta
        delta_ed = deltas[:, :, 1] * self.max_delta
        delta_conf = deltas[:, :, 2] * 0.3
        
        st = (candidate_spans[:, :, 0] + delta_st).clamp(min=0.0, max=150.0)
        ed = (candidate_spans[:, :, 1] + delta_ed).clamp(min=0.0, max=150.0)
        # Ensure positive non-zero window duration
        ed = torch.maximum(ed, st + 1.0)
        conf = (candidate_spans[:, :, 2] + delta_conf).clamp(min=0.0, max=1.0)
        
        refined_spans = torch.stack([st, ed, conf], dim=-1)
        return refined_spans, deltas

class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) layer for Transformer projections."""
    def __init__(self, base_linear, rank=8, alpha=16):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, base_linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        res = self.base_linear(x)
        lora_term = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return res + lora_term

def compute_boundary_refine_loss(refined_spans, metadata, device, job=None):
    """
    Computes SmoothL1 + tIoU loss on positive queries against ground truth windows.
    Zero loss on negative queries (protects against negative gradient contamination).
    Optionally aligns candidate confidences with max tIoU (IoU-quality aware ranking).
    """
    total_loss = torch.tensor(0.0, device=device)
    count = 0
    use_iou_conf = bool(job.get('refine_iou_conf', False)) if job else False
    
    for i, meta in enumerate(metadata):
        gt_windows = meta.get('relevant_windows', [])
        if not gt_windows:
            continue # Skip negative samples completely!
            
        gt_t = torch.tensor(gt_windows, device=device, dtype=torch.float32) # (M, 2)
        preds = refined_spans[i] # (K, 3)
        
        pred_st = preds[:, 0].unsqueeze(1) # (K, 1)
        pred_ed = preds[:, 1].unsqueeze(1) # (K, 1)
        gt_st = gt_t[:, 0].unsqueeze(0)    # (1, M)
        gt_ed = gt_t[:, 1].unsqueeze(0)    # (1, M)
        
        inter = (torch.minimum(pred_ed, gt_ed) - torch.maximum(pred_st, gt_st)).clamp(min=0.0)
        union = (pred_ed - pred_st) + (gt_ed - gt_st) - inter
        ious = inter / union.clamp(min=1e-5) # (K, M)
        
        # Candidate matching: default single best candidate per GT (preserves proposal diversity)
        best_cand_for_gt = ious.argmax(dim=0) # (M,)
        if bool(job.get('multi_cand_match', False)) if job else False:
            matched_cand_set = set(best_cand_for_gt.tolist())
            pred_sub = [preds[best_cand_for_gt[m]] for m in range(gt_t.shape[0])]
            gt_sub = [gt_t[m] for m in range(gt_t.shape[0])]
            min_iou_refine = float(job.get('min_iou_refine', 0.35)) if job else 0.35
            for c_idx in range(preds.shape[0]):
                if c_idx not in matched_cand_set:
                    max_val, best_m = ious[c_idx].max(dim=0)
                    if max_val >= min_iou_refine:
                        pred_sub.append(preds[c_idx])
                        gt_sub.append(gt_t[best_m])
            matched_preds = torch.stack(pred_sub, dim=0)
            matched_gt = torch.stack(gt_sub, dim=0)
        else:
            matched_preds = preds[best_cand_for_gt]
            matched_gt = gt_t
        
        # Loss: Normalized Smooth L1 on boundaries + tIoU loss
        norm_scale = 150.0
        l1_st = F.smooth_l1_loss(matched_preds[:, 0] / norm_scale, matched_gt[:, 0] / norm_scale, beta=0.01)
        l1_ed = F.smooth_l1_loss(matched_preds[:, 1] / norm_scale, matched_gt[:, 1] / norm_scale, beta=0.01)
        
        matched_inter = (torch.minimum(matched_preds[:, 1], matched_gt[:, 1]) - torch.maximum(matched_preds[:, 0], matched_gt[:, 0])).clamp(min=0.0)
        matched_union = (matched_preds[:, 1] - matched_preds[:, 0]) + (matched_gt[:, 1] - matched_gt[:, 0]) - matched_inter
        
        if bool(job.get('refine_giou', False)) if job else False:
            enc_st = torch.minimum(matched_preds[:, 0], matched_gt[:, 0])
            enc_ed = torch.maximum(matched_preds[:, 1], matched_gt[:, 1])
            enc_len = (enc_ed - enc_st).clamp(min=1e-5)
            giou = (matched_inter / matched_union.clamp(min=1e-5)) - (enc_len - matched_union) / enc_len
            reg_loss = (1.0 - giou).mean()
        else:
            reg_loss = (1.0 - (matched_inter / matched_union.clamp(min=1e-5))).mean()
            
        sample_loss = 2.0 * (l1_st + l1_ed) + 1.0 * reg_loss
        if use_iou_conf:
            max_iou_per_cand = ious.max(dim=1).values.detach() # (K,)
            conf_loss = F.binary_cross_entropy(preds[:, 2].clamp(1e-5, 1.0 - 1e-5), max_iou_per_cand)
            sample_loss = sample_loss + 0.5 * conf_loss
            
        total_loss = total_loss + sample_loss
        count += 1
        
    if count > 0:
        return total_loss / count
    return torch.tensor(0.0, device=device)
