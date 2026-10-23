"""Moment-Conditioned Event Verification (MCEV) with Temporal Boundary Refinement.

Decouples event verification while leveraging localized candidate temporal proposals
as concrete hypothesis evidence, resolving false rejections of localized positive moments,
and actively refining temporal boundaries (mAP) for true events.
"""
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
            in_window = (frame_centers.view(1, 1, T) >= st.unsqueeze(-1)) & (frame_centers.view(1, 1, T) <= ed.unsqueeze(-1))
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
    
    return torch.stack([grad_st, grad_ed, s_mid, grad_st_wide, grad_ed_wide, s_contrast], dim=-1) # (B, K, 6)


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


class MomentConditionedVerifier(nn.Module):
    """
    Moment-Conditioned Event Verification Module.
    Combines global contextual cross-attention with candidate moment verification and optional refinement.
    """
    def __init__(self, depth, job):
        super().__init__()
        self.job = dict(job)
        hidden_dim = 256
        self.use_moment_conditioning = bool(job.get('use_moment_conditioning', True))
        self.use_global_branch = bool(job.get('use_global_branch', True))
        self.use_boundary_refine = bool(job.get('use_boundary_refine', False))
        self.top_k = int(job.get('top_k', 8))
        
        # Global Event Readout Branch
        self.token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.text_read = EvidenceRead(hidden_dim)
        self.video_reads = nn.ModuleList([EvidenceRead(hidden_dim) for _ in range(depth)])
        self.global_head = nn.Linear(hidden_dim, 1)
        
        # Plurality Head on text
        self.plurality = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2)
        )
        
        if self.use_moment_conditioning:
            # Local Candidate Moment Branch
            self.extractor = CandidateMomentExtractor(hidden_dim=hidden_dim, top_k=self.top_k, pooling_type=job.get('pooling_type', 'gaussian'))
            self.moment_cross_attn = nn.MultiheadAttention(hidden_dim, 8, dropout=0.1, batch_first=True)
            self.moment_match_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 3 + 1, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )
            self.local_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            # Evidence Fusion Gate
            self.fusion_gate = nn.Sequential(
                nn.Linear(6, 32),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
            )
            nn.init.zeros_(self.fusion_gate[-1].weight)
            nn.init.zeros_(self.fusion_gate[-1].bias)

        if self.use_boundary_refine:
            max_delta = float(job.get('refine_max_delta', 8.0))
            use_cand_self_attn = bool(job.get('use_cand_self_attn', False))
            self.refiner = TemporalBoundaryRefiner(hidden_dim=hidden_dim, max_delta=max_delta, use_cand_self_attn=use_cand_self_attn)

        self.context = None

    def forward(self, _pooled_input):
        context = self.context
        assert context is not None
        bsz = context['text'].shape[0]
        
        # 1. Global Readout Branch
        state, text_attention = self.text_read(
            self.token.expand(bsz, -1, -1),
            context['text'],
            context['text_valid'],
        )
        text_state = state[:, 0]
        memories = context['video']
        if self.job.get('query_video_source') == 'final_repeated':
            memories = [memories[-1]] * len(self.video_reads)
        for reader, memory in zip(self.video_reads, memories):
            state, _ = reader(state, memory, context['video_valid'])
        global_state = state[:, 0]
        global_logit = self.global_head(global_state).squeeze(-1) # (B,)
        
        context['text_state'] = text_state
        context['plurality'] = self.plurality(text_state)
        context['global_logit'] = global_logit
        
        source = self.job.get('presence_source', 'fused' if self.use_global_branch else 'local')
        if source not in ('global_only', 'global_refine', 'fused', 'local'):
            raise ValueError(f'Unknown presence_source: {source}')
        if source == 'global_only' or not self.use_moment_conditioning or context.get('candidate_spans') is None:
            context['event_logit'] = global_logit
            return global_logit.unsqueeze(-1)
            
        # 2. Local Candidate Moment Verification Branch
        candidate_spans = context['candidate_spans'] # (B, K, 3)
        moment_tokens, moment_confs, moment_spans = self.extractor(
            final_video=context['final_video_units'],
            video_mask=context['final_video_valid'],
            candidate_spans=candidate_spans,
        )
        context['moment_tokens'] = moment_tokens
        
        # Optional: Temporal Boundary Refinement
        if self.use_boundary_refine:
            refined_spans, deltas = self.refiner(
                moment_tokens,
                context['text'],
                context['text_valid'],
                candidate_spans,
                saliency_scores=context.get('saliency_scores', None)
            )
            context['refined_spans'] = refined_spans
            context['refine_deltas'] = deltas
        else:
            context['refined_spans'] = candidate_spans
        
        # Decouple candidate localization from presence fusion.
        if source == 'global_refine':
            context['event_logit'] = global_logit
            return global_logit.unsqueeze(-1)

        # Text query attends to candidate moments
        q_expanded = text_state.unsqueeze(1) # (B, 1, D)
        local_attended, _ = self.moment_cross_attn(q_expanded, moment_tokens, moment_tokens)
        local_logit = self.local_head(local_attended.squeeze(1)).squeeze(-1) # (B,)
        
        # Explicit match scoring between query and each candidate moment
        K = candidate_spans.shape[1]
        q_rep = text_state.unsqueeze(1).expand(-1, K, -1) # (B, K, D)
        elem_prod = q_rep * moment_tokens
        match_in = torch.cat([q_rep, moment_tokens, elem_prod, moment_confs.unsqueeze(-1)], dim=-1)
        match_scores = self.moment_match_mlp(match_in).squeeze(-1) # (B, K)
        
        v_max = match_scores.max(dim=1).values
        v_top3 = match_scores[:, :min(3, K)].mean(dim=1)
        c_max = moment_confs.max(dim=1).values
        c_top3 = moment_confs[:, :min(3, K)].mean(dim=1)
        
        # 3. Evidence Fusion
        if source == 'fused':
            gate_inputs = torch.stack([global_logit, local_logit, v_max, v_top3, c_max, c_top3], dim=1)
            residual = self.fusion_gate(gate_inputs).squeeze(-1)
            context['gate_inputs'] = gate_inputs
            context['fusion_residual'] = residual
            final_logit = global_logit + residual
        else:
            final_logit = local_logit + 0.5 * v_max
        
        if self.job.get('evidence_gate', False) and self.job.get('presence_source') != 'global_only':
            suppress_thresh = float(self.job.get('evidence_thresh', 0.0))
            suppress_scale = float(self.job.get('evidence_scale', 1.0))
            evidence_suppress = F.relu(suppress_thresh - v_max) * suppress_scale
            context['evidence_suppress'] = evidence_suppress
            final_logit = final_logit - evidence_suppress
            
        logit_bias = float(self.job.get('logit_bias', 0.0))
        if logit_bias != 0.0:
            final_logit = final_logit + logit_bias
        
        context['local_logit'] = local_logit
        context['match_scores'] = match_scores
        context['v_max'] = v_max
        context['moment_confs'] = moment_confs
        context['moment_spans'] = moment_spans
        context['event_logit'] = final_logit
        return final_logit.unsqueeze(-1)


class MCEVModel(nn.Module):
    """Wraps FlashVTG with optional LoRA adaptation, MCEV existence head, and boundary refiner."""
    def __init__(self, base, job):
        super().__init__()
        self.base = base
        self.job = dict(job)
        self.depth = len(base.transformer.encoder.layers)
        self.top_k = int(job.get('top_k', 8))
        self.use_boundary_refine = bool(job.get('use_boundary_refine', False))
        self.use_lora = bool(job.get('use_lora', False))
        
        # Replace exist_head with MomentConditionedVerifier
        base.exist_head = MomentConditionedVerifier(self.depth, job)
        
        # Optional: Inject LoRA into Transformer Encoder
        if self.use_lora:
            lora_rank = int(job.get('lora_rank', 8))
            for layer in base.transformer.encoder.layers:
                layer.linear1 = LoRALinear(layer.linear1, rank=lora_rank)
                layer.linear2 = LoRALinear(layer.linear2, rank=lora_rank)
        
        self.gradient_attached = bool(job.get('gradient_attached', False))
        
        # Register hooks to capture intermediate representations
        self.handles = [
            base.input_txt_proj.register_forward_hook(self._capture_text),
            base.transformer.register_forward_hook(self._capture_transformer),
        ]
        self.handles.extend(
            layer.register_forward_hook(self._capture_video)
            for layer in base.transformer.encoder.layers
        )

    def _capture_text(self, _mod, _inp, output):
        if self.gradient_attached:
            self.base.exist_head.context['text'] = output
        else:
            self.base.exist_head.context['text'] = output.detach()

    def _capture_video(self, _mod, _inp, output):
        t_out = output.transpose(0, 1)
        if self.gradient_attached:
            self.base.exist_head.context['video'].append(t_out)
        else:
            self.base.exist_head.context['video'].append(t_out.detach())

    def _capture_transformer(self, _mod, _inp, output):
        context = self.base.exist_head.context
        t_out = output[0].transpose(0, 1)
        if self.gradient_attached:
            context['final_video_units'] = t_out
        else:
            context['final_video_units'] = t_out.detach()
        context['final_video_valid'] = (~output[1].bool()).detach()

    def forward(self, **inputs):
        head = self.base.exist_head
        assert head.context is None, "Context re-entrancy error"
        
        context = {
            'text_valid': inputs['src_txt_mask'].bool(),
            'video_valid': inputs['src_vid_mask'].bool(),
            'video': [],
        }
        head.context = context
        try:
            output = self.base(**inputs)
            
            # Capture frame saliency scores
            if 'saliency_scores' in output and output['saliency_scores'] is not None:
                context['saliency_scores'] = output['saliency_scores'] if self.gradient_attached else output['saliency_scores'].detach()
            elif '_out' in output and 'saliency' in output['_out'] and output['_out']['saliency'] is not None:
                sal = output['_out']['saliency'].unsqueeze(0)
                context['saliency_scores'] = sal if self.gradient_attached else sal.detach()
            else:
                context['saliency_scores'] = None
                
            # Extract candidate moments from base output
            # In train: output['pred_spans'] is (B, 139, 3) where [st, ed, conf]
            # In eval:  output['_out']['boundary'] is (50, 3) where [st, ed, conf]
            if self.job.get('presence_source') == 'global_only':
                top_spans = None
            elif 'pred_spans' in output:
                spans = output['pred_spans']
                if not self.gradient_attached:
                    spans = spans.detach()
                confs = spans[:, :, 2]
                _, top_idx = confs.topk(min(self.top_k, spans.shape[1]), dim=1)
                batch_idx = torch.arange(spans.shape[0], device=spans.device).unsqueeze(1).expand(-1, top_idx.shape[1])
                top_spans = spans[batch_idx, top_idx] # (B, K, 3)
            elif '_out' in output and 'boundary' in output['_out']:
                bnd = output['_out']['boundary']
                if not self.gradient_attached:
                    bnd = bnd.detach()
                top_spans = bnd[:self.top_k].unsqueeze(0) # (1, K, 3)
            else:
                top_spans = None
                
            context['candidate_spans'] = top_spans
            
            # Recompute existence logit with candidate spans
            logit = head(None).squeeze(-1)
            output['pred_exist_logits'] = logit
            output['pred_multi_logits'] = context['plurality']
            if 'v_max' in context:
                output['v_max'] = context['v_max']
            if 'match_scores' in context:
                output['match_scores'] = context['match_scores']
            
            # Propagate refined spans if refiner is enabled
            if self.use_boundary_refine and 'refined_spans' in context:
                output['refined_spans'] = context['refined_spans']
                if '_out' in output and 'boundary' in output['_out']:
                    bnd = output['_out']['boundary']
                    refined_bnd = context['refined_spans'][0] # (K, 3)
                    K_eff = min(self.top_k, bnd.shape[0], refined_bnd.shape[0])
                    bnd[:K_eff, :2] = refined_bnd[:K_eff, :2]
                    bnd[:K_eff, 2] = refined_bnd[:K_eff, 2]
                    # Sort candidates by confidence descending
                    sort_idx = torch.argsort(bnd[:, 2], descending=True)
                    output['_out']['boundary'] = bnd[sort_idx]
            
            if '_out' in output:
                output['_out']['pred_exist_score'] = torch.sigmoid(logit).detach().cpu()
            return output
        finally:
            head.context = None


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


def compute_mcev_loss(output, metadata, targets, job, model=None, model_inputs=None):
    """
    Computes balanced loss for MCEV:
    - Base BCE on existence label
    - Plurality cross-entropy on positive queries
    - Balanced Margin Contrastive loss with semantic hard negatives
    - Optional Boundary Refinement loss on positive queries
    - Optional Evidence Alignment loss on candidate moment match scores
    """
    existence = targets['exist_label'].float().to(output['pred_exist_logits'].device)
    details = {}
    
    prob = torch.sigmoid(output['pred_exist_logits'])
    if job.get('focal', False):
        pt = prob * existence + (1.0 - prob) * (1.0 - existence)
        loss_exist = (-0.5 * (1.0 - pt).square() * torch.log(pt.clamp_min(1e-7))).mean()
    else:
        loss_exist = F.binary_cross_entropy_with_logits(output['pred_exist_logits'], existence)
        
    if job.get('margin_calibrated', False):
        tau_neg = float(job.get('tau_neg', 0.25))
        tau_pos = float(job.get('tau_pos', 0.75))
        margin_neg = F.relu(prob - tau_neg).square() * (1.0 - existence)
        margin_pos = F.relu(tau_pos - prob).square() * existence
        margin_weight = float(job.get('margin_weight', 2.0))
        loss_margin = margin_weight * (margin_neg.mean() + margin_pos.mean())
        loss_exist = loss_exist + loss_margin
        details['loss_margin'] = loss_margin

    if job.get('balanced_safeguard', False):
        tau_neg = float(job.get('tau_neg', 0.35))
        tau_pos = float(job.get('tau_pos', 0.65))
        pos_weight = float(job.get('pos_safeguard_weight', 3.0))
        neg_weight = float(job.get('neg_safeguard_weight', 1.5))
        margin_neg = F.relu(prob - tau_neg).square() * (1.0 - existence)
        margin_pos = F.relu(tau_pos - prob).square() * existence
        loss_safeguard = neg_weight * margin_neg.mean() + pos_weight * margin_pos.mean()
        loss_exist = loss_exist + loss_safeguard
        details['loss_safeguard'] = loss_safeguard

    if job.get('evidence_alignment', False) and 'v_max' in output:
        loss_evidence = F.binary_cross_entropy_with_logits(output['v_max'], existence)
        evidence_align_weight = float(job.get('evidence_align_weight', 0.4))
        loss_exist = loss_exist + evidence_align_weight * loss_evidence
        details['loss_evidence'] = loss_evidence
    
    details['loss_exist'] = loss_exist
    total_loss = loss_exist
    
    # 2. Plurality classification loss on positive queries
    positive = existence > 0.5
    if positive.any() and float(job.get('multi_weight', 0.1)) > 0:
        plural = torch.tensor(
            [len(row.get('relevant_windows', [])) >= 2 for row in metadata],
            device=existence.device,
            dtype=torch.long,
        )
        loss_multi = F.cross_entropy(output['pred_multi_logits'][positive], plural[positive])
        multi_weight = float(job.get('multi_weight', 0.1))
        total_loss = total_loss + multi_weight * loss_multi
        details['loss_multi'] = loss_multi
        
    # 3. Boundary Refinement loss on positive queries
    if job.get('use_boundary_refine', False) and 'refined_spans' in output and positive.any():
        loss_refine = compute_boundary_refine_loss(output['refined_spans'], metadata, existence.device, job=job)
        refine_weight = float(job.get('refine_weight', 0.5))
        total_loss = total_loss + refine_weight * loss_refine
        details['loss_refine'] = loss_refine
        
    # 4. Hard negative contrastive loss
    if job.get('hard_negative', False) and model is not None and model_inputs is not None:
        batch_size = model_inputs['src_txt'].shape[0]
        if batch_size >= 2 and positive.any():
            # In-batch text embeddings
            pooled = (model_inputs['src_txt'] * model_inputs['src_txt_mask'].unsqueeze(-1)).sum(1)
            denom = model_inputs['src_txt_mask'].sum(1, keepdim=True).clamp(min=1)
            pooled = F.normalize(pooled / denom, dim=-1)
            
            sim_matrix = pooled @ pooled.t()
            sim_matrix.fill_diagonal_(-1e4)
            hard_neg_indices = sim_matrix.argmax(dim=1)
            
            neg_inputs = dict(model_inputs)
            neg_inputs['src_txt'] = model_inputs['src_txt'][hard_neg_indices]
            neg_inputs['src_txt_mask'] = model_inputs['src_txt_mask'][hard_neg_indices]
            
            device = model_inputs['src_txt'].device
            rng_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
            with torch.random.fork_rng(devices=rng_devices):
                neg_output = model(**neg_inputs, targets={})
                
            pos_logits = output['pred_exist_logits'][positive]
            neg_logits = neg_output['pred_exist_logits'][positive]
            
            loss_mode = job.get('contrast_loss_mode', 'margin')
            if loss_mode == 'margin':
                margin = float(job.get('contrast_margin', 2.0))
                loss_contrast = F.relu(margin - (pos_logits - neg_logits)).mean()
            elif loss_mode == 'info_nce':
                temp = float(job.get('contrast_temp', 0.5))
                pair_logits = torch.stack([pos_logits / temp, neg_logits / temp], dim=1)
                loss_contrast = F.cross_entropy(pair_logits, torch.zeros(len(pos_logits), dtype=torch.long, device=device))
            elif loss_mode == 'naive_bce':
                loss_contrast = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
            else:
                raise ValueError(f"Unknown contrast_loss_mode: {loss_mode}")
                
            contrast_weight = float(job.get('hard_negative_weight', 0.5))
            total_loss = total_loss + contrast_weight * loss_contrast
            details['loss_contrast'] = loss_contrast
            
    details['loss_total'] = total_loss
    return total_loss, details
