"""SI-Bind: Structurally Isolated Inter-Layer Binding and Slot-Private Refinement.

This module implements the lightweight interlayer adapter for E1-E8:
- E1: Pure temporal binding supervision (identity residual return).
- E2: Temporal binding + lightweight residual update (no RCG/basis/FRF).
- E3: E2 + Slot-Private Innovation (relative to detached mean temporal context).
- E4: E2 + Uniform slot refresh control.
- E5: E2 + Dominance-aware slot refresh (EMA of positive Top-1 slot index).
- E6: E2 + Innovation + Dominance refresh.
- E7: E6 with binding_loss_coef = 0.
- E8: Strict serial A -> B without C (Temporal binding + slot-private innovation, no refresh).
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class InterLayerBindOutput(NamedTuple):
    """Outputs and diagnostics from an InterLayerBind forward pass.

    All tensors follow batch-first format [batch, num_queries, ...] except
    adapted_state which matches decoder input layout [num_queries, batch, hidden_dim].
    """
    adapted_state: Tensor
    temporal_logits: Tensor
    temporal_attention: Tensor
    temporal_context: Tensor
    residual_update: Optional[Tensor]
    basis_weights: Optional[Tensor] = None
    actual_refresh_count: int = 0
    opportunity_count: int = 0


class InterLayerBind(nn.Module):
    """Lightweight Interlayer Binding Module with Slot-Private Innovation & Dominance Refresh."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 10,
        beta: float = 0.05,
        enable_residual: bool = True,
        use_innovation: bool = False,
        lambda_nov: float = 0.2,
        refresh_mode: str = "none",  # "none", "dominance", "uniform"
        p_max: float = 0.25,
        alpha: float = 0.25,
        warmup_epochs: int = 5,
        ema_decay: float = 0.95,
        frf_hidden_dim: Optional[int] = None,
        binding_loss_coef: float = 0.2,
        variant_name: str = "E2",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.enable_residual = bool(enable_residual)
        self.use_innovation = bool(use_innovation)
        self.lambda_nov = float(lambda_nov)
        self.refresh_mode = str(refresh_mode).lower()
        self.p_max = float(p_max)
        self.alpha = float(alpha)
        self.warmup_epochs = int(warmup_epochs)
        self.ema_decay = float(ema_decay)
        self.binding_loss_coef = float(binding_loss_coef)
        self.variant_name = str(variant_name).upper()

        if self.variant_name == "E8":
            self.refresh_mode = "none"
            self.enable_residual = True
            self.use_innovation = True

        if frf_hidden_dim is None:
            frf_hidden_dim = 2 * hidden_dim
        self.frf_hidden_dim = int(frf_hidden_dim)

        if self.refresh_mode not in ("none", "dominance", "uniform"):
            raise ValueError(f"Unknown refresh_mode: {refresh_mode}")

        # Beta as a buffer for visibility
        self.register_buffer("beta", torch.tensor(float(beta)))

        # Temporal binding components (A)
        self.decoder_norm = nn.LayerNorm(self.hidden_dim)
        self.candidate_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.semantic_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.memory_norm = nn.LayerNorm(self.hidden_dim)
        self.memory_key_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.memory_value_projection = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Lightweight residual mapping (Wr: [3d -> frf_dim], Wo: [frf_dim -> d])
        if self.enable_residual:
            self.residual_fc = nn.Linear(3 * self.hidden_dim, self.frf_hidden_dim)
            self.residual_out = nn.Linear(self.frf_hidden_dim, self.hidden_dim)
            self.residual_norm = nn.LayerNorm(self.hidden_dim)

        # Slot-Private Innovation (B)
        if self.use_innovation:
            self.innovation_projection = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Dominance-Aware Refresh components (C)
        if self.refresh_mode in ("dominance", "uniform"):
            self.refresh_fc = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
            self.refresh_norm = nn.LayerNorm(self.hidden_dim)

        # Registered buffers for training-time statistics & EMA
        self.register_buffer("top1_ema", torch.full((self.num_queries,), 1.0 / self.num_queries, dtype=torch.float32))
        self.register_buffer("current_epoch", torch.tensor(0, dtype=torch.long))
        self.register_buffer("actual_refresh_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("opportunity_count", torch.tensor(0, dtype=torch.long))

        # Runtime storage for diagnostics
        self.last_output: Optional[InterLayerBindOutput] = None

    def clear_diagnostics(self) -> None:
        self.last_output = None

    def update_top1_ema(self, top1_slots: Optional[Tensor], epoch: int) -> None:
        """Updates slot dominance EMA using positive training samples only.
        
        Must NOT be called during validation or testing.
        """
        if self.refresh_mode not in ("dominance", "uniform"):
            return
        self.current_epoch.fill_(epoch)
        if top1_slots is None or top1_slots.numel() == 0:
            return

        with torch.no_grad():
            counts = torch.bincount(top1_slots.detach().view(-1), minlength=self.num_queries).float()
            total = counts.sum()
            if total > 0:
                dist = counts / total
                # EMA update
                decay = self.ema_decay
                self.top1_ema.mul_(decay).add_(dist.to(self.top1_ema.device) * (1.0 - decay))
                self.top1_ema.div_(self.top1_ema.sum().clamp_min(1e-6))

    @staticmethod
    def _masked_temporal_softmax(logits: Tensor, padding_mask: Tensor) -> Tensor:
        """Softmax over valid frames, returning zeros for an all-padding row."""
        valid = ~padding_mask.bool()
        masked_logits = logits.masked_fill(~valid.unsqueeze(1), torch.finfo(logits.dtype).min)
        attention = torch.softmax(masked_logits, dim=-1)
        attention = attention * valid.unsqueeze(1).to(attention.dtype)
        denominator = attention.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(attention.dtype).eps)
        return attention / denominator

    def forward(
        self,
        decoder_state: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor,
        query_semantic: Tensor,
        video_length: int,
    ) -> Tensor:
        """Performs interlayer binding forward pass.

        Args:
            decoder_state: [num_queries, batch, hidden_dim]
            memory: [source_length, batch, hidden_dim]
            memory_key_padding_mask: [batch, source_length] (True = padding)
            query_semantic: [batch, hidden_dim]
            video_length: int number of video frames
        Returns:
            adapted_state: [num_queries, batch, hidden_dim]
        """
        num_queries, batch_size, d = decoder_state.shape
        candidate = decoder_state.transpose(0, 1)  # [B, K, d]
        video_memory = memory[:video_length].transpose(0, 1)  # [B, T, d]
        video_padding_mask = memory_key_padding_mask[:, :video_length].bool()

        # 1. Temporal attention (A)
        # q_i = Wq LN(h_i) + Ws s
        cand_norm = self.decoder_norm(candidate)
        q_cand = self.candidate_projection(cand_norm)  # [B, K, d]
        q_sem = self.semantic_projection(query_semantic).unsqueeze(1)  # [B, 1, d]
        temporal_query = q_cand + q_sem  # [B, K, d]

        # k_t = Wk LN(v_t)
        v_norm = self.memory_norm(video_memory)
        temporal_key = self.memory_key_projection(v_norm)  # [B, T, d]

        # a_it = masked_softmax(<q_i, k_t> / sqrt(d))
        temporal_logits = torch.einsum("bmd,btd->bmt", temporal_query, temporal_key) / math.sqrt(self.hidden_dim)
        temporal_attention = self._masked_temporal_softmax(temporal_logits, video_padding_mask)  # [B, K, T]

        # c_i = sum_t a_it Wv v_t
        video_value = self.memory_value_projection(video_memory)  # [B, T, d]
        temporal_context = torch.einsum("bmt,btd->bmd", temporal_attention, video_value)  # [B, K, d]

        # In E1 mode: residual is completely disabled; returns input state unmodified
        # But temporal_attention graph connects to candidate, memory, query_semantic
        if not self.enable_residual:
            self.last_output = InterLayerBindOutput(
                adapted_state=decoder_state,
                temporal_logits=temporal_logits,
                temporal_attention=temporal_attention,
                temporal_context=temporal_context,
                residual_update=None,
                basis_weights=None,
                actual_refresh_count=0,
                opportunity_count=0,
            )
            return decoder_state

        # E8: Strict serial A -> B without C
        if self.variant_name == "E8":
            sem_expand = query_semantic.unsqueeze(1).expand(-1, num_queries, -1)
            # Step 1: Mechanism A residual update
            # r = R([S ; c ; s])
            res_in = torch.cat([candidate, temporal_context, sem_expand], dim=-1)
            r_i = self.residual_norm(self.residual_out(F.gelu(self.residual_fc(res_in))))
            beta_val = self.beta.to(candidate.dtype)
            # Intermediate slot state H_A = S + beta * r
            h_A = candidate + beta_val * r_i

            # Step 2: Mechanism B recomputes temporal context using H_A
            # c_A = T(H_A, V, s)
            cand_norm_A = self.decoder_norm(h_A)
            q_cand_A = self.candidate_projection(cand_norm_A)
            temporal_query_A = q_cand_A + q_sem
            temporal_logits_A = torch.einsum("bmd,btd->bmt", temporal_query_A, temporal_key) / math.sqrt(self.hidden_dim)
            temporal_attention_A = self._masked_temporal_softmax(temporal_logits_A, video_padding_mask)
            temporal_context_A = torch.einsum("bmt,btd->bmd", temporal_attention_A, video_value)

            # Context difference innovation: n_A = W_nov(c_A - sg(mean_slot(c_A)))
            c_bar_A = temporal_context_A.mean(dim=1, keepdim=True).detach()
            n_A = self.innovation_projection(temporal_context_A - c_bar_A)

            # S_out = H_A + beta * lambda_nov * n_A
            adapted_candidate = h_A + beta_val * (self.lambda_nov * n_A)
            adapted_state = adapted_candidate.transpose(0, 1)  # [K, B, d]

            self.last_output = InterLayerBindOutput(
                adapted_state=adapted_state,
                temporal_logits=temporal_logits,
                temporal_attention=temporal_attention,
                temporal_context=temporal_context,
                residual_update=r_i,
                basis_weights=None,
                actual_refresh_count=0,
                opportunity_count=0,
            )
            return adapted_state

        # 2. Refresh Mechanism (C) - Training only, strictly disabled in eval
        h_current = candidate
        refresh_count = 0
        opportunity = 0
        if self.training and self.refresh_mode in ("dominance", "uniform") and self.current_epoch.item() >= self.warmup_epochs:
            excess = torch.clamp(self.top1_ema - 1.0 / self.num_queries, min=0.0)
            max_excess = excess.max()
            if max_excess > 1e-6:
                if self.refresh_mode == "dominance":
                    p_i = self.p_max * excess / (max_excess + 1e-7)
                else:  # uniform control
                    p_val = (self.p_max * excess / (max_excess + 1e-7)).mean()
                    p_i = torch.full_like(self.top1_ema, float(p_val))
            else:
                p_i = torch.zeros_like(self.top1_ema)

            # Sample Bernoulli mask z_bi ~ Bernoulli(p_i)
            # Shapes: p_i is [K], z is [B, K]
            p_expand = p_i.unsqueeze(0).expand(batch_size, -1)
            z = torch.bernoulli(p_expand)  # [B, K]
            refresh_count = int(z.sum().item())
            opportunity = int(z.numel())
            self.actual_refresh_count.add_(refresh_count)
            self.opportunity_count.add_(opportunity)

            if refresh_count > 0:
                # h_refresh_bi = (1 - alpha * z) * h + alpha * z * LN(Wm [c_bi; s_b])
                sem_expand = query_semantic.unsqueeze(1).expand(-1, num_queries, -1)
                cat_cs = torch.cat([temporal_context, sem_expand], dim=-1)
                refreshed_state = self.refresh_norm(self.refresh_fc(cat_cs))
                z_expand = z.unsqueeze(-1)  # [B, K, 1]
                h_current = (1.0 - self.alpha * z_expand) * candidate + (self.alpha * z_expand) * refreshed_state

        # 3. Residual update r_i (A)
        # r_i = Wo GELU( Wr [ h_i ; c_i ; s ] )
        sem_expand = query_semantic.unsqueeze(1).expand(-1, num_queries, -1)
        res_in = torch.cat([candidate, temporal_context, sem_expand], dim=-1)
        r_i = self.residual_norm(self.residual_out(F.gelu(self.residual_fc(res_in))))

        # 4. Slot-Private Innovation nov_i (B)
        if self.use_innovation:
            c_bar = temporal_context.mean(dim=1, keepdim=True).detach()
            nov_i = self.innovation_projection(temporal_context - c_bar)
            update = r_i + self.lambda_nov * nov_i
        else:
            update = r_i

        # h_i^(+) = h_current + beta * update
        beta_val = self.beta.to(candidate.dtype)
        adapted_candidate = h_current + beta_val * update
        adapted_state = adapted_candidate.transpose(0, 1)  # [K, B, d]

        self.last_output = InterLayerBindOutput(
            adapted_state=adapted_state,
            temporal_logits=temporal_logits,
            temporal_attention=temporal_attention,
            temporal_context=temporal_context,
            residual_update=r_i,
            basis_weights=None,
            actual_refresh_count=refresh_count,
            opportunity_count=opportunity,
        )
        return adapted_state


def build_interlayer_adapter(options, hidden_dim: int, num_queries: int) -> Optional[nn.Module]:
    """Factory creating either InterLayerBind, DETRQueryCGP, or None based on options."""
    adapter_type = getattr(options, "interlayer_adapter_type", None)

    # Legacy DQ-CGPv3 compatibility
    if adapter_type == "dq_cgp" or (adapter_type is None and getattr(options, "use_query_cgp", False)):
        from soccergmr.modules.dq_cgp import DETRQueryCGP
        return DETRQueryCGP(
            hidden_dim=hidden_dim,
            num_basis=int(getattr(options, "query_cgp_num_basis", 16)),
            prompt_length=int(getattr(options, "query_cgp_prompt_length", 6)),
            router_hidden_dim=int(getattr(options, "query_cgp_router_hidden_dim", hidden_dim)),
            frf_hidden_dim=int(getattr(options, "query_cgp_frf_hidden_dim", 2 * hidden_dim)),
            temperature=float(getattr(options, "query_cgp_temperature", 1.0)),
            beta=float(getattr(options, "query_cgp_beta", 0.05)),
        )

    if adapter_type == "si_bind":
        variant = str(getattr(options, "interlayer_variant", "E2")).upper()
        # Default flags per variant
        enable_residual = True
        use_innovation = False
        refresh_mode = "none"
        binding_loss_coef = float(getattr(options, "si_bind_loss_coef", 0.2))

        if variant == "E1":
            enable_residual = False
        elif variant == "E2":
            pass
        elif variant == "E3":
            use_innovation = True
        elif variant == "E4":
            refresh_mode = "uniform"
        elif variant == "E5":
            refresh_mode = "dominance"
        elif variant == "E6":
            use_innovation = True
            refresh_mode = "dominance"
        elif variant == "E7":
            use_innovation = True
            refresh_mode = "dominance"
            binding_loss_coef = 0.0
        elif variant == "E8":
            enable_residual = True
            use_innovation = True
            refresh_mode = "none"
            binding_loss_coef = float(getattr(options, "si_bind_loss_coef", 0.2))

        # Allow explicit overrides from options if specified
        if hasattr(options, "si_bind_enable_residual"):
            enable_residual = bool(options.si_bind_enable_residual)
        if hasattr(options, "si_bind_innovation"):
            use_innovation = bool(options.si_bind_innovation)
        if hasattr(options, "si_bind_refresh_mode"):
            refresh_mode = str(options.si_bind_refresh_mode)

        return InterLayerBind(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            beta=float(getattr(options, "si_bind_beta", 0.05)),
            enable_residual=enable_residual,
            use_innovation=use_innovation,
            lambda_nov=float(getattr(options, "si_bind_lambda_nov", 0.2)),
            refresh_mode=refresh_mode,
            p_max=float(getattr(options, "si_bind_p_max", 0.25)),
            alpha=float(getattr(options, "si_bind_alpha", 0.25)),
            warmup_epochs=int(getattr(options, "si_bind_warmup", 5)),
            ema_decay=float(getattr(options, "si_bind_ema_decay", 0.95)),
            frf_hidden_dim=int(getattr(options, "si_bind_frf_hidden_dim", 2 * hidden_dim)),
            binding_loss_coef=binding_loss_coef,
            variant_name=variant,
        )

    return None
