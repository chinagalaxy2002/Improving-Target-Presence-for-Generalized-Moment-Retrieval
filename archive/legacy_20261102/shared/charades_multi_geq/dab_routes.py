"""Training-only Temporal O2M routes shared by three DAB-style backbones.

The base model and its inference output are untouched.  During training a decoder
pre-hook captures the encoded memory used by the native route, then the same
decoder is called with 24 auxiliary queries.  Route switches are implemented by
small proxies so SA/CA stay shared, while Separate-FFN owns only linear1,
linear2, and norm3.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn

VARIANTS = ("baseline", "shared", "sepffn", "instruction", "multi", "adaptive")


def _routes(variant: str):
    if variant == "baseline":
        return []
    if variant in ("shared", "sepffn", "instruction"):
        return [variant]
    return ["sepffn", "instruction"]


class _Choice(nn.Module):
    """Use a private FFN module only while the sepffn route is active."""

    def __init__(self, owner: "MultiRouteModel", shared: nn.Module):
        super().__init__()
        self.owner = [owner]  # avoid registering a circular nn.Module reference
        self.shared = shared
        self.private = copy.deepcopy(shared)

    def forward(self, *args, **kwargs):
        module = self.private if self.owner[0]._active_route == "sepffn" else self.shared
        return module(*args, **kwargs)


class _InstructionAttention(nn.Module):
    """Prepend four learned projected Q/K/V tokens, then discard their output."""

    def __init__(self, owner: "MultiRouteModel", shared: nn.Module):
        super().__init__()
        self.owner = [owner]
        self.shared = shared

    def forward(self, query, key, value, **kwargs):
        instruction = self.owner[0].instructions
        if self.owner[0]._active_route != "instruction" or instruction is None:
            return self.shared(query, key, value, **kwargs)
        count, batch, dim = instruction.shape[0], query.shape[1], query.shape[2]
        token = instruction[:, None, :].expand(count, batch, dim)
        q = torch.cat((token, query), dim=0)
        k = torch.cat((token, key), dim=0)
        v = torch.cat((token, value), dim=0)
        key_padding_mask = kwargs.get("key_padding_mask")
        if key_padding_mask is not None:
            pad = torch.zeros((batch, count), dtype=key_padding_mask.dtype, device=key_padding_mask.device)
            kwargs["key_padding_mask"] = torch.cat((pad, key_padding_mask), dim=1)
        attn_mask = kwargs.get("attn_mask")
        if attn_mask is not None:
            raise ValueError("Instruction route currently requires unmasked decoder self-attention")
        result = self.shared(q, k, v, **kwargs)
        output = result[0][-query.shape[0] :]
        return (output, *result[1:])


class _SlotAttentionProxy(nn.Module):
    """EaTR Slot Attention with route-specific 24 learned slots."""

    def __init__(self, owner: "MultiRouteModel", shared: nn.Module, routes: Iterable[str], dim: int):
        super().__init__()
        self.owner = [owner]
        self.shared = shared
        self.aux_slots = nn.ParameterDict({r: nn.Parameter(torch.empty(24, dim)) for r in routes})
        for parameter in self.aux_slots.values():
            nn.init.xavier_normal_(parameter)

    def forward(self, inputs, mask):
        route = self.owner[0]._active_route
        if route not in self.aux_slots:
            return self.shared(inputs, mask)
        module = self.shared
        batch = inputs.shape[0]
        normalized = module.norm_inputs(inputs)
        key, value = module.project_k(normalized), module.project_v(normalized)
        slots = self.aux_slots[route].unsqueeze(0).expand(batch, -1, -1)
        for _ in range(module.num_iterations):
            previous = slots
            query = module.project_q(module.norm_slots(slots))
            dots = torch.einsum("bid,bjd->bij", query, key) * (module.d_model ** -0.5)
            dots.masked_fill_(mask.unsqueeze(1), -torch.finfo(dots.dtype).max)
            attention = module.attn_holder(dots.softmax(dim=1))
            attention = attention / (attention.sum(dim=-1, keepdim=True) + module.epsilon)
            updates = torch.einsum("bjd,bij->bid", value, attention)
            slots = previous + updates
            slots = slots + module.mlp(module.norm_mlp(slots))
        return module.norm_out(slots)


class MultiRouteModel(nn.Module):
    """Wrap a native GMR backbone with training-only auxiliary routes."""

    def __init__(self, base: nn.Module, variant: str, backbone: str, hidden_dim: int = 256):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.base = base
        self.variant = variant
        self.backbone = backbone
        self.routes = _routes(variant)
        self._active_route: Optional[str] = None
        self._decoder_call = None
        self.instructions = nn.Parameter(torch.randn(4, hidden_dim) * 0.02) if "instruction" in self.routes else None
        self.aux_refs = nn.ParameterDict()

        decoder = base.transformer.decoder
        if backbone == "eatr":
            decoder.slot_atten = _SlotAttentionProxy(self, decoder.slot_atten, self.routes, hidden_dim)
        else:
            for route in self.routes:
                parameter = nn.Parameter(torch.empty(24, 2))
                nn.init.uniform_(parameter, 0.0, 1.0)
                self.aux_refs[route] = parameter

        for layer in decoder.layers:
            if "sepffn" in self.routes:
                layer.linear1 = _Choice(self, layer.linear1)
                layer.linear2 = _Choice(self, layer.linear2)
                layer.norm3 = _Choice(self, layer.norm3)
            if "instruction" in self.routes and hasattr(layer, "self_attn"):
                layer.self_attn = _InstructionAttention(self, layer.self_attn)

        decoder.register_forward_pre_hook(self._capture_decoder, with_kwargs=True)

    def _capture_decoder(self, module, args, kwargs):
        if self._active_route is None:
            self._decoder_call = (args, dict(kwargs))

    @staticmethod
    def _inverse_sigmoid(value, eps=1e-3):
        value = value.clamp(0, 1)
        return torch.log(value.clamp(min=eps) / (1 - value).clamp(min=eps))

    def _predict_dab(self, states, references):
        logits = self.base.class_embed(states)
        spans = self.base.span_embed(states)
        spans[..., :2] += self._inverse_sigmoid(references)
        spans = spans.sigmoid()
        predictions = [{"pred_logits": a, "pred_spans": b} for a, b in zip(logits, spans)]
        return dict(predictions[-1], aux_outputs=predictions[:-1])

    def _predict_eatr(self, states, references):
        layers = self.base.num_dec_layers
        moment_states = states[-layers:]
        logits = self.base.class_embed(moment_states)
        spans = self.base.moment_span_embed(moment_states)
        spans[..., :2] += self._inverse_sigmoid(references[-layers:])
        spans = spans.sigmoid()
        predictions = [{"pred_logits": a, "pred_spans": b} for a, b in zip(logits, spans)]
        return dict(predictions[-1], aux_outputs=predictions[:-1])

    def _run_route(self, route):
        decoder = self.base.transformer.decoder
        args, kwargs = self._decoder_call
        self._active_route = route
        try:
            if self.backbone == "eatr":
                states, references = decoder(*args, **kwargs)
                return self._predict_eatr(states, references)
            args = list(args)
            native_tgt = args[0]
            batch = native_tgt.shape[1]
            reference = self.aux_refs[route][:, None, :].expand(-1, batch, -1)
            args[0] = native_tgt.new_zeros((24, batch, native_tgt.shape[-1]))
            kwargs["refpoints_unsigmoid"] = reference
            states, references = decoder(*args, **kwargs)
            return self._predict_dab(states, references)
        finally:
            self._active_route = None

    def forward(self, *args, **kwargs):
        output = self.base(*args, **kwargs)
        if self.training and self.routes:
            if self._decoder_call is None:
                raise RuntimeError("Native decoder call was not captured")
            output["o2m_outputs"] = {route: self._run_route(route) for route in self.routes}
        return output

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)


def _span_xx(spans):
    center, width = spans.unbind(-1)
    return torch.stack((center - width / 2, center + width / 2), dim=-1)


def _generalized_iou(pred, target):
    left = torch.maximum(pred[:, 0], target[:, 0])
    right = torch.minimum(pred[:, 1], target[:, 1])
    intersection = (right - left).clamp(min=0)
    union = (pred[:, 1] - pred[:, 0]) + (target[:, 1] - target[:, 0]) - intersection
    enclosing = torch.maximum(pred[:, 1], target[:, 1]) - torch.minimum(pred[:, 0], target[:, 0])
    return intersection / union.clamp(min=1e-8) - (enclosing - union) / enclosing.clamp(min=1e-8)


class MultiRouteCriterion(nn.Module):
    """Native main loss plus replicated/adaptive Temporal O2M loss."""

    def __init__(self, native: nn.Module, matcher_owner: nn.Module, variant: str):
        super().__init__()
        self.native = native
        self.matcher_owner = matcher_owner
        self.variant = variant
        self.routes = _routes(variant)
        self.weight_dict = dict(native.weight_dict)
        route_weight = 0.3 / max(1, len(self.routes))
        layer_count = 3 if variant != "baseline" else 0
        for route in self.routes:
            for depth in range(layer_count):
                suffix = "" if depth == layer_count - 1 else f"_{depth}"
                for key in ("loss_span", "loss_giou", "loss_label"):
                    if key in matcher_owner.weight_dict:
                        self.weight_dict[f"{route}_{key}{suffix}"] = matcher_owner.weight_dict[key] * route_weight
        self.last_match_stats: Dict[str, float] = {}

    def _counts(self, prediction, spans):
        if not len(spans):
            return torch.empty(0, dtype=torch.long, device=spans.device)
        counts = torch.full((len(spans),), 2, dtype=torch.long, device=spans.device)
        if self.variant == "adaptive":
            pred = _span_xx(prediction.detach())
            gt = _span_xx(spans)
            inter = (torch.minimum(pred[:, None, 1], gt[None, :, 1]) - torch.maximum(pred[:, None, 0], gt[None, :, 0])).clamp(min=0)
            union = (pred[:, 1] - pred[:, 0])[:, None] + (gt[:, 1] - gt[:, 0])[None, :] - inter
            second = (inter / union.clamp(min=1e-8)).topk(2, dim=0).values[1]
            counts = 1 + (second >= 0.25).long()
        return counts

    def _loss(self, prediction, targets, indices):
        logits = prediction["pred_logits"]
        foreground = getattr(self.matcher_owner, "foreground_label", 0)
        background = getattr(self.matcher_owner, "background_label", 1)
        classes = torch.full(logits.shape[:2], background, dtype=torch.long, device=logits.device)
        batch_indices, source_indices = [], []
        target_spans = []
        for batch, (source, target) in enumerate(indices):
            if len(source):
                classes[batch, source] = foreground
                batch_indices.append(torch.full_like(source, batch))
                source_indices.append(source)
                target_spans.append(targets["span_labels"][batch]["spans"][target])
        empty_weight = getattr(self.matcher_owner, "empty_weight", None)
        label = F.cross_entropy(logits.transpose(1, 2), classes, weight=empty_weight)
        if not source_indices:
            zero = prediction["pred_spans"].sum() * 0
            return {"loss_label": label, "loss_span": zero, "loss_giou": zero}
        batch_idx = torch.cat(batch_indices)
        source_idx = torch.cat(source_indices)
        predicted = prediction["pred_spans"][batch_idx, source_idx]
        target = torch.cat(target_spans)
        span = F.l1_loss(predicted, target)
        giou = (1 - _generalized_iou(_span_xx(predicted), _span_xx(target))).mean()
        return {"loss_label": label, "loss_span": span, "loss_giou": giou}

    def forward(self, outputs, targets):
        main = {key: value for key, value in outputs.items() if key != "o2m_outputs"}
        losses = self.native(main, targets)
        total_matches = total_gt = 0
        for route, route_output in outputs.get("o2m_outputs", {}).items():
            expanded = []
            for batch, target in enumerate(targets["span_labels"]):
                counts = self._counts(route_output["pred_spans"][batch], target["spans"])
                expanded.append({"spans": target["spans"].repeat_interleave(counts, dim=0)})
                total_matches += int(counts.sum())
                total_gt += len(counts)
            route_targets = {"span_labels": expanded}
            predictions = route_output.get("aux_outputs", []) + [
                {key: route_output[key] for key in ("pred_logits", "pred_spans")}
            ]
            for depth, prediction in enumerate(predictions):
                indices = self.matcher_owner.matcher(prediction, route_targets)
                suffix = "" if depth == len(predictions) - 1 else f"_{depth}"
                for key, value in self._loss(prediction, route_targets, indices).items():
                    losses[f"{route}_{key}{suffix}"] = value
        self.last_match_stats = {
            "matches": total_matches,
            "gt": total_gt,
            "matches_per_gt": total_matches / max(1, total_gt),
        }
        return losses
