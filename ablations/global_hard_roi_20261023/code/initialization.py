"""Verified E0-to-LoRA key mapping for future matched training experiments.

Does not modify existing checkpoints or shared training code.
"""
import json
from pathlib import Path
import sys


def load_e0_weights(model, baseline_state):
    """Load every shared backbone tensor, retaining new verifier/adapter tensors."""
    state = model.state_dict()
    loaded = []
    retained = []
    for key in state:
        if '.exist_head.' in key or '.refiner.' in key or '.lora_' in key:
            retained.append(key)
            continue
        source = key.replace('.base_linear.', '.')
        if source not in baseline_state:
            raise KeyError(f'Missing backbone initializer: {key} <- {source}')
        if state[key].shape != baseline_state[source].shape:
            raise ValueError(f'Backbone shape mismatch: {key}')
        state[key] = baseline_state[source]
        loaded.append((key, source))
    model.load_state_dict(state, strict=True)
    for key, source in loaded:
        assert model.state_dict()[key].detach().cpu().equal(baseline_state[source].detach().cpu()), key
    return {'verified_shared_tensors': len(loaded), 'retained_new_tensors': len(retained),
            'remapped_LoRA_base_tensors': sum('.base_linear.' in key for key, _ in loaded)}


