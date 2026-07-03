# PAPER: Corruption utility for the restoration experiments (Results 1-6). See RESULTS_TO_CODE.md
"""
Corruption utility for the restoration experiment.

Start from the teacher weights and damage a fraction of the MIDDLE decoder
layers (never the tied embedding/head, since tie_word_embeddings=True on
Qwen2.5-0.5B means corrupting embeddings also breaks the output projection).

Two corruption modes:
  - "reinit" : replace selected layers' weights with fresh init (full damage
               of those layers; basis there is destroyed).
  - "noise"  : add Gaussian noise scaled by each tensor's std (graded damage).

strength s in [0,1]:
  - reinit: fraction of middle layers reinitialized (rounded).
  - noise : relative noise magnitude (std multiplier).

Returns the corrupted student model (a separate instance from teacher).
"""
import copy
import torch
import torch.nn as nn


def _decoder_layers(model):
    # Qwen2.5: model.model.layers
    return model.model.layers


def corrupt_model(teacher, strength=0.5, mode="reinit", protect_first=2,
                  protect_last=2, seed=0, verbose=True):
    """
    teacher: a loaded HF model (will NOT be modified; we deep-copy).
    Returns a corrupted copy.
    """
    g = torch.Generator().manual_seed(seed)
    student = copy.deepcopy(teacher)
    layers = _decoder_layers(student)
    L = len(layers)
    # candidate middle layers (protect a few at each end)
    cand = list(range(protect_first, L - protect_last))

    if mode == "reinit":
        k = int(round(strength * len(cand)))
        # pick k middle layers spread across the band
        if k <= 0:
            chosen = []
        else:
            idx = torch.linspace(0, len(cand) - 1, k).round().long().tolist()
            chosen = sorted(set(cand[i] for i in idx))
        for li in chosen:
            for name, p in layers[li].named_parameters():
                with torch.no_grad():
                    if p.dim() >= 2:
                        # fresh Kaiming-ish init for matrices
                        std = (2.0 / p.shape[-1]) ** 0.5
                        p.copy_(torch.randn(p.shape, generator=g) * std)
                    else:
                        p.zero_()
        if verbose:
            print(f"[corrupt] reinit layers {chosen} "
                  f"({len(chosen)}/{L})", flush=True)

    elif mode == "noise":
        chosen = cand
        for li in chosen:
            for name, p in layers[li].named_parameters():
                with torch.no_grad():
                    sd = p.detach().float().std().item()
                    noise = torch.randn(p.shape, generator=g) * (strength * sd)
                    p.add_(noise.to(p.dtype))
        if verbose:
            print(f"[corrupt] noise std*{strength} on layers "
                  f"{chosen[0]}..{chosen[-1]}", flush=True)
    else:
        raise ValueError(mode)

    return student
