# PAPER: Section 5 (Predictions 5.1-5.3) -- graft-success from subspace overlap. See RESULTS_TO_CODE.md
"""
Graft-success prediction experiment (paper Section 5 validation).

Question: does the subspace overlap at a transplant boundary predict whether
the graft preserves function?

Controlled design on Qwen2.5-0.5B:
  host  H = the intact teacher T (frozen reference).
  donor variants = T perturbed to varying degrees (different layers perturbed,
                   different noise levels) -> a spectrum of boundary overlaps.
  graft : copy donor's layer-block [a..b] into a fresh copy of the host.
  overlap : CKA( donor boundary repr , host boundary repr ) at the splice
            layer, on a probe set. Also Procrustes residual.
  success : how well the grafted model preserves the host's function, measured
            as PPL_graft / PPL_host (close to 1 = graft preserved function)
            and top-1 next-token agreement with the host.

We sweep (which layers transplanted, donor perturbation strength) to spread
overlap across [low, high], then regress success on overlap.

This is eval-only: no training. Each graft is a weight copy + a forward pass.
Live per-graft output (flush).

Run: python -u graft_experiment.py --n_donors 8 --seeds 3
"""
import argparse, copy, math, json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from invariants import linear_cka, procrustes_residual, cka_null


# ---------------------------------------------------------------------------
def perturb_layers(model, layer_ids, strength, seed):
    """Add scaled Gaussian noise to the given layers' weights -> a donor
    whose representation at those layers differs from the host by a controlled
    amount. Returns a new model (host unchanged)."""
    g = torch.Generator().manual_seed(seed)
    donor = copy.deepcopy(model)
    layers = donor.model.layers
    with torch.no_grad():
        for li in layer_ids:
            for p in layers[li].parameters():
                sd = p.detach().float().std().item()
                noise = torch.randn(p.shape, generator=g) * (strength * sd)
                p.add_(noise.to(p.dtype))
    return donor


def transplant(host, donor, layer_ids):
    """Return a copy of host with donor's `layer_ids` weights spliced in."""
    g = copy.deepcopy(host)
    gl = g.model.layers
    dl = donor.model.layers
    with torch.no_grad():
        for li in layer_ids:
            gl[li].load_state_dict(dl[li].state_dict())
    return g


@torch.no_grad()
def boundary_repr(model, tok, texts, layer, device):
    """Last-token pooled hidden state at the output of `layer` (residual
    stream), shape (N, d)."""
    model.eval()
    vecs = []
    for t in texts:
        enc = tok(t, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True)
        vecs.append(out.hidden_states[layer + 1][0, -1].float().cpu().numpy())
    return np.stack(vecs, 0)


@torch.no_grad()
def function_metrics(model, host, tok, texts, device, max_len=48):
    """PPL ratio (model vs host) and top-1 agreement with host."""
    model.eval(); host.eval()
    nll_m = nll_h = ntok = 0
    agree = pos = 0
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True,
                  max_length=max_len).to(device)
        ids = enc["input_ids"]
        if ids.size(1) < 2:
            continue
        lm = model(**enc).logits[0]
        lh = host(**enc).logits[0]
        tgt = ids[0, 1:]
        nll_m += F.nll_loss(F.log_softmax(lm[:-1], -1), tgt,
                            reduction="sum").item()
        nll_h += F.nll_loss(F.log_softmax(lh[:-1], -1), tgt,
                            reduction="sum").item()
        ntok += tgt.numel()
        agree += (lm[:-1].argmax(-1) == lh[:-1].argmax(-1)).sum().item()
        pos += tgt.numel()
    ppl_m = math.exp(nll_m / max(ntok, 1))
    ppl_h = math.exp(nll_h / max(ntok, 1))
    return ppl_m / max(ppl_h, 1e-9), agree / max(pos, 1)


# ---------------------------------------------------------------------------
def logreg(X, y, iters=50, l2=1e-3):
    X = np.asarray(X, float); X = np.hstack([np.ones((len(X), 1)), X])
    y = np.asarray(y, float); w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        W = p * (1 - p) + 1e-9
        g = X.T @ (p - y) + l2 * w
        H = X.T @ (X * W[:, None]) + l2 * np.eye(X.shape[1])
        w -= np.linalg.solve(H, g)
    return w


def auc(s, y):
    s = np.asarray(s); y = np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(1, len(s)+1)
    return float((r[y == 1].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--success_ppl", type=float, default=2.0,
                    help="graft is 'success' if PPL ratio < this threshold")
    ap.add_argument("--out", default="graft_results.json")
    args = ap.parse_args()
    dev = args.device

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    host = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        output_hidden_states=True).to(dev).eval()
    L = len(host.model.layers)
    print(f"[graft] host loaded, {L} layers", flush=True)

    probes = [
        "The capital of France is", "Water boils at",
        "The chemical symbol for gold is", "A triangle has",
        "The largest planet is", "Photosynthesis occurs in",
        "The speed of light is", "DNA stands for",
        "Machine learning models are", "The history of Rome",
        "Economic policy affects", "Neural networks consist of",
    ]

    # sweep: which contiguous block to transplant x donor perturbation strength
    blocks = [[i] for i in range(2, L - 2)]            # single-layer grafts
    blocks += [[i, i + 1] for i in range(2, L - 3, 3)]  # some 2-layer blocks
    strengths = [0.05, 0.1, 0.2, 0.4, 0.8]

    rows = []
    gid = 0
    for blk in blocks:
        for s in strengths:
            gid += 1
            donor = perturb_layers(host, blk, s, seed=gid)
            grafted = transplant(host, donor, blk)
            # boundary = output of the last transplanted layer
            bl = blk[-1]
            Hd = boundary_repr(grafted, tok, probes, bl, dev)
            Hh = boundary_repr(host, tok, probes, bl, dev)
            cka = linear_cka(Hd, Hh)
            _, rho, _ = procrustes_residual(Hd, Hh)
            ppl_ratio, top1 = function_metrics(grafted, host, tok, probes, dev)
            success = int(ppl_ratio < args.success_ppl)
            rows.append({"gid": gid, "block": blk, "strength": s,
                         "cka": cka, "rho": rho,
                         "ppl_ratio": ppl_ratio, "top1": top1,
                         "success": success})
            print(f"[graft {gid:3d}] blk={blk} str={s:.2f} "
                  f"CKA={cka:.3f} rho={rho:.3f} "
                  f"PPLratio={ppl_ratio:6.2f} top1={top1:.3f} "
                  f"-> {'OK' if success else 'FAIL'}", flush=True)
            del donor, grafted
            torch.cuda.empty_cache()

    # ---- analysis: does overlap predict success? ----
    cka = np.array([r["cka"] for r in rows])
    rho = np.array([r["rho"] for r in rows])
    y = np.array([r["success"] for r in rows])
    ppl = np.array([r["ppl_ratio"] for r in rows])

    print("\n=== Prediction 1: success vs boundary overlap ===", flush=True)
    print(f"  n={len(rows)}  success_rate={y.mean():.2f}  "
          f"CKA range [{cka.min():.3f}, {cka.max():.3f}]", flush=True)
    if 0 < y.mean() < 1:
        w = logreg(cka[:, None], y)
        print(f"  CKA -> success : logit coef {w[1]:+.2f} (expect >0), "
              f"AUC={auc(cka, y):.3f}", flush=True)
        wr = logreg(rho[:, None], y)
        print(f"  rho -> success : logit coef {wr[1]:+.2f} (expect <0), "
              f"AUC(-rho)={auc(-rho, y):.3f}", flush=True)
    # continuous: correlation of overlap with -log(ppl_ratio)
    quality = -np.log(ppl)
    r_cka = np.corrcoef(cka, quality)[0, 1]
    r_rho = np.corrcoef(rho, quality)[0, 1]
    print(f"  corr(CKA, -log PPLratio) = {r_cka:+.3f} (expect >0)", flush=True)
    print(f"  corr(rho, -log PPLratio) = {r_rho:+.3f} (expect <0)", flush=True)

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[graft] wrote {len(rows)} grafts -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
