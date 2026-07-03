# PAPER: Results 1-6 restoration driver (CKA restore; tab:cap, tab:ablation, tab:models). See RESULTS_TO_CODE.md
"""
Restoration experiment driver.

Pipeline (paper validation):
  teacher T  = Qwen2.5-0.5B (frozen)
  student S  = corrupt_model(T, strength s)        # damage middle layers
  Stage 1    = train S with basis-invariant loss (rel|cka) on text batches
               to match T's per-layer Gram structure (NO label supervision)
  measure    = CKA(S, T) per layer, on in-domain and out-domain probes
  control    = repeat with kind="abs" (ill-posed) -> expected to fail

This is the minimal test of: "basis-invariant distillation substantially
restores pretraining; SFT then restores T near the SFT subspace."

Run (single config):
  python -u restore.py --strength 0.5 --kind cka --steps 300

Live per-step loss + periodic CKA, flushed.
"""
import argparse, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from corrupt import corrupt_model
from losses import multilayer_loss
# reuse numpy CKA for measurement (detached) from the invariants module
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from invariants import linear_cka
except Exception:
    # local minimal copy if invariants.py not alongside
    def linear_cka(A, B):
        def c(H): return H - H.mean(0, keepdims=True)
        Ga, Gb = c(A) @ c(A).T, c(B) @ c(B).T
        num = (Ga * Gb).sum()
        den = np.sqrt((Ga*Ga).sum() * (Gb*Gb).sum())
        return float(num/den) if den else 0.0


def pooled_hidden(model, tok, texts, device, layers_keep=None):
    """Return list over layers of (N, d) numpy arrays (last-token pooled)."""
    model.eval()
    per_layer = None
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt").to(device)
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states  # tuple length L+1
            if per_layer is None:
                per_layer = [[] for _ in hs]
            for i, h in enumerate(hs):
                per_layer[i].append(h[0, -1].float().cpu().numpy())
    return [np.stack(col, 0) for col in per_layer]


def measure_cka(student, teacher, tok, probes, device, tag=""):
    Hs = pooled_hidden(student, tok, probes, device)
    Ht = pooled_hidden(teacher, tok, probes, device)
    ckas = [linear_cka(a, b) for a, b in zip(Hs, Ht)]
    mean = float(np.mean(ckas[1:]))  # skip embedding layer 0
    print(f"    [cka {tag}] per-layer mean={mean:.4f} "
          f"min={min(ckas[1:]):.3f}@L{int(np.argmin(ckas[1:]))+1} "
          f"last={ckas[-1]:.3f}", flush=True)
    return mean, ckas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--mode", default="reinit", choices=["reinit", "noise"])
    ap.add_argument("--kind", default="cka", choices=["rel", "cka", "abs"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[load] teacher {args.model}", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, output_hidden_states=True
    ).to(dev)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print(f"[corrupt] strength={args.strength} mode={args.mode}", flush=True)
    student = corrupt_model(teacher, strength=args.strength, mode=args.mode)
    student = student.to(dev)
    student.train()
    for p in student.parameters():
        p.requires_grad_(True)

    # ---- probe sets: in-domain vs out-domain (Experiment 2) ----
    in_domain = [
        "The capital of France is", "Water boils at a temperature of",
        "The chemical symbol for gold is", "A triangle has this many sides:",
        "The largest planet in the solar system is",
        "Photosynthesis occurs in the", "The speed of light is approximately",
        "DNA stands for", "The author of Romeo and Juliet is",
        "The square root of 144 is",
    ]
    out_domain = [
        "삼성전자의 2023년 매출은", "방배5구역 재개발 조합원의 권리는",
        "금융감독원의 역할은", "도시정비법상 관리처분계획이란",
        "벡터공간의 기저란", "고유값과 고유벡터의 정의는",
    ]

    # ---- training corpus (generic text); reuse probes + simple sentences ----
    corpus = in_domain * 6 + [
        "Machine learning models are trained on data.",
        "The history of Rome spans many centuries.",
        "Photosynthesis converts sunlight into energy.",
        "Economic policy affects inflation and growth.",
        "Neural networks consist of layers of neurons.",
        "The ocean covers most of the planet's surface.",
    ] * 4

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)

    print(f"[restore] kind={args.kind} steps={args.steps}", flush=True)
    print("  --- baseline (corrupted, before restore) ---", flush=True)
    measure_cka(student, teacher, tok, in_domain, dev, tag="in ")
    measure_cka(student, teacher, tok, out_domain, dev, tag="out")

    rng = np.random.default_rng(0)
    for step in range(1, args.steps + 1):
        batch = [corpus[i] for i in rng.integers(0, len(corpus), args.bs)]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=32).to(dev)
        with torch.no_grad():
            t_out = teacher(**enc, output_hidden_states=True)
        s_out = student(**enc, output_hidden_states=True)
        # pool last non-pad token per sequence
        mask = enc["attention_mask"]
        last_idx = mask.sum(1) - 1
        def pool(hs_tuple):
            pooled = []
            for h in hs_tuple:
                pooled.append(h[torch.arange(h.size(0)), last_idx])
            return pooled
        Hs = pool(s_out.hidden_states)
        Ht = [h.detach() for h in pool(t_out.hidden_states)]
        loss = multilayer_loss(Hs, Ht, kind=args.kind)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        if step % 25 == 0 or step == 1:
            print(f"  step {step:4d}/{args.steps} loss={loss.item():.4f}",
                  flush=True)
        if step % 100 == 0:
            measure_cka(student, teacher, tok, in_domain, dev, tag="in ")
            measure_cka(student, teacher, tok, out_domain, dev, tag="out")
            student.train()

    print("\n[restore] final", flush=True)
    mi, _ = measure_cka(student, teacher, tok, in_domain, dev, tag="in ")
    mo, _ = measure_cka(student, teacher, tok, out_domain, dev, tag="out")
    print(f"\n=== strength={args.strength} kind={args.kind}: "
          f"in-domain CKA={mi:.4f}  out-domain CKA={mo:.4f}  "
          f"gap={mi-mo:+.4f} ===", flush=True)


if __name__ == "__main__":
    main()
