# PAPER: Prop 2.1 (prop:illposed), Table 1 (tab:rotation) -- training-level. See RESULTS_TO_CODE.md
"""
Training-level proof of Proposition 2.1.

The loss-level experiment (rotation_proof.py) shows that a function-preserving
rotation inflates the Feature-KD loss while leaving CKA / L_rel at zero. This
script takes the next step: it lets each objective *train* and shows the
downstream consequence on the network FUNCTION.

Setup.
  - Teacher T: frozen pretrained model. Its layer-l hidden is H_T (the target).
  - Student S: a copy of T into whose layer l we inject a function-preserving
    orthogonal reparametrization. Concretely we right-multiply the output side
    of block l by Q and left-multiply the input side of block l+1 by Q^T, so the
    composed function is unchanged at init (S and T are the same function) but
    S's layer-l hidden is H_T Q -- a different representative of the same class.

  Only block l's parameters are trainable; everything else (including block
  l+1's compensating Q^T and the head) is frozen. So the ONLY way to reduce a
  representational mismatch is to move block l.

Two objectives, trained separately from the same rotated init:
  (A) Feature-KD : min || H_S - H_T ||_F^2     (absolute-coordinate matching)
  (B) CKA-KD     : min  1 - CKA(H_S, H_T)       (basis-invariant)

What we measure, before and after training:
  - representational loss of each kind
  - the FUNCTION: full-sequence KL(T || S) and perplexity ratio on held-out text

Prediction (Prop 2.1, as a training fact):
  (A) Feature-KD drives H_S toward H_T's *coordinates*. Because block l+1 still
      carries the frozen Q^T, undoing the rotation in block l BREAKS the
      compensation, so the function degrades: KL(T||S) and PPL ratio rise.
  (B) CKA-KD sees ~zero gradient (the rotated student is already CKA-optimal),
      so block l is left essentially untouched and the function is preserved:
      KL(T||S) ~ 0, PPL ratio ~ 1.

Thus Feature-KD "succeeds" representationally while failing functionally;
CKA-KD leaves the function intact. This is Prop 2.1 shown through learning.

Run:
  python -u rotation_train.py --model Qwen/Qwen2.5-0.5B --device cuda:0 \
      --layer 12 --steps 300
"""
import argparse, copy, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def random_orthogonal(d, seed, device):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g)
    Q, R = torch.linalg.qr(A)
    Q = Q @ torch.diag(torch.sign(torch.diag(R)))
    return Q.to(device)


def linear_cka(A, B):
    A = A - A.mean(0, keepdim=True); B = B - B.mean(0, keepdim=True)
    ga, gb = A @ A.t(), B @ B.t()
    return (ga * gb).sum() / (ga.norm() * gb.norm() + 1e-9)


def feature_loss(A, B):
    return ((A - B) ** 2).sum() / A.shape[0]


class RotatedBlockWrapper(nn.Module):
    """Wrap decoder block l so its hidden output is right-multiplied by Q.
    Wrap block l+1 so its hidden input is left-multiplied by Q^T (frozen),
    exactly compensating, so the composed function is unchanged at init."""
    def __init__(self, block, Q, on_output=True):
        super().__init__()
        self.block = block
        self.register_buffer("Q", Q)
        self.on_output = on_output

    def forward(self, hidden_states, *args, **kwargs):
        if self.on_output:
            out = self.block(hidden_states, *args, **kwargs)
            if isinstance(out, tuple):
                h = out[0] @ self.Q
                return (h,) + out[1:]
            return out @ self.Q
        else:
            hidden_states = hidden_states @ self.Q  # Q^T passed in
            return self.block(hidden_states, *args, **kwargs)


@torch.no_grad()
def hidden_at(model, tok, texts, layer, device):
    model.eval()
    vs = []
    for t in texts:
        enc = tok(t, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True)
        vs.append(out.hidden_states[layer][0])
    return torch.cat(vs, 0)


@torch.no_grad()
def function_metrics(student, teacher, tok, texts, device):
    """full-seq KL(T||S) and PPL ratio on held-out text."""
    student.eval(); teacher.eval()
    kl_sum = ntok = 0
    nll_s = nll_t = 0
    for t in texts:
        enc = tok(t, return_tensors="pt").to(device)
        ls = student(**enc).logits[0]
        lt = teacher(**enc).logits[0]
        pt = F.softmax(lt, -1)
        kl = (pt * (F.log_softmax(lt, -1) - F.log_softmax(ls, -1))).sum(-1)
        kl_sum += kl.sum().item(); ntok += kl.numel()
        ids = enc["input_ids"][0]
        nll_s += F.cross_entropy(ls[:-1], ids[1:], reduction="sum").item()
        nll_t += F.cross_entropy(lt[:-1], ids[1:], reduction="sum").item()
        ntok_lm = ids.numel() - 1
    ppl_s = math.exp(nll_s / max(ntok, 1))
    ppl_t = math.exp(nll_t / max(ntok, 1))
    return kl_sum / max(ntok, 1), ppl_s / ppl_t


def build_rotated_student(base, layer, Q):
    """Inject Q after block `layer` and Q^T before block `layer+1`."""
    s = copy.deepcopy(base)
    layers = s.model.layers
    layers[layer] = RotatedBlockWrapper(layers[layer], Q, on_output=True)
    if layer + 1 < len(layers):
        layers[layer + 1] = RotatedBlockWrapper(layers[layer + 1], Q.t().contiguous(),
                                                on_output=False)
    return s


def train_one(objective, base, teacher, tok, H_T, probes, eval_txt,
              layer, Q, steps, lr, device):
    student = build_rotated_student(base, layer, Q).to(device)
    # train ONLY block l (the rotated-output one); freeze the rest incl. Q^T comp
    for p in student.parameters():
        p.requires_grad_(False)
    blk = student.model.layers[layer]  # RotatedBlockWrapper
    for p in blk.block.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in blk.block.parameters()], lr=lr)

    kl0, pr0 = function_metrics(student, teacher, tok, eval_txt, device)
    print(f"  [{objective}] init: KL(T||S)={kl0:.4f}  PPL_ratio={pr0:.3f}",
          flush=True)

    for step in range(1, steps + 1):
        student.train()
        # recompute student hidden at layer l on probes
        hs = []
        for t in probes:
            enc = tok(t, return_tensors="pt").to(device)
            out = student(**enc, output_hidden_states=True)
            hs.append(out.hidden_states[layer][0])
        H_S = torch.cat(hs, 0)
        if objective == "feature":
            loss = feature_loss(H_S, H_T.detach())
        else:  # cka
            loss = 1 - linear_cka(H_S, H_T.detach())
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 or step == 1:
            print(f"    [{objective} {step:3d}] rep_loss={loss.item():.5f}",
                  flush=True)

    kl1, pr1 = function_metrics(student, teacher, tok, eval_txt, device)
    print(f"  [{objective}] final: KL(T||S)={kl1:.4f}  PPL_ratio={pr1:.3f}",
          flush=True)
    return (kl0, pr0, kl1, pr1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = args.device

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        output_hidden_states=True).to(dev)
    teacher = copy.deepcopy(base).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    probes = [
        "The capital of France is Paris.",
        "Water boils at one hundred degrees Celsius.",
        "Machine learning models are trained on data.",
        "Photosynthesis converts sunlight into chemical energy.",
    ]
    eval_txt = [
        "The history of Rome spans many centuries.",
        "Neural networks consist of layers of neurons.",
        "Economic policy affects inflation and growth.",
    ]

    H_T = hidden_at(teacher, tok, probes, args.layer, dev).float()
    d = H_T.shape[1]
    Q = random_orthogonal(d, args.seed, dev)
    print(f"[setup] layer={args.layer} d={d}; injected function-preserving "
          f"rotation (S==T as a function at init)\n", flush=True)

    print("=== (A) Feature-KD: absolute-coordinate matching ===", flush=True)
    fa = train_one("feature", base, teacher, tok, H_T, probes, eval_txt,
                   args.layer, Q, args.steps, args.lr, dev)
    print("\n=== (B) CKA-KD: basis-invariant ===", flush=True)
    fb = train_one("cka", base, teacher, tok, H_T, probes, eval_txt,
                   args.layer, Q, args.steps, args.lr, dev)

    print("\n=== SUMMARY (Prop 2.1 through training) ===", flush=True)
    print(f"  Feature-KD: KL {fa[0]:.4f}->{fa[2]:.4f}  "
          f"PPL_ratio {fa[1]:.3f}->{fa[3]:.3f}", flush=True)
    print(f"  CKA-KD    : KL {fb[0]:.4f}->{fb[2]:.4f}  "
          f"PPL_ratio {fb[1]:.3f}->{fb[3]:.3f}", flush=True)
    print("\nExpected: Feature-KD degrades the function (KL, PPL ratio rise) "
          "by fighting a\nfunction-preserving rotation; CKA-KD leaves the "
          "function intact (near-zero gradient).", flush=True)


if __name__ == "__main__":
    main()
