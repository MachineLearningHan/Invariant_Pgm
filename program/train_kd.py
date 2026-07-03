# PAPER: Result 8 -- feature-only collapse; modes logit/fm/fm_only (tab:fmonly). See RESULTS_TO_CODE.md
"""
Qwen2.5-1.5B (teacher) -> Qwen2.5-0.5B (student) knowledge distillation.

Two modes via --mode:
  logit  : CE + logit-KD only            (baseline)
  fm     : CE + logit-KD + feature-match (attention-output feature matching)

Feature matching:
  - 4 anchor layers, per-anchor learnable linear projection (student_hidden -> teacher_hidden, no bias)
  - smooth-L1 on hidden states, masked-mean over real tokens
  - lambda warm-up 0 -> LAMBDA_MAX over first WARMUP_FRAC of steps

Anchors are derived at runtime from the ACTUAL configs (endpoints aligned),
so a spec mismatch (layers/hidden) self-corrects instead of silently breaking.

Standing rule: live per-step logging with flush=True; run under `python -u`.

Usage:
  python -u train_kd.py --mode fm \
    --teacher Qwen/Qwen2.5-1.5B --student Qwen/Qwen2.5-0.5B \
    --data <jsonl with {"text": ...}> --steps 2000 --bs 4 --seqlen 1024
"""
import argparse, math, sys, json, os, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic data order; keep cudnn fast (KD compares are robust to it)
    os.environ["PYTHONHASHSEED"] = str(seed)

KD_T = 2.0
LOGIT_KD_W, CE_W = 0.5, 0.5
LAMBDA_MAX = 0.3
WARMUP_FRAC = 0.10
N_ANCHORS = 4


# ---------------------------------------------------------------- anchors
def build_anchors(s_layers, t_layers, n=N_ANCHORS):
    """Pick n student anchors spread across depth incl. last layer,
    map each to teacher via endpoint-aligned even spacing."""
    # student anchor indices: evenly spaced, always include the final layer
    s_idx = sorted({round((s_layers - 1) * k / (n - 1)) for k in range(n)})
    def to_teacher(i):
        return round(i * (t_layers - 1) / (s_layers - 1))
    return {s: to_teacher(s) for s in s_idx}


# ---------------------------------------------------------------- data
class JsonlText(Dataset):
    def __init__(self, path, tok, seqlen):
        rows, bad = [], 0
        with open(path) as f:
            for l in f:
                if not l.strip():
                    continue
                try:
                    obj = json.loads(l)
                    t = obj["text"]
                    if isinstance(t, str) and t:
                        rows.append(t)
                    else:
                        bad += 1
                except (json.JSONDecodeError, KeyError):
                    bad += 1
        if bad:
            print(f"[data] skipped {bad} malformed/empty lines, kept {len(rows)}",
                  flush=True)
        if not rows:
            raise ValueError(f"no valid rows in {path} (all {bad} lines malformed)")
        self.rows = rows
        self.tok, self.seqlen = tok, seqlen
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        enc = self.tok(self.rows[i], truncation=True, max_length=self.seqlen,
                       padding="max_length", return_tensors="pt")
        ids = enc["input_ids"][0]
        m = enc["attention_mask"][0]
        labels = ids.clone()
        labels[m == 0] = -100
        return {"input_ids": ids, "attention_mask": m, "labels": labels}


# ---------------------------------------------------------------- feature matcher
class FeatureMatcher(nn.Module):
    def __init__(self, anchors, s_hidden, t_hidden):
        super().__init__()
        self.anchors = anchors
        self.heads = nn.ModuleDict(
            {str(s): nn.Linear(s_hidden, t_hidden, bias=False) for s in anchors})
    def forward(self, s_hs, t_hs, mask):
        m = mask.unsqueeze(-1).float()
        denom = m.sum().clamp_min(1.0)
        total = 0.0
        for s_idx, t_idx in self.anchors.items():
            proj = self.heads[str(s_idx)](s_hs[s_idx + 1])
            tgt = t_hs[t_idx + 1].detach()
            l = F.smooth_l1_loss(proj, tgt, reduction="none").mean(-1, keepdim=True)
            total = total + (l * m).sum() / denom
        return total / len(self.anchors)


# ---------------------------------------------------------------- losses
def logit_kd(s_logits, t_logits, mask, chunk=256):
    """Memory-lean KL(teacher||student) at temperature KD_T.
    Flattens to [N, V] and processes N in chunks so we never hold two
    full [B, L, V] fp32 tensors at once (the OOM culprit on 32GB cards).
    """
    B, L, V = s_logits.shape
    s_flat = s_logits.reshape(-1, V)
    t_flat = t_logits.reshape(-1, V)
    m_flat = mask.reshape(-1).float()               # [N]
    N = s_flat.size(0)
    total = s_logits.new_zeros(())                  # scalar accumulator
    for i in range(0, N, chunk):
        sl = s_flat[i:i + chunk]
        tl = t_flat[i:i + chunk].detach()
        mm = m_flat[i:i + chunk]
        s = F.log_softmax(sl / KD_T, -1)
        t = F.softmax(tl / KD_T, -1)
        kl = (t * (t.clamp_min(1e-9).log() - s)).sum(-1)   # [chunk], KL(t||s)
        total = total + (kl * mm).sum()
    return total / m_flat.sum().clamp_min(1.0) * (KD_T ** 2)

def lam_at(step, total):
    w = int(total * WARMUP_FRAC)
    return LAMBDA_MAX if (w <= 0 or step >= w) else LAMBDA_MAX * step / w


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["logit", "fm"], required=True)
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ce_w", type=float, default=0.5, help="CE (hard label) weight")
    ap.add_argument("--kd_w", type=float, default=0.5, help="logit-KD weight")
    ap.add_argument("--fm_only", action="store_true",
                    help="feature-matching ONLY: no CE, no logit-KD; constant lambda=fm_w")
    ap.add_argument("--fm_w", type=float, default=1.0,
                    help="constant feature-matching weight when --fm_only")
    ap.add_argument("--save", default="student_kd.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    args = ap.parse_args()

    set_seed(args.seed)
    print(f"[seed] {args.seed}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if dev == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(args.student)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    print("loading teacher...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=dt).to(dev).eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    print("loading student...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=dt).to(dev).train()

    # ---- verify specs from the actual loaded configs, build anchors ----
    s_layers = student.config.num_hidden_layers
    t_layers = teacher.config.num_hidden_layers
    s_hid = student.config.hidden_size
    t_hid = teacher.config.hidden_size
    assert tok.vocab_size <= student.config.vocab_size
    if not args.fm_only:
        assert student.config.vocab_size == teacher.config.vocab_size, \
            "vocab mismatch -> logit-KD needs aligned vocab (use --fm_only for cross-vocab)"
    anchors = build_anchors(s_layers, t_layers)
    print(f"[cfg] student {s_layers}L/{s_hid}H  teacher {t_layers}L/{t_hid}H", flush=True)
    print(f"[cfg] anchors (student->teacher): {anchors}", flush=True)
    if args.fm_only:
        print("[mode] FEATURE-ONLY: no CE, no logit-KD, constant "
              f"lambda={args.fm_w}", flush=True)

    use_hidden = (args.mode == "fm") or args.fm_only
    matcher = None
    params = list(student.parameters())
    if use_hidden:
        matcher = FeatureMatcher(anchors, s_hid, t_hid).to(dev).to(dt)
        params += list(matcher.parameters())

    opt = torch.optim.AdamW(params, lr=args.lr)
    g = torch.Generator()
    g.manual_seed(args.seed)
    dl = DataLoader(JsonlText(args.data, tok, args.seqlen),
                    batch_size=args.bs, shuffle=True, drop_last=True, generator=g)

    step = 0
    opt.zero_grad()
    accum_ct = 0
    log_ce = log_kd = log_fm = 0.0
    while step < args.steps:
        for batch in dl:
            if step >= args.steps: break
            batch = {k: v.to(dev) for k, v in batch.items()}
            ids, mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]

            with torch.no_grad():
                t_out = teacher(ids, attention_mask=mask,
                                output_hidden_states=use_hidden)
            s_out = student(ids, attention_mask=mask,
                            output_hidden_states=use_hidden)

            if args.fm_only:
                fm = matcher(s_out.hidden_states, t_out.hidden_states, mask)
                loss = args.fm_w * fm
                ce_v, kd_v, fm_v, lam_v = 0.0, 0.0, fm.item(), args.fm_w
            else:
                ce = F.cross_entropy(
                    s_out.logits.view(-1, s_out.logits.size(-1)),
                    labels.view(-1), ignore_index=-100)
                kd = logit_kd(s_out.logits, t_out.logits, mask)
                if args.mode == "fm":
                    fm = matcher(s_out.hidden_states, t_out.hidden_states, mask)
                    lam = lam_at(step, args.steps)
                    loss = args.ce_w * ce + args.kd_w * kd + lam * fm
                    fm_v, lam_v = fm.item(), lam
                else:
                    loss = args.ce_w * ce + args.kd_w * kd
                    fm_v, lam_v = 0.0, 0.0
                ce_v, kd_v = ce.item(), kd.item()

            (loss / args.accum).backward()
            log_ce += ce_v; log_kd += kd_v; log_fm += fm_v
            accum_ct += 1

            if accum_ct == args.accum:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad()
                a = args.accum
                tag = "fm_only" if args.fm_only else args.mode
                print(f"[{tag}][step {step:>6}/{args.steps}] "
                      f"ce={log_ce/a:.4f} kd={log_kd/a:.4f} "
                      f"fm={log_fm/a:.4f} lam={lam_v:.3f}", flush=True)
                log_ce = log_kd = log_fm = 0.0
                accum_ct = 0
                step += 1

    torch.save(student.state_dict(), args.save)
    print(f"saved -> {args.save}", flush=True)


if __name__ == "__main__":
    main()
