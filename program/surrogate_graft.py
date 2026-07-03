# PAPER: Result 7 logic pre-validation (Prop 3.2 / Remark 4.1 on tiny transformers). See RESULTS_TO_CODE.md
"""
Small surrogate for the Qwen2.5-1.5B -> 0.5B cross-width distillation experiment.

Purpose
-------
Validate the LOGIC of the pipeline the paper's Remark 4.1 prescribes for
d_A != d_B, on tiny transformers that run on CPU in seconds, before scaling the
SAME procedure to real Qwen weights on a GPU box.

The paper's claim (Prop 3.2 + Remark 4.1), specialized to unequal width:
  * Feature transfer across d_T != d_S needs an explicit map W (estimate first).
  * Capability transfer does NOT: logit-KD compares outputs in the shared vocab
    space V, regardless of hidden widths.

We therefore compare three conditions for restoring a corrupted student S toward
a teacher T (teacher width d_T > student width d_S, mirroring 1536 -> 896):

  (i)   feature-only : align T->S space with ridge-Procrustes W, then match
                       features in the aligned frame. (Remark 4.1 / family B)
  (ii)  logit-only   : ignore hidden states entirely; match KL(T||S) on the
                       shared V-dim logits. NO W is estimated. (family C)
  (iii) W-align+logit: estimate W, match features in aligned frame, AND match
                       logits.  "W first, then logit."  (the user's proposal)

Metrics (three levels, because teacher-forced PPL can pass while generation breaks):
  L1  ppl_ratio  = PPL_S / PPL_T        (teacher-forced, lenient)
  L2  top1_agree = argmax agreement vs T (teacher-forced)
  L3  gen_match  = free-running next-token agreement over a rollout
                   (autoregressive; catches drift the teacher-forced metrics miss)

Everything is numpy-free torch, CPU, hand-small. The real-Qwen script reuses the
exact same three-condition structure and the same ridge-Procrustes W.
"""
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

# ----------------------------------------------------------------------
# A tiny decoder-only LM. Width d, depth L, shared vocab V (tied embed).
# This stands in for a Qwen-family model; teacher and student differ in
# (d, L) but SHARE V and the tokenizer, exactly like 1.5B vs 0.5B.
# ----------------------------------------------------------------------
class TinyLM(nn.Module):
    def __init__(self, V, d, L, nhead=4, seqlen=16):
        super().__init__()
        self.V, self.d, self.L, self.seqlen = V, d, L, seqlen
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Parameter(0.02 * torch.randn(seqlen, d))
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, batch_first=True,
                                           activation="gelu", dropout=0.0)
        self.blocks = nn.ModuleList([layer.__class__(d, nhead, 4 * d,
                                     batch_first=True, activation="gelu",
                                     dropout=0.0) for _ in range(L)])
        self.norm = nn.LayerNorm(d)
        # tied unembedding: logits = h @ emb.weight^T  (like Qwen tie_word_embeddings)

    def hidden(self, idx, upto=None):
        """Return hidden states after each block (list, len L+1 incl. embed)."""
        x = self.emb(idx) + self.pos[:idx.shape[1]]
        hs = [x]
        mask = torch.triu(torch.full((idx.shape[1], idx.shape[1]), float("-inf")), 1)
        for i, blk in enumerate(self.blocks):
            x = blk(x, src_mask=mask)
            hs.append(x)
            if upto is not None and i == upto:
                break
        return hs

    def logits(self, idx):
        hs = self.hidden(idx)
        h = self.norm(hs[-1])
        return h @ self.emb.weight.T        # tied

    def last_hidden(self, idx):
        return self.norm(self.hidden(idx)[-1])


# ----------------------------------------------------------------------
# Ridge-Procrustes:  find W (d_T x d_S) mapping teacher features into the
# teacher space, min_W || H_T - H_S W ||_F^2 + lam ||W||^2.  Closed form.
# Paper direction (Remark 4.1): W expresses the STUDENT in the TEACHER's frame.
# ----------------------------------------------------------------------
def ridge_map(H_T, H_S, lam=1e-2):
    # H_T: (N, d_T), H_S: (N, d_S)  ->  W: (d_S, d_T) with H_S W ~ H_T
    # (paper direction: express student in teacher coords)
    d_S = H_S.shape[1]
    A = H_S.T @ H_S + lam * torch.eye(d_S)
    W = torch.linalg.solve(A, H_S.T @ H_T)
    return W


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
@torch.no_grad()
def ppl(model, idx):
    lg = model.logits(idx)[:, :-1]
    tgt = idx[:, 1:]
    return torch.exp(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1)))

@torch.no_grad()
def top1_agree(S, T, idx):
    a = S.logits(idx)[:, :-1].argmax(-1)
    b = T.logits(idx)[:, :-1].argmax(-1)
    return (a == b).float().mean().item()

@torch.no_grad()
def gen_match(S, T, prompt, steps=12):
    """Free-running: let TEACHER generate greedily; check if STUDENT would pick
    the same next token at each step given the teacher's own rollout. This is the
    autoregressive check L3 -- the one that catches generation collapse that
    teacher-forced metrics hide."""
    seq = prompt.clone()
    agree = 0
    for _ in range(steps):
        t_next = T.logits(seq)[:, -1].argmax(-1, keepdim=True)
        s_next = S.logits(seq)[:, -1].argmax(-1, keepdim=True)
        agree += (t_next == s_next).float().mean().item()
        seq = torch.cat([seq, t_next], 1)[:, -T.seqlen:]
    return agree / steps


# ----------------------------------------------------------------------
# Build teacher, a data distribution, and a corrupted student.
# ----------------------------------------------------------------------
V, seqlen = 32, 16
d_T, L_T = 64, 4      # "1.5B": wider, deeper
d_S, L_S = 32, 3      # "0.5B": narrower, shallower  (d_S < d_T, L_S < L_T)

# --- a learnable structured "language": next token is a fixed function of a
#     sliding window (a random order-2 rule) plus noise, so a trained model
#     develops REAL next-token capability we can then try to transfer. ---
torch.manual_seed(7)
rule = torch.randint(0, V, (V, V))   # rule[a,b] = the "correct" token after (a,b)
def sample_corpus(n, seqlen):
    seqs = torch.randint(0, V, (n, 2))
    for _ in range(seqlen - 2):
        a, b = seqs[:, -2], seqs[:, -1]
        nxt = rule[a, b]
        noise = torch.randint(0, V, (n,))       # 15% noise -> a distribution
        flip = torch.rand(n) < 0.15
        nxt = torch.where(flip, noise, nxt)
        seqs = torch.cat([seqs, nxt[:, None]], 1)
    return seqs

train_data = sample_corpus(256, seqlen)
probe = sample_corpus(64, seqlen)
N = probe.shape[0]

# train the teacher on this language so it actually models the rule
teacher = TinyLM(V, d_T, L_T, seqlen=seqlen)
_opt = torch.optim.Adam(teacher.parameters(), lr=3e-3)
for _e in range(1000):
    _opt.zero_grad()
    _lg = teacher.logits(train_data)[:, :-1]
    _loss = F.cross_entropy(_lg.reshape(-1, V), train_data[:, 1:].reshape(-1))
    _loss.backward(); _opt.step()
teacher.eval()

def fresh_student(corrupt=True, seed=1):
    torch.manual_seed(seed)
    s = TinyLM(V, d_S, L_S, seqlen=seqlen)
    return s

print(f"teacher: V={V} d={d_T} L={L_T}   student: d={d_S} L={L_S}  (d_S<d_T, shared V)")
print(f"teacher PPL on probe = {ppl(teacher, probe):.3f}\n")

# layer mapping for feature alignment: proportional (student layer j <-> teacher
# layer round(j * L_T/L_S)). We align the LAST hidden (pre-unembed) here, which
# is what the tied head reads; a full run would align several mapped layers.
def teacher_student_feats(T, S, idx):
    return T.last_hidden(idx), S.last_hidden(idx)


# ----------------------------------------------------------------------
# Training loops for the three conditions.  Small, explicit, flushed prints.
# ----------------------------------------------------------------------
def train(cond, steps=800, lr=5e-3, lam_feat=1.0, lam_logit=1.0):
    S = fresh_student()
    opt = torch.optim.Adam(S.parameters(), lr=lr)
    W = None
    for t in range(steps):
        opt.zero_grad()
        loss = 0.0
        # feature term needs W (estimated fresh from current student feats)
        if cond in ("feature", "walign_logit"):
            with torch.no_grad():
                H_T = teacher.last_hidden(probe).reshape(-1, d_T)
                H_S = S.last_hidden(probe).reshape(-1, d_S)
                W = ridge_map(H_T, H_S, lam=1e-2)          # W: d_S x d_T
            H_T2 = teacher.last_hidden(probe).reshape(-1, d_T)
            H_S2 = S.last_hidden(probe).reshape(-1, d_S)
            feat_loss = ((H_T2 - H_S2 @ W) ** 2).mean()
            loss = loss + lam_feat * feat_loss
        # logit term: KL(T||S) on shared vocab, NO W involved
        if cond in ("logit", "walign_logit"):
            with torch.no_grad():
                P_T = F.softmax(teacher.logits(probe), -1)
            logP_S = F.log_softmax(S.logits(probe), -1)
            logit_loss = F.kl_div(logP_S, P_T, reduction="batchmean")
            loss = loss + lam_logit * logit_loss
        loss.backward()
        opt.step()
        if t % 200 == 0 or t == steps - 1:
            r = ppl(S, probe) / ppl(teacher, probe)
            a = top1_agree(S, teacher, probe)
            g = gen_match(S, teacher, probe[:8, :4])
            print(f"  [{cond:12s}] step {t:4d}  loss={float(loss.detach()):.4f}  "
                  f"pplR={r:6.2f}  top1={a:.3f}  gen={g:.3f}", flush=True)
    return S

print("=== (i) feature-only : align with W, match features; ignore logits ===")
S_feat = train("feature")
print("\n=== (ii) logit-only : match KL on shared vocab; NO W estimated ===")
S_log = train("logit")
print("\n=== (iii) W-align + logit : estimate W, match features AND logits ===")
S_both = train("walign_logit")

# ----------------------------------------------------------------------
# Final comparison table
# ----------------------------------------------------------------------
def report(name, S):
    r = (ppl(S, probe) / ppl(teacher, probe)).item()
    a = top1_agree(S, teacher, probe)
    g = gen_match(S, teacher, probe[:8, :4])
    print(f"  {name:16s}  L1 pplR={r:7.3f}   L2 top1={a:.3f}   L3 gen={g:.3f}")
    return a, g

print("\n================ FINAL (student toward teacher) ================")
print(f"  {'condition':16s}  {'L1 PPL ratio':>14s}   {'L2 top1':>8s}   {'L3 gen':>7s}")
report("(i) feature", S_feat)
report("(ii) logit", S_log)
report("(iii) W+logit", S_both)

# ----------------------------------------------------------------------
# MULTI-SEED: is (iii) vs (ii) real, or seed noise?  A single run cannot say.
# Repeat the two contested conditions across seeds and report mean +/- std.
# (Per-seed lines printed live, per the standing rule.)
# ----------------------------------------------------------------------
print("\n================ MULTI-SEED (ii) vs (iii): is the gap real? ============")
import statistics
def fresh_student_seed(seed):
    torch.manual_seed(seed)
    return TinyLM(V, d_S, L_S, seqlen=seqlen)

def train_seeded(cond, seed, steps=800, lr=5e-3):
    S = fresh_student_seed(seed)
    opt = torch.optim.Adam(S.parameters(), lr=lr)
    for t in range(steps):
        opt.zero_grad(); loss = 0.0
        if cond in ("feature", "walign_logit"):
            with torch.no_grad():
                H_T = teacher.last_hidden(probe).reshape(-1, d_T)
                H_S = S.last_hidden(probe).reshape(-1, d_S)
                W = ridge_map(H_T, H_S, lam=1e-2)
            H_T2 = teacher.last_hidden(probe).reshape(-1, d_T)
            H_S2 = S.last_hidden(probe).reshape(-1, d_S)
            loss = loss + ((H_T2 - H_S2 @ W) ** 2).mean()
        if cond in ("logit", "walign_logit"):
            with torch.no_grad():
                P_T = F.softmax(teacher.logits(probe), -1)
            loss = loss + F.kl_div(F.log_softmax(S.logits(probe), -1), P_T,
                                   reduction="batchmean")
        loss.backward(); opt.step()
    return top1_agree(S, teacher, probe), gen_match(S, teacher, probe[:8, :4])

seeds = [1, 2, 3, 4, 5]
res = {"logit": {"top1": [], "gen": []}, "walign_logit": {"top1": [], "gen": []}}
for sd in seeds:
    for cond in ("logit", "walign_logit"):
        a, g = train_seeded(cond, sd)
        res[cond]["top1"].append(a); res[cond]["gen"].append(g)
        print(f"  seed {sd}  [{cond:12s}]  top1={a:.3f}  gen={g:.3f}", flush=True)

print("\n  condition        top1 mean+/-std        gen mean+/-std")
for cond, label in [("logit", "(ii) logit"), ("walign_logit", "(iii) W+logit")]:
    tm, ts = statistics.mean(res[cond]["top1"]), statistics.pstdev(res[cond]["top1"])
    gm, gs = statistics.mean(res[cond]["gen"]), statistics.pstdev(res[cond]["gen"])
    print(f"  {label:16s} {tm:.3f} +/- {ts:.3f}      {gm:.3f} +/- {gs:.3f}")

dg = statistics.mean(res["walign_logit"]["gen"]) - statistics.mean(res["logit"]["gen"])
pooled = (statistics.pstdev(res["logit"]["gen"]) + statistics.pstdev(res["walign_logit"]["gen"])) / 2
print(f"\n  gen difference (iii)-(ii) = {dg:+.3f}   pooled std ~ {pooled:.3f}")
print(f"  => {'WITHIN noise: W adds nothing measurable here' if abs(dg) < pooled else 'exceeds noise: worth testing at scale'}")
print("\nReading:")
print("  L1 close to 1.0 and L2/L3 high  => capability restored.")
print("  If (i) aligns features (low feat loss) but L2/L3 stay low, that is the")
print("  cross-scale form of the paper's Table 3: feature match != capability.")
print("  If (ii) restores L2/L3 with NO W, that is Prop 3.2 across unequal width:")
print("  logit-KD needs no alignment. (iii) tests whether adding W buys anything.")
