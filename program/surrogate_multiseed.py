# PAPER: Result 8 -- controlled 5-seed surrogate (tab:surrogate). See RESULTS_TO_CODE.md
import torch, torch.nn as nn, torch.nn.functional as F, statistics
torch.manual_seed(0)
V, seqlen = 32, 16
d_T, L_T = 64, 4
d_S, L_S = 32, 3


class TinyLM(nn.Module):
    def __init__(s, V, d, L, nhead=4, seqlen=16):
        super().__init__()
        s.V, s.d, s.L, s.seqlen = V, d, L, seqlen
        s.emb = nn.Embedding(V, d)
        s.pos = nn.Parameter(0.02 * torch.randn(seqlen, d))
        s.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d, nhead, 4 * d, batch_first=True,
                                       activation="gelu", dropout=0.0)
            for _ in range(L)])
        s.norm = nn.LayerNorm(d)

    def hidden(s, idx):
        x = s.emb(idx) + s.pos[:idx.shape[1]]
        hs = [x]
        m = torch.triu(torch.full((idx.shape[1], idx.shape[1]), float("-inf")), 1)
        for b in s.blocks:
            x = b(x, src_mask=m)
            hs.append(x)
        return hs

    def logits(s, idx):
        return s.norm(s.hidden(idx)[-1]) @ s.emb.weight.T

    def last_hidden(s, idx):
        return s.norm(s.hidden(idx)[-1])


# ---- data: a fixed bigram rule with 15% noise (nonlinear-ish next-token map) ----
torch.manual_seed(7)
rule = torch.randint(0, V, (V, V))


def corpus(n, sl):
    q = torch.randint(0, V, (n, 2))
    for _ in range(sl - 2):
        a, b = q[:, -2], q[:, -1]
        nx = rule[a, b]
        fl = torch.rand(n) < 0.15
        nx = torch.where(fl, torch.randint(0, V, (n,)), nx)
        q = torch.cat([q, nx[:, None]], 1)
    return q


train_data = corpus(128, seqlen)   # teacher pretraining
probe = corpus(64, seqlen)         # student KD signal (what student trains on)
heldout = corpus(64, seqlen)       # eval ONLY — never trained on (generalization test)

# ---- teacher ----
teacher = TinyLM(V, d_T, L_T, seqlen=seqlen)
o = torch.optim.Adam(teacher.parameters(), lr=3e-3)
for _ in range(500):
    o.zero_grad()
    lg = teacher.logits(train_data)[:, :-1]
    F.cross_entropy(lg.reshape(-1, V), train_data[:, 1:].reshape(-1)).backward()
    o.step()
teacher.eval()
torch.save(teacher.state_dict(), 'teacher.pt')
tppl = torch.exp(F.cross_entropy(
    teacher.logits(heldout)[:, :-1].reshape(-1, V), heldout[:, 1:].reshape(-1)))
print(f"teacher ready, held-out PPL={tppl:.1f}", flush=True)


def ridge(HT, HS, lam=1e-2):
    # paper direction (Remark 4.1): W (d_S x d_T), HS W ~ HT
    A = HS.T @ HS + lam * torch.eye(HS.shape[1])
    return torch.linalg.solve(A, HS.T @ HT)


@torch.no_grad()
def top1(S, T, idx):
    return (S.logits(idx)[:, :-1].argmax(-1)
            == T.logits(idx)[:, :-1].argmax(-1)).float().mean().item()


@torch.no_grad()
def gen(S, T, pr, steps=12):
    seq = pr.clone()
    ag = 0
    for _ in range(steps):
        tn = T.logits(seq)[:, -1].argmax(-1, keepdim=True)
        sn = S.logits(seq)[:, -1].argmax(-1, keepdim=True)
        ag += (tn == sn).float().mean().item()
        seq = torch.cat([seq, tn], 1)[:, -T.seqlen:]
    return ag / steps


@torch.no_grad()
def ppl(S, idx):
    lg = S.logits(idx)[:, :-1]
    ce = F.cross_entropy(lg.reshape(-1, V), idx[:, 1:].reshape(-1))
    return torch.exp(ce).item()


def trn(cond, seed, steps=300, lr=5e-3):
    """cond in {feature, logit, wl}:
       feature = H-only (ridge-aligned hidden match, NO logit signal)
       logit   = output-function KD only
       wl      = feature + logit
    Trains on `probe`, returns metrics on held-out."""
    torch.manual_seed(seed)
    S = TinyLM(V, d_S, L_S, seqlen=seqlen)
    op = torch.optim.Adam(S.parameters(), lr=lr)
    for t in range(steps):
        op.zero_grad()
        loss = 0.0
        if cond in ("feature", "wl"):
            with torch.no_grad():
                HT = teacher.last_hidden(probe).reshape(-1, d_T)
                HS = S.last_hidden(probe).reshape(-1, d_S)
                W = ridge(HT, HS)            # optimal alignment, closed-form
            HT2 = teacher.last_hidden(probe).reshape(-1, d_T)
            HS2 = S.last_hidden(probe).reshape(-1, d_S)
            # paper direction: express student in teacher coords, match there
            loss = loss + ((HT2 - HS2 @ W) ** 2).mean()
        if cond in ("logit", "wl"):
            with torch.no_grad():
                PT = F.softmax(teacher.logits(probe), -1)
            loss = loss + F.kl_div(F.log_softmax(S.logits(probe), -1),
                                   PT, reduction="batchmean")
        loss.backward()
        op.step()
    # all metrics on held-out (generalization)
    return (top1(S, teacher, heldout),
            gen(S, teacher, heldout[:8, :4]),
            ppl(S, heldout))


SEEDS = [1, 2, 3, 4, 5]
CONDS = ["feature", "logit", "wl"]
res = {c: {"t": [], "g": [], "p": []} for c in CONDS}

for sd in SEEDS:
    for c in CONDS:
        a, g, p = trn(c, sd)
        res[c]["t"].append(a)
        res[c]["g"].append(g)
        res[c]["p"].append(p)
        print(f"  seed {sd} [{c:7s}] top1={a:.3f} gen={g:.3f} ppl={p:.2f}",
              flush=True)

label = {"feature": "(i) H-only", "logit": "(ii) logit", "wl": "(iii) W+logit"}
print("\n  cond            top1 mean+/-std     gen mean+/-std      ppl mean+/-std")
for c in CONDS:
    t, g, p = res[c]["t"], res[c]["g"], res[c]["p"]
    sd_t = statistics.stdev(t) if len(t) > 1 else 0.0
    sd_g = statistics.stdev(g) if len(g) > 1 else 0.0
    sd_p = statistics.stdev(p) if len(p) > 1 else 0.0
    print(f"  {label[c]:15s} {statistics.mean(t):.3f}+/-{sd_t:.3f}   "
          f"{statistics.mean(g):.3f}+/-{sd_g:.3f}   "
          f"{statistics.mean(p):.2f}+/-{sd_p:.2f}")

# pillar 2: is (iii) W+logit different from (ii) logit? (noise check on gen)
dg = statistics.mean(res["wl"]["g"]) - statistics.mean(res["logit"]["g"])
pooled = (statistics.stdev(res["logit"]["g"]) + statistics.stdev(res["wl"]["g"])) / 2
verdict = "WITHIN noise (FM adds nothing)" if abs(dg) < pooled else "exceeds noise"
print(f"\n  [pillar 2] gen (iii)-(ii) = {dg:+.3f}  pooled std ~ {pooled:.3f}  => {verdict}")

# pillar 3: H-only collapse relative to teacher-matching
print(f"  [pillar 3] H-only gen = {statistics.mean(res['feature']['g']):.3f} "
      f"vs logit gen = {statistics.mean(res['logit']['g']):.3f} "
      f"(H-only ppl = {statistics.mean(res['feature']['p']):.1f})")
