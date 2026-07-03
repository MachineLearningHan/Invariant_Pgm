# PAPER: Result 7 -- cross-width T=1.5B->S=0.5B, teacher-forced/generation gap. See RESULTS_TO_CODE.md
"""
Cross-width distillation: Qwen2.5-1.5B (teacher) -> Qwen2.5-0.5B (student).

Tests the paper's Prop 3.2 / Remark 4.1 at real scale, across unequal hidden
width (d_T=1536 -> d_S=896), with the shared Qwen tokenizer/vocab (V=151936).

Three conditions (the surrogate's logic, unchanged):
  (i)   feature : estimate ridge map W (d_S x d_T), express the student in the
                  teacher's frame (H_S W) and match teacher features there,
                  ||H_T - H_S W||; NO logit term.              [family B]
  (ii)  logit   : match KL(T||S) on the shared vocab; NO W.    [family C]
  (iii) wl      : estimate W, match features AND logits.       ["W first, then logit"]

Three metrics (teacher-forced can pass while generation breaks):
  L1 ppl_ratio  = PPL_S / PPL_T          (teacher-forced)
  L2 top1_agree = argmax agreement vs T  (teacher-forced)
  L3 gen_match  = free-running next-token agreement over a rollout (autoregressive)

Multi-seed with per-seed live output and mean +/- std, because the surrogate
showed the (ii)-vs-(iii) gap and even top1 itself sit inside seed noise at small
scale -- a single run says nothing. This is the standing rule (flush per seed).

STUDENT INIT (choose --init):
  corrupt : start from pretrained 0.5B, re-initialize a fraction of middle layers
            (clean "restoration" measurement, mirrors paper Sec 7)
  scratch : re-initialize ALL 0.5B decoder layers (Result-6 style, hardest)
  pre     : start from the intact pretrained 0.5B (measures improvement, not restore)

Run (single GPU is enough for 0.5B student + 1.5B frozen teacher in bf16):
  python qwen_crosswidth.py --init corrupt --seeds 3 --steps 2000 \
      --conditions feature logit wl --out results.json

unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE

python qwen_crosswidth.py \
    --teacher /home/models/hub/Qwen2.5-1.5B \
    --student /home/models/hub/Qwen2.5-0.5B \
    --dataset wikitext --data_cache /home/models/hub \
    --init corrupt --seeds 1 --steps 500 --n_probe 32 \
    --log_every 500 --show_gen \
    --conditions logit --out check2.json
    

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -u qwen_crosswidth.py \
  --init corrupt --frac 0.5 --conditions feature logit wl \
  --seeds 3 --steps 2000 --lam_feat 0.2 \
  --teacher /home/models/hub/Qwen2.5-1.5B \
  --student /home/models/hub/Qwen2.5-0.5B \
  --offline --out results_crosswidth_lam02.json 2>&1 | tee crosswidth_lam02.log
  
      
Requires: torch, transformers, datasets (optional; falls back to a builtin corpus).
Teacher stays frozen in eval/bf16; only student params (and W, which is closed-form)
are updated. W is re-estimated from current student features each K steps.
"""
import argparse, json, math, statistics, sys, time
import torch, torch.nn as nn, torch.nn.functional as F

# Defaults are HF hub IDs; override with --teacher / --student to point at a
# local directory (e.g. /home/models/hub/models--Qwen--Qwen2.5-1.5B/
# snapshots/<hash>, or any folder holding config.json + weights).
TEACHER_ID = "Qwen/Qwen2.5-1.5B"
STUDENT_ID = "Qwen/Qwen2.5-0.5B"


# ----------------------------------------------------------------------
# Data: a probe corpus of token sequences. Uses a HF dataset if available,
# else a builtin repetitive corpus (enough to exercise the pipeline).
# ----------------------------------------------------------------------
def build_probe(tok, n_seq, seqlen, device, dataset=None, cache_dir=None,
                text_file=None):
    """Build a probe corpus. Priority: (1) a plain local text file/dir if given
    (bypasses the datasets library entirely), (2) a cached HF dataset, (3) a
    builtin fallback. The text_file route is the robust one for offline boxes."""
    texts = []

    # (1) plain local text -- most robust, no datasets machinery
    if text_file:
        import os, glob
        paths = []
        if os.path.isdir(text_file):
            for ext in ("*.txt", "*.jsonl", "*.json"):
                paths += glob.glob(os.path.join(text_file, "**", ext),
                                   recursive=True)
        else:
            paths = [text_file]
        for pth in paths:
            try:
                with open(pth, "r", errors="ignore") as f:
                    if pth.endswith(".jsonl"):
                        import json
                        for line in f:
                            try:
                                o = json.loads(line)
                                t = o.get("text") or o.get("content") or ""
                                if len(t.strip()) > 60:
                                    texts.append(t.strip())
                            except Exception:
                                pass
                            if len(texts) >= n_seq * 2:
                                break
                    else:
                        buf = f.read()
                        for para in buf.split("\n"):
                            if len(para.strip()) > 60:
                                texts.append(para.strip())
                            if len(texts) >= n_seq * 2:
                                break
            except Exception as e:
                print(f"  [data] read {pth} failed: {e}", flush=True)
            if len(texts) >= n_seq * 2:
                break
        if texts:
            print(f"  [data] loaded {len(texts)} lines from text_file "
                  f"{text_file}", flush=True)

    # (2) HF dataset from local cache
    if not texts and dataset:
        from datasets import load_dataset
        attempts = []
        if dataset == "wikitext":
            for cfg in ["wikitext-103-raw-v1", "wikitext-2-raw-v1",
                        "wikitext-103-v1", "wikitext-2-v1"]:
                attempts.append(dict(path="wikitext", name=cfg, split="train"))
        else:
            attempts.append(dict(path=dataset, split="train"))
        for kw in attempts:
            try:
                ds = load_dataset(streaming=True, cache_dir=cache_dir, **kw)
                for ex in ds:
                    t = ex.get("text") or ex.get("content") or ""
                    if len(t.strip()) > 60:
                        texts.append(t.strip())
                    if len(texts) >= n_seq * 2:
                        break
                if texts:
                    print(f"  [data] loaded {len(texts)} lines from "
                          f"{kw.get('path')} {kw.get('name','')}", flush=True)
                    break
            except Exception as e:
                print(f"  [data] {kw.get('name', dataset)}: "
                      f"{type(e).__name__}: {str(e)[:70]}", flush=True)

    # (3) builtin fallback
    if not texts:
        base = ("The quantity that survives a change of basis is the output "
                "function; capability lives on the joint quotient, not the "
                "representation. ")
        texts = [base * 4 for _ in range(n_seq)]
        print("  [data] USING BUILTIN CORPUS -- gen values on repetitive text "
              "are NOT meaningful; pass --text_file with real text.", flush=True)

    enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True,
              max_length=seqlen)
    ids = enc["input_ids"][:n_seq].to(device)
    return ids


# ----------------------------------------------------------------------
# Ridge-Procrustes: W (d_S x d_T), min_W ||H_T - H_S W||^2 + lam||W||^2.
# Closed form; this is Remark 4.1's learned linear map in its ridge case.
# Paper direction: W expresses the STUDENT's representation in the TEACHER's
# coordinate system (student -> teacher), matching Remark 4.1's
# min_W ||H_A - H_B W|| with A=teacher, B=student. (Earlier code solved the
# reverse, projecting teacher into student space; flipped here to match paper.)
# ----------------------------------------------------------------------
def ridge_map(H_T, H_S, lam=1e-2):
    d_S = H_S.shape[1]
    A = H_S.T @ H_S + lam * torch.eye(d_S, device=H_S.device, dtype=H_S.dtype)
    return torch.linalg.solve(A, H_S.T @ H_T)          # W: (d_S, d_T)


# ----------------------------------------------------------------------
# Hidden-state extraction. We align the LAST hidden state (pre-lm_head), which
# the tied/again-untied head reads. A fuller run maps several layers; last-layer
# is the cleanest single choice and what capability is read from.
# ----------------------------------------------------------------------
@torch.no_grad()
def last_hidden(model, ids, attn):
    out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
    return out.hidden_states[-1]           # (B, T, d)

def last_hidden_grad(model, ids, attn):
    out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
    return out.hidden_states[-1]


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
@torch.no_grad()
def ppl(model, ids, attn):
    lg = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
    tgt = ids[:, 1:]
    l = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tgt.reshape(-1),
                        reduction="mean")
    return torch.exp(l).item()

@torch.no_grad()
def top1_agree(S, T, ids, attn):
    a = S(input_ids=ids, attention_mask=attn).logits[:, :-1].argmax(-1)
    b = T(input_ids=ids, attention_mask=attn).logits[:, :-1].argmax(-1)
    return (a == b).float().mean().item()

@torch.no_grad()
def gen_match(S, T, prompt, attn, steps, pad_id):
    """Free-running: teacher generates greedily; does the student pick the same
    next token at each step of the teacher's own rollout? Catches autoregressive
    drift that teacher-forced metrics hide -- the L3 the surrogate flagged."""
    seq = prompt.clone()
    m = attn.clone()
    agree = 0.0
    for _ in range(steps):
        t_next = T(input_ids=seq, attention_mask=m).logits[:, -1].argmax(-1, keepdim=True)
        s_next = S(input_ids=seq, attention_mask=m).logits[:, -1].argmax(-1, keepdim=True)
        agree += (t_next == s_next).float().mean().item()
        seq = torch.cat([seq, t_next], 1)
        m = torch.cat([m, torch.ones_like(t_next)], 1)
    return agree / steps


def _rep_distinct(tokens, n=3):
    """repetition rate and distinct-n for one token list."""
    if len(tokens) < n:
        return 0.0, 1.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    uniq = len(set(grams))
    total = len(grams)
    distinct = uniq / total
    rep = 1.0 - distinct
    return rep, distinct


@torch.no_grad()
def gen_quality(model, prompt, attn, steps, n=3):
    """Free-running greedy generation from `model` alone, then measure how much it
    loops (rep-n) / how varied it is (distinct-n). This is capability of the
    generator itself -- independent of the teacher -- so it catches collapse a
    teacher-token-match metric misses. Returns mean over the batch."""
    seq = prompt.clone(); m = attn.clone()
    for _ in range(steps):
        nx = model(input_ids=seq, attention_mask=m).logits[:, -1].argmax(-1, keepdim=True)
        seq = torch.cat([seq, nx], 1); m = torch.cat([m, torch.ones_like(nx)], 1)
    gen_part = seq[:, prompt.shape[1]:]           # only the generated tail
    reps, dists = [], []
    for row in gen_part.tolist():
        r, d = _rep_distinct(row, n)
        reps.append(r); dists.append(d)
    return sum(reps) / len(reps), sum(dists) / len(dists)


@torch.no_grad()
def rollout_kl(S, T, prompt, attn, steps):
    """Along the TEACHER's greedy rollout, average KL(teacher || student) over the
    next-token distributions. Unlike top1/gen this is a soft distributional match
    -- a student that picks a different-but-reasonable token scores well, so it
    separates 'fluent but divergent' from 'genuinely wrong'."""
    seq = prompt.clone(); m = attn.clone()
    kls = 0.0
    for _ in range(steps):
        tl = T(input_ids=seq, attention_mask=m).logits[:, -1].float()
        sl = S(input_ids=seq, attention_mask=m).logits[:, -1].float()
        P = F.softmax(tl, -1); logP = F.log_softmax(tl, -1); logQ = F.log_softmax(sl, -1)
        kls += (P * (logP - logQ)).sum(-1).mean().item()
        t_next = tl.argmax(-1, keepdim=True)
        seq = torch.cat([seq, t_next], 1); m = torch.cat([m, torch.ones_like(t_next)], 1)
    return kls / steps


# ----------------------------------------------------------------------
# Student corruption
# ----------------------------------------------------------------------
def corrupt_student(student, init, frac=0.5):
    layers = student.model.layers
    L = len(layers)
    if init == "pre":
        return student
    if init == "scratch":
        idxs = list(range(L))
    else:  # corrupt: middle fraction, protect first/last two
        lo, hi = 2, L - 2
        k = max(1, int(frac * (hi - lo)))
        mid = (lo + hi) // 2
        idxs = list(range(mid - k // 2, mid - k // 2 + k))
    for i in idxs:
        for p in layers[i].parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, std=0.02)
            else:
                nn.init.zeros_(p)
    print(f"  [init={init}] re-initialized layers {idxs} of {L}", flush=True)
    return student


# ----------------------------------------------------------------------
# One training run for one (condition, seed)
# ----------------------------------------------------------------------
def run(cond, seed, teacher, tok, probe, attn, args, device):
    from transformers import AutoModelForCausalLM
    torch.manual_seed(seed)
    student = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=torch.bfloat16).to(device)
    student = corrupt_student(student, args.init, args.frac)
    student.train()
    d_T = teacher.config.hidden_size
    d_S = student.config.hidden_size

    # only train student decoder params (+ lm_head if untied); teacher frozen
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)

    prompt = probe[:min(8, probe.shape[0]), :8]
    pattn = attn[:prompt.shape[0], :8]
    ppl_T = ppl(teacher, probe, attn)
    # teacher's own generation quality = the baseline any student is measured against
    t_rep, t_dist = gen_quality(teacher, prompt, pattn, args.gen_steps)

    W = None
    t0 = time.time()
    for step in range(args.steps):
        opt.zero_grad()
        loss = torch.zeros((), device=device)

        if cond in ("feature", "wl"):
            with torch.no_grad():
                H_T = last_hidden(teacher, probe, attn).reshape(-1, d_T).float()
                H_S = last_hidden(student, probe, attn).reshape(-1, d_S).float()
                W = ridge_map(H_T, H_S, lam=args.ridge_lam)   # (d_S, d_T)
            H_T2 = last_hidden_grad(teacher, probe, attn).reshape(-1, d_T).float()
            H_S2 = last_hidden_grad(student, probe, attn).reshape(-1, d_S).float()
            # paper direction: express student in teacher coords (H_S W) and
            # match the teacher there -> ||H_T - H_S W||^2 (Remark 4.1).
            loss = loss + args.lam_feat * ((H_T2 - H_S2 @ W) ** 2).mean()

        if cond in ("logit", "wl"):
            with torch.no_grad():
                P_T = F.softmax(teacher(input_ids=probe, attention_mask=attn)
                                .logits.float(), -1)
            logP_S = F.log_softmax(student(input_ids=probe, attention_mask=attn)
                                   .logits.float(), -1)
            loss = loss + args.lam_logit * F.kl_div(logP_S, P_T, reduction="batchmean")

        loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            student.eval()
            r = ppl(student, probe, attn) / ppl_T
            a = top1_agree(student, teacher, probe, attn)
            g = gen_match(student, teacher, prompt, pattn, args.gen_steps,
                          tok.pad_token_id or 0)
            student.train()
            print(f"  seed {seed} [{cond:8s}] step {step:5d}  loss={loss.item():.4f}"
                  f"  pplR={r:8.3f}  top1={a:.3f}  gen={g:.3f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)

    student.eval()
    s_rep, s_dist = gen_quality(student, prompt, pattn, args.gen_steps)
    final = dict(ppl_ratio=ppl(student, probe, attn) / ppl_T,
                 top1=top1_agree(student, teacher, probe, attn),
                 gen=gen_match(student, teacher, prompt, pattn, args.gen_steps,
                               tok.pad_token_id or 0),
                 rep=s_rep, distinct=s_dist,
                 rollout_kl=rollout_kl(student, teacher, prompt, pattn,
                                       args.gen_steps),
                 teacher_rep=t_rep, teacher_distinct=t_dist)

    # diagnostic: print a short greedy continuation from BOTH teacher and student
    # on the same prompt, so the top1-vs-gen gap can be judged by eye -- is the
    # student producing garbage, or fluent text that merely diverges in tokens?
    if args.show_gen:
        with torch.no_grad():
            p = probe[:1, :8]
            m = attn[:1, :8]
            def greedy(model, seq, mm, n=24):
                seq, mm = seq.clone(), mm.clone()
                for _ in range(n):
                    nx = model(input_ids=seq, attention_mask=mm).logits[:, -1].argmax(-1, keepdim=True)
                    seq = torch.cat([seq, nx], 1); mm = torch.cat([mm, torch.ones_like(nx)], 1)
                return seq
            t_out = tok.decode(greedy(teacher, p, m)[0], skip_special_tokens=True)
            s_out = tok.decode(greedy(student, p, m)[0], skip_special_tokens=True)
        print(f"    [gen sample cond={cond} seed={seed}]", flush=True)
        print(f"      TEACHER: {t_out!r}", flush=True)
        print(f"      STUDENT: {s_out!r}", flush=True)

    del student
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return final


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", choices=["corrupt", "scratch", "pre"], default="corrupt")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--conditions", nargs="+", default=["feature", "logit", "wl"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam_feat", type=float, default=1.0)
    ap.add_argument("--lam_logit", type=float, default=1.0)
    ap.add_argument("--ridge_lam", type=float, default=1e-2)
    ap.add_argument("--n_probe", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=128)
    ap.add_argument("--gen_steps", type=int, default=32)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--teacher", default=TEACHER_ID,
                    help="teacher model id or local path")
    ap.add_argument("--student", default=STUDENT_ID,
                    help="student model id or local path")
    ap.add_argument("--dataset", default=None, help="HF dataset id for probes "
                    "(e.g. 'wikitext') or a local path")
    ap.add_argument("--text_file", default=None,
                    help="local .txt/.jsonl file or dir of them (bypasses "
                    "datasets; the robust offline route)")
    ap.add_argument("--data_cache", default=None,
                    help="datasets cache dir, e.g. /home/models/hub")
    ap.add_argument("--hf_home", default=None,
                    help="local HF cache dir, e.g. /home/models "
                    "(sets HF_HOME so weights/data load offline)")
    ap.add_argument("--offline", action="store_true",
                    help="force HF offline mode (use only local cache)")
    ap.add_argument("--show_gen", action="store_true",
                    help="print teacher vs student greedy continuations")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    # point HF at the local cache and (optionally) forbid network access, so
    # both Qwen weights and wikitext resolve from /home/models/hub
    import os
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
        hub = os.path.join(args.hf_home, "hub")
        # newer huggingface_hub reads HF_HUB_CACHE; older reads
        # HUGGINGFACE_HUB_CACHE. Set both so the local hub/ is found.
        os.environ["HF_HUB_CACHE"] = hub
        os.environ["HUGGINGFACE_HUB_CACHE"] = hub
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  teacher={args.teacher}  student={args.student}",
          flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.teacher)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    from transformers import AutoConfig
    scfg = AutoConfig.from_pretrained(args.student)
    print(f"d_T={teacher.config.hidden_size}  d_S={scfg.hidden_size}  "
          f"V={teacher.config.vocab_size}  "
          f"L_T={teacher.config.num_hidden_layers}  "
          f"L_S={scfg.num_hidden_layers}", flush=True)

    probe = build_probe(tok, args.n_probe, args.seqlen, device, args.dataset,
                        cache_dir=args.data_cache, text_file=args.text_file)
    attn = (probe != tok.pad_token_id).long()
    print(f"probe: {tuple(probe.shape)}   teacher PPL="
          f"{ppl(teacher, probe, attn):.2f}\n", flush=True)

    keys = ("ppl_ratio", "top1", "gen", "rep", "distinct", "rollout_kl")
    results = {c: {k: [] for k in keys} for c in args.conditions}
    teacher_ref = {}
    for seed in range(1, args.seeds + 1):
        for cond in args.conditions:
            fin = run(cond, seed, teacher, tok, probe, attn, args, device)
            for k in keys:
                results[cond][k].append(fin[k])
            teacher_ref = dict(rep=fin["teacher_rep"], distinct=fin["teacher_distinct"])
            print(f"  >>> seed {seed} [{cond}] FINAL  pplR={fin['ppl_ratio']:.3f}"
                  f"  top1={fin['top1']:.3f}  gen={fin['gen']:.3f}"
                  f"  rep={fin['rep']:.3f}  distinct={fin['distinct']:.3f}"
                  f"  rollKL={fin['rollout_kl']:.3f}\n", flush=True)

    def ms(x): return (statistics.mean(x),
                       statistics.pstdev(x) if len(x) > 1 else 0.0)

    print("\n================ SUMMARY (mean +/- std over seeds) ================")
    print(f"  teacher baseline:  rep={teacher_ref.get('rep', 0):.3f}  "
          f"distinct={teacher_ref.get('distinct', 1):.3f}   "
          f"(a healthy student should match these, not 0/1)\n")
    hdr = f"  {'cond':8s} {'PPLratio':>13s} {'top1':>12s} {'gen':>12s} " \
          f"{'rep':>12s} {'distinct':>12s} {'rollKL':>12s}"
    print(hdr)
    for c in args.conditions:
        row = {k: ms(results[c][k]) for k in keys}
        print(f"  {c:8s} "
              f"{row['ppl_ratio'][0]:6.3f}+/-{row['ppl_ratio'][1]:4.3f} "
              f"{row['top1'][0]:.3f}+/-{row['top1'][1]:.3f} "
              f"{row['gen'][0]:.3f}+/-{row['gen'][1]:.3f} "
              f"{row['rep'][0]:.3f}+/-{row['rep'][1]:.3f} "
              f"{row['distinct'][0]:.3f}+/-{row['distinct'][1]:.3f} "
              f"{row['rollout_kl'][0]:.3f}+/-{row['rollout_kl'][1]:.3f}")

    if "logit" in args.conditions and "wl" in args.conditions:
        print("\n  (ii) vs (iii): is adding W measurable, or seed noise?")
        for k in ("gen", "rep", "rollout_kl"):
            d = statistics.mean(results["wl"][k]) - statistics.mean(results["logit"][k])
            pooled = (ms(results["logit"][k])[1] + ms(results["wl"][k])[1]) / 2 or 1e-9
            verd = "within noise" if abs(d) < pooled else "EXCEEDS noise"
            print(f"    {k:10s} diff(wl-logit)={d:+.3f}  pooled_std~{pooled:.3f}  => {verd}")
    print("\n  Reading: top1 high + rep near teacher_rep + low rollKL => capability")
    print("  AND generation restored. High rep (loops) or high rollKL => the")
    print("  teacher-forced top1 masks a generation deficit.")

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\n  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
