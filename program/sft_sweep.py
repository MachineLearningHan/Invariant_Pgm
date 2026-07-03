# PAPER: Discussion support -- SFT-ceiling sweep (restored vs corrupted vs teacher). See RESULTS_TO_CODE.md
"""
Does restoration help downstream SFT? Teacher-relative evaluation.

Question: if we corrupt a model, restore it with a basis-invariant + logit
signal, and THEN fine-tune (SFT) on instructions, does it approach what the
*teacher* reaches under the same SFT? The teacher is the ceiling; we ask how
close each condition gets.

Conditions (all SFT'd identically on a small Alpaca subset, then evaluated by
held-out PPL):
  A  teacher           : T -> SFT                      (ceiling)
  B  restored          : corrupt -> restore(cka+logit) -> SFT   (ours)
  C  corrupted         : corrupt -> (no restore) -> SFT         (control)

Key number: PPL(condition) / PPL(A).  B close to 1 and C far from 1 would show
that basis-invariant restoration recovers a good SFT starting point.

Short SFT on purpose: if SFT is long, all conditions converge and the effect
of the starting point vanishes. We want to see whether restoration gives a
better starting point that survives a brief SFT.

Live per-step / per-condition output, flushed.

Run (server, unsloth-cu13):
  python -u sft_eval.py --model Qwen/Qwen2.5-0.5B \
     --restore_steps 300 --sft_steps 200 --strength 0.4 --bf16
"""
import argparse, copy, math, random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from corrupt import corrupt_model


# --------------------------- data ---------------------------
ALPACA_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instr}\n\n### Response:\n"
)

def build_alpaca_prompts(tok, n_restore, n_sft, n_eval, max_len, seed=0):
    """3-way DISJOINT split. Returns PROMPTS only (instruction up to the
    'Response:' marker). The teacher will generate the answers, so that all
    SFT/eval supervision is the teacher's own output, not the dataset label."""
    ds = __import__("datasets").load_dataset(
        "parquet",
        data_dir=__import__("glob").glob(__import__("os").path.expanduser(
            "~/models/hub/datasets--tatsu-lab--alpaca/snapshots/*/data"))[0],
        split="train")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    a = n_restore; b = a + n_sft; c = b + n_eval
    r_idx, s_idx, e_idx = idx[:a], idx[a:b], idx[b:c]   # disjoint
    def prompt(i):
        r = ds[i]
        instr = r["instruction"] + ("\n" + r["input"] if r["input"] else "")
        return ALPACA_PROMPT.format(instr=instr)
    return ([prompt(i) for i in r_idx],
            [prompt(i) for i in s_idx],
            [prompt(i) for i in e_idx])


@torch.no_grad()
def teacher_label(teacher, tok, prompts, dev, max_new=96, max_prompt=160):
    """For each prompt, greedily generate the teacher's answer and build the
    full SFT sequence (prompt + answer). Returns list of (input_ids,
    prompt_len) so the SFT loss can mask the prompt and train only the answer
    (i.e. learn to reproduce the TEACHER's answer)."""
    teacher.eval()
    seqs = []
    for k, p in enumerate(prompts):
        enc = tok(p, return_tensors="pt", truncation=True,
                  max_length=max_prompt).to(dev)
        plen = enc["input_ids"].size(1)
        gen = teacher.generate(**enc, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        ids = gen[0].detach().cpu()
        seqs.append((ids, plen))
        if (k + 1) % 200 == 0:
            print(f"    [teacher-gen {k+1}/{len(prompts)}]", flush=True)
    return seqs


def batches(seqs, tok, bs, device, shuffle=True, seed=0):
    """seqs: list of plain input_ids tensors (used for the restoration corpus).
    Yields (ids, att)."""
    order = list(range(len(seqs)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for k in range(0, len(order), bs):
        chunk = [seqs[i] for i in order[k:k + bs]]
        m = max(x.size(0) for x in chunk)
        pad = tok.pad_token_id
        ids = torch.full((len(chunk), m), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), m), dtype=torch.long)
        for j, x in enumerate(chunk):
            ids[j, :x.size(0)] = x
            att[j, :x.size(0)] = 1
        yield ids.to(device), att.to(device)


def label_batches(seqs, tok, bs, device, shuffle=True, seed=0):
    """seqs: list of (input_ids, prompt_len). Yields (ids, att, labels) where
    labels mask BOTH padding AND the prompt, so the loss is computed only on the
    teacher's answer tokens (learning to reproduce the teacher's answer)."""
    order = list(range(len(seqs)))
    if shuffle:
        random.Random(seed).shuffle(order)
    pad = tok.pad_token_id
    for k in range(0, len(order), bs):
        chunk = [seqs[i] for i in order[k:k + bs]]
        m = max(x.size(0) for x, _ in chunk)
        ids = torch.full((len(chunk), m), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), m), dtype=torch.long)
        lab = torch.full((len(chunk), m), -100, dtype=torch.long)
        for j, (x, plen) in enumerate(chunk):
            L = x.size(0)
            ids[j, :L] = x
            att[j, :L] = 1
            # labels only on answer span [plen:L]
            if L > plen:
                lab[j, plen:L] = x[plen:L]
        yield ids.to(device), att.to(device), lab.to(device)


# --------------------------- restoration ---------------------------
def pooled_all_layers(model, ids, att):
    out = model(input_ids=ids, attention_mask=att, output_hidden_states=True)
    last = att.sum(1) - 1
    hs = out.hidden_states  # tuple (L+1)
    pooled = [h[torch.arange(h.size(0)), last] for h in hs]
    return pooled, out.logits, last


def linear_cka_t(A, B):
    A = A - A.mean(0, keepdim=True); B = B - B.mean(0, keepdim=True)
    ga, gb = A @ A.t(), B @ B.t()
    return (ga * gb).sum() / (ga.norm() * gb.norm() + 1e-9)


def restore(student, teacher, tok, corrupted_layers, train_ids, args, dev):
    """basis-invariant (CKA) + full-seq logit KD on the corrupted layers + head."""
    for p in student.parameters():
        p.requires_grad_(False)
    sl = student.model.layers
    train_params = []
    for li in corrupted_layers:
        for p in sl[li].parameters():
            p.requires_grad_(True); train_params.append(p)
    if hasattr(student, "lm_head"):
        for p in student.lm_head.parameters():
            p.requires_grad_(True); train_params.append(p)
    for li in [len(sl) - 2, len(sl) - 1]:
        for p in sl[li].parameters():
            if not p.requires_grad:
                p.requires_grad_(True); train_params.append(p)
    opt = torch.optim.AdamW(train_params, lr=args.lr)
    student.train(); teacher.eval()
    step = 0
    while step < args.restore_steps:
        for ids, att in batches(train_ids, tok, args.bs, dev, seed=step):
            step += 1
            if step > args.restore_steps:
                break
            with torch.no_grad():
                tp, tl, _ = pooled_all_layers(teacher, ids, att)
            sp, slg, _ = pooled_all_layers(student, ids, att)
            rep = sum(1 - linear_cka_t(sp[i], tp[i].detach())
                      for i in range(1, len(sp))) / (len(sp) - 1)
            m = att.bool()
            tpd = F.softmax(tl.detach()[m], -1)
            lg = (tpd * (F.log_softmax(tl.detach()[m], -1)
                         - F.log_softmax(slg[m], -1))).sum(-1).mean()
            loss = rep + args.logit_w * lg
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 50 == 0 or step == 1:
                print(f"    [restore {step:3d}] rep={rep.item():.4f} "
                      f"logit={lg.item():.4f}", flush=True)
    return student


def sft_epochs(model, tok, train_seqs, args, dev, tag, n_epochs=1):
    """SFT for an exact number of EPOCHS over train_seqs (teacher-answer tokens
    only). Epoch-based so that scaling data size = scaling supervision, without
    the overfitting that comes from re-looping a fixed small set many times."""
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.sft_lr)
    model.train()
    gstep = 0
    for ep in range(n_epochs):
        for ids, att, lab in label_batches(train_seqs, tok, args.bs, dev,
                                           seed=1000 + ep):
            gstep += 1
            out = model(input_ids=ids, attention_mask=att)
            logits = out.logits[:, :-1]
            labels = lab[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
            if gstep % 100 == 0:
                print(f"    [sft-{tag} ep{ep} step{gstep}] "
                      f"loss={loss.item():.4f}", flush=True)
    return model


# --------------------------- SFT ---------------------------
def sft(model, tok, train_seqs, args, dev, tag):
    """train_seqs: list of (ids, prompt_len). Trains on teacher-answer tokens
    only (prompt is masked)."""
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.sft_lr)
    model.train()
    step = 0
    while step < args.sft_steps:
        for ids, att, lab in label_batches(train_seqs, tok, args.bs, dev,
                                           seed=1000 + step):
            step += 1
            if step > args.sft_steps:
                break
            out = model(input_ids=ids, attention_mask=att)
            logits = out.logits[:, :-1]
            labels = lab[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 50 == 0 or step == 1:
                print(f"    [sft-{tag} {step:3d}] loss={loss.item():.4f}",
                      flush=True)
    return model


@torch.no_grad()
def eval_ppl(model, tok, eval_seqs, args, dev):
    """PPL on teacher-answer tokens only (prompt masked)."""
    model.eval()
    nll = ntok = 0
    for ids, att, lab in label_batches(eval_seqs, tok, args.bs, dev,
                                       shuffle=False):
        out = model(input_ids=ids, attention_mask=att)
        logits = out.logits[:, :-1]
        labels = lab[:, 1:]
        l = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                            labels.reshape(-1), ignore_index=-100,
                            reduction="sum")
        nll += l.item(); ntok += (labels != -100).sum().item()
    return math.exp(nll / max(ntok, 1))


# --------------------------- main (data-scale sweep) ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--strength", type=float, default=0.4)
    ap.add_argument("--restore_steps", type=int, default=300)
    ap.add_argument("--n_restore", type=int, default=1000)
    ap.add_argument("--n_sft_max", type=int, default=5000)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--sizes", default="250,500,1000,2500,5000")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sft_lr", type=float, default=2e-5)
    ap.add_argument("--logit_w", type=float, default=1.0)
    ap.add_argument("--max_new", type=int, default=96)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()
    dev = args.device
    dt = torch.bfloat16 if args.bf16 else torch.float32
    sizes = [int(s) for s in args.sizes.split(",")]

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def fresh():
        return AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dt, output_hidden_states=True).to(dev)

    print("[data] building Alpaca prompts (disjoint restore/sft/eval)",
          flush=True)
    restore_prompts, sft_prompts, eval_prompts = build_alpaca_prompts(
        tok, args.n_restore, args.n_sft_max, args.n_eval, args.max_len)
    print(f"[data] restore={len(restore_prompts)} sft_pool={len(sft_prompts)} "
          f"eval={len(eval_prompts)}", flush=True)

    teacher = fresh().eval()

    # teacher generates ALL answers once (max pool); subsets reused per size
    print("\n[teacher-gen] generating teacher answers (once)", flush=True)
    sft_pool = teacher_label(teacher, tok, sft_prompts, dev, max_new=args.max_new)
    eval_seqs = teacher_label(teacher, tok, eval_prompts, dev, max_new=args.max_new)
    restore_seqs_pl = teacher_label(teacher, tok, restore_prompts, dev,
                                    max_new=args.max_new)
    restore_ids = [ids for ids, _ in restore_seqs_pl]

    # corrupt once; restore once (reused across all SFT sizes)
    corrupted, corrupted_layers = corrupt_model(
        teacher, strength=args.strength, mode="reinit", seed=0)
    print(f"\n[corrupt] layers {corrupted_layers}", flush=True)
    print("[restore] restoring once (reused for all sizes)", flush=True)
    restored = copy.deepcopy(corrupted).to(dev)
    restored = restore(restored, teacher, tok, corrupted_layers,
                       restore_ids, args, dev)

    # ceiling PPL (teacher itself on held-out teacher answers, no SFT)
    teacher_self_ppl = eval_ppl(teacher, tok, eval_seqs, args, dev)
    print(f"\n[ceiling] teacher self-PPL on its own answers = "
          f"{teacher_self_ppl:.3f}", flush=True)

    print("\n=== SWEEP: SFT data size vs teacher-replication PPL ===",
          flush=True)
    print(f"{'size':>6} | {'A_teacher':>10} {'B_restored':>11} "
          f"{'C_corrupted':>12} | {'B/A':>6} {'C/A':>6}", flush=True)
    rows = []
    for n in sizes:
        subset = sft_pool[:n]
        # A: teacher -> SFT
        A = copy.deepcopy(teacher)
        A = sft_epochs(A, tok, subset, args, dev, f"A{n}", args.epochs)
        pA = eval_ppl(A, tok, eval_seqs, args, dev); del A; torch.cuda.empty_cache()
        # B: restored -> SFT
        B = copy.deepcopy(restored)
        B = sft_epochs(B, tok, subset, args, dev, f"B{n}", args.epochs)
        pB = eval_ppl(B, tok, eval_seqs, args, dev); del B; torch.cuda.empty_cache()
        # C: corrupted -> SFT
        C = copy.deepcopy(corrupted).to(dev)
        C = sft_epochs(C, tok, subset, args, dev, f"C{n}", args.epochs)
        pC = eval_ppl(C, tok, eval_seqs, args, dev); del C; torch.cuda.empty_cache()
        rows.append((n, pA, pB, pC))
        print(f"{n:>6} | {pA:>10.3f} {pB:>11.3f} {pC:>12.3f} | "
              f"{pB/pA:>6.2f} {pC/pA:>6.2f}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    print(f"teacher self-PPL (no SFT) = {teacher_self_ppl:.3f}", flush=True)
    print("size, A, B, C:", flush=True)
    for n, pA, pB, pC in rows:
        print(f"  {n:>6}: A={pA:.3f} B={pB:.3f} C={pC:.3f} "
              f"(B-A={pB-pA:+.3f}, C-B={pC-pB:+.3f})", flush=True)
    print("\nRead: does B approach A as data grows? does B stay below C "
          "(restoration = data efficiency) or do they converge?", flush=True)


if __name__ == "__main__":
    main()
