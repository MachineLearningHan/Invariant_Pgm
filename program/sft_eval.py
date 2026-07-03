# PAPER: Discussion support -- does restoration help downstream SFT reach the teacher ceiling? See RESULTS_TO_CODE.md
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
    ds = load_dataset("tatsu-lab/alpaca", split="train")
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


# --------------------------- main ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--strength", type=float, default=0.4)
    ap.add_argument("--restore_steps", type=int, default=300)
    ap.add_argument("--sft_steps", type=int, default=200)
    ap.add_argument("--n_restore", type=int, default=2000)
    ap.add_argument("--n_sft", type=int, default=1000)
    ap.add_argument("--n_eval", type=int, default=200)
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

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def fresh():
        return AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dt, output_hidden_states=True).to(dev)

    print("[data] building Alpaca 3-way split of PROMPTS (disjoint)",
          flush=True)
    restore_prompts, sft_prompts, eval_prompts = build_alpaca_prompts(
        tok, args.n_restore, args.n_sft, args.n_eval, args.max_len)
    print(f"[data] restore={len(restore_prompts)} sft={len(sft_prompts)} "
          f"eval={len(eval_prompts)} (eval is OUT-OF-SAMPLE)", flush=True)

    teacher = fresh().eval()

    # teacher generates the answers -> ALL supervision is the teacher's output
    print("\n[teacher-gen] generating answers for SFT and eval splits",
          flush=True)
    sft_seqs = teacher_label(teacher, tok, sft_prompts, dev,
                             max_new=args.max_new)
    eval_seqs = teacher_label(teacher, tok, eval_prompts, dev,
                              max_new=args.max_new)
    # restoration corpus: use prompt+teacher-answer as plain text (full-seq CKA)
    restore_seqs_pl = teacher_label(teacher, tok, restore_prompts, dev,
                                    max_new=args.max_new)
    restore_ids = [ids for ids, _ in restore_seqs_pl]

    results = {}

    # ---- A: teacher -> SFT on its own answers (ceiling; should be ~perfect) ----
    print("\n=== Condition A: teacher -> SFT (on teacher answers) ===",
          flush=True)
    A = copy.deepcopy(teacher)
    A = sft(A, tok, sft_seqs, args, dev, "A")
    results["A_teacher_sft"] = eval_ppl(A, tok, eval_seqs, args, dev)
    del A; torch.cuda.empty_cache()

    # ---- corrupt once, reuse for B and C ----
    corrupted, corrupted_layers = corrupt_model(
        teacher, strength=args.strength, mode="reinit", seed=0)
    print(f"\n[corrupt] layers {corrupted_layers}", flush=True)

    # ---- B: corrupt -> restore -> SFT (ours) ----
    print("\n=== Condition B: corrupt -> restore -> SFT ===", flush=True)
    B = copy.deepcopy(corrupted).to(dev)
    B = restore(B, teacher, tok, corrupted_layers, restore_ids, args, dev)
    print("  [B] restored CKA check:", flush=True)
    B = sft(B, tok, sft_seqs, args, dev, "B")
    results["B_restored_sft"] = eval_ppl(B, tok, eval_seqs, args, dev)
    del B; torch.cuda.empty_cache()

    # ---- C: corrupt -> SFT (no restore) ----
    print("\n=== Condition C: corrupt -> SFT (no restore) ===", flush=True)
    C = copy.deepcopy(corrupted).to(dev)
    C = sft(C, tok, sft_seqs, args, dev, "C")
    results["C_corrupted_sft"] = eval_ppl(C, tok, eval_seqs, args, dev)
    del C; torch.cuda.empty_cache()

    # ---- report ----
    base = results["A_teacher_sft"]
    print("\n=== RESULTS: held-out Alpaca PPL (teacher-relative) ===",
          flush=True)
    for k in ["A_teacher_sft", "B_restored_sft", "C_corrupted_sft"]:
        print(f"  {k:20s} PPL={results[k]:8.3f}  "
              f"ratio_to_teacher={results[k]/base:6.3f}x", flush=True)
    print("\n  Interpretation: B close to 1.0 and C >> 1.0 means basis-invariant"
          "\n  restoration recovers a good SFT starting point.", flush=True)


if __name__ == "__main__":
    main()
