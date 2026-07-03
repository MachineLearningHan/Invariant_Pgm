# PAPER: Result 8 -- wikitext-103 val PPL for the tab:fmonly checkpoints. See RESULTS_TO_CODE.md
"""
Evaluate student checkpoints on wikitext-103 validation perplexity.

Loads the base student architecture, applies each saved state_dict, and computes
token-weighted perplexity over the SAME held-out text under identical conditions.
Teacher is not needed (pure LM ability).

Usage:
  python -u eval_ppl.py --student Qwen/Qwen2.5-0.5B --seqlen 512 \
    --ckpts student_logit.pt student_fm.pt --min-chars 200 --max-rows 2000
"""
import argparse, os, math, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


@torch.no_grad()
def perplexity(model, tok, texts, seqlen, dev):
    """Token-weighted PPL: sum(NLL over real target tokens) / sum(tokens)."""
    total_nll = 0.0
    total_tok = 0
    for i, t in enumerate(texts):
        enc = tok(t, truncation=True, max_length=seqlen, return_tensors="pt")
        ids = enc["input_ids"].to(dev)
        if ids.size(1) < 2:
            continue
        out = model(ids)
        # shift: predict token t+1 from t
        logits = out.logits[:, :-1, :]
        target = ids[:, 1:]
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            reduction="sum",
        )
        ntok = target.numel()
        total_nll += nll.item()
        total_tok += ntok
        if (i + 1) % 200 == 0:
            cur = math.exp(total_nll / total_tok)
            print(f"    [{i+1}/{len(texts)}] running ppl={cur:.3f}", flush=True)
    return math.exp(total_nll / total_tok), total_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B",
                    help="base arch / tokenizer (local path or repo id)")
    ap.add_argument("--ckpts", nargs="*", default=[],
                    help="state_dict .pt files to evaluate")
    ap.add_argument("--base-only", action="store_true",
                    help="evaluate the untrained base model (no checkpoint)")
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-rows", type=int, default=2000)
    ap.add_argument("--parquet", default=None,
                    help="path to a wikitext validation parquet (bypasses offline hub lookup)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if dev == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(args.student)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # identical held-out set for every checkpoint
    if args.parquet:
        ds = load_dataset("parquet", data_files=args.parquet, split="train")
    else:
        ds = load_dataset("parquet", data_files="/home/models/hub/datasets--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1/validation-00000-of-00001.parquet", split="train")
    texts = []
    for r in ds:
        s = r["text"].strip()
        if len(s) < args.min_chars:
            continue
        if s.startswith("=") and s.endswith("="):
            continue
        texts.append(s)
        if len(texts) >= args.max_rows:
            break
    print(f"held-out rows: {len(texts)} (seqlen={args.seqlen})", flush=True)

    results = {}
    eval_targets = list(args.ckpts)
    if args.base_only or not eval_targets:
        eval_targets = ["__BASE__"] + eval_targets

    for ck in eval_targets:
        label = "base (untrained)" if ck == "__BASE__" else ck
        print(f"\n=== {label} ===", flush=True)
        model = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=dt).to(dev).eval()
        if ck != "__BASE__":
            sd = torch.load(ck, map_location=dev)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                print(f"  [warn] missing keys: {len(missing)} (e.g. {missing[:3]})", flush=True)
            if unexpected:
                print(f"  [warn] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})", flush=True)
        ppl, ntok = perplexity(model, tok, texts, args.seqlen, dev)
        results[label] = ppl
        print(f"  -> PPL = {ppl:.4f}  (over {ntok} tokens)", flush=True)
        del model
        torch.cuda.empty_cache()

    print("\n===== SUMMARY =====", flush=True)
    best = min(results, key=results.get)
    for label, ppl in sorted(results.items(), key=lambda kv: kv[1]):
        tag = "  <-- best" if label == best else ""
        print(f"  {label:<24} PPL={ppl:.4f}{tag}", flush=True)
    base = results.get("base (untrained)")
    if base is not None:
        for label, ppl in results.items():
            if label == "base (untrained)":
                continue
            rel = (ppl - base) / base * 100
            arrow = "worse" if rel > 0 else "better"
            print(f"  {label:<24} vs base: {rel:+.2f}% ({arrow})", flush=True)


if __name__ == "__main__":
    main()
