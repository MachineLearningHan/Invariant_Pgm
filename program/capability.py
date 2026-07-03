# PAPER: Results 1,2,5 -- FUNCTION metrics (logit_kl / ppl / top1) beyond CKA. See RESULTS_TO_CODE.md
"""
Capability-restoration metrics (beyond representational CKA).

CKA says the representations are geometrically aligned; it does NOT say the
model behaves the same. These metrics test whether the FUNCTION is restored:

  logit_kl      : mean KL(teacher || student) over next-token distributions.
                  -> 0 means same predictive distribution (the logit-distill
                     level of the paper; basis-invariant by construction).
  perplexity    : student vs teacher PPL on held-out text. student PPL
                  approaching teacher PPL = language-model capability restored.
  top1_agree    : fraction of positions where argmax next-token matches.
                  the most direct "same behavior" check.

These are eval-only (no grad). Pass the same in-domain / out-domain probe or
text sets used for CKA so the capability picture lines up with the geometry.
"""
import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def capability_metrics(student, teacher, tok, texts, device, max_len=64,
                       tag=""):
    student.eval(); teacher.eval()
    kl_sum = 0.0
    tok_count = 0
    agree = 0
    nll_s = 0.0
    nll_t = 0.0
    nll_tokens = 0
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True,
                  max_length=max_len).to(device)
        ids = enc["input_ids"]
        if ids.size(1) < 2:
            continue
        s_logits = student(**enc).logits[0]      # (seq, V)
        t_logits = teacher(**enc).logits[0]
        # next-token positions: predict ids[1:] from logits[:-1]
        s_lp = F.log_softmax(s_logits[:-1], dim=-1)
        t_lp = F.log_softmax(t_logits[:-1], dim=-1)
        t_p = t_lp.exp()
        # KL(teacher || student) = sum t_p * (t_lp - s_lp)
        kl = (t_p * (t_lp - s_lp)).sum(-1)        # (seq-1,)
        kl_sum += kl.sum().item()
        tok_count += kl.numel()
        # top-1 agreement
        agree += (s_logits[:-1].argmax(-1) ==
                  t_logits[:-1].argmax(-1)).sum().item()
        # teacher-forced NLL of the actual tokens (perplexity)
        tgt = ids[0, 1:]
        nll_s += F.nll_loss(s_lp, tgt, reduction="sum").item()
        nll_t += F.nll_loss(t_lp, tgt, reduction="sum").item()
        nll_tokens += tgt.numel()

    kl = kl_sum / max(tok_count, 1)
    ppl_s = math.exp(nll_s / max(nll_tokens, 1))
    ppl_t = math.exp(nll_t / max(nll_tokens, 1))
    a = agree / max(tok_count, 1)
    print(f"    [cap {tag}] KL(t||s)={kl:.4f}  "
          f"PPL student={ppl_s:.2f} teacher={ppl_t:.2f} "
          f"(ratio {ppl_s/max(ppl_t,1e-9):.2f}x)  top1_agree={a:.3f}",
          flush=True)
    return {"kl": kl, "ppl_student": ppl_s, "ppl_teacher": ppl_t,
            "top1_agree": a}
