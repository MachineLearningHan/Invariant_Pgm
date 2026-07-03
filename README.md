# Experiments ↔ Code map

Maps every result and table in **Section 7 (Experiments)** of `Invariant8.tex`
— plus the two supporting theory sections it leans on — to the script that
produces it. Each `.py` file also carries a matching `# PAPER:` comment at the
top pointing back here.

Legend: **T** = teacher, **S** = student. All restoration runs corrupt `T`,
retrain `S` with a representation loss, and measure recovery.

---

## Theory checks that Section 7 cites (static / pre-Result)

| Paper object | What it establishes | Code |
|---|---|---|
| Prop 2.1 (`prop:illposed`), **Table 1** (`tab:rotation`) | Absolute feature matching is ill-posed: a function-preserving rotation `H→HQ` inflates `‖H−HQ‖²` while `1−CKA` and `L_rel` stay 0. | `rotation_proof.py` (loss-level), `rotation_train.py` (training-level consequence on the function) |
| Prop 3.1–3.2, orbit/fiber/quotient, **Fig. 2** (`fig:orbit`) | The output function is the joint-quotient invariant; capability lives there, not on the representation quotient. | `orbit_fiber_quotient.py`, `orbit_fiber_quotient_explained.py`; the figure asset is drawn by `make_fig2.py` |
| MLP appendix, **Table 8** (`tab:mlp`) | The claims hold exactly on a normalization-free 2-layer MLP (no RMSNorm gain to break exactness). | `minimal_mlp.py` |
| Loss definitions (`tab:losses`), §4.1 | `L_rel`, `L_cka` (basis-invariant) vs `L_abs` (ill-posed control). | `losses.py` (differentiable), `invariants.py` (numpy invariants), `demo_losses.py` (toy illustration), `test_invariants.py` (self-tests) |

---

## Section 7 results

| Result | Claim | Table(s) | Code |
|---|---|---|---|
| **Result 1** — representation restored, function not | Basis-invariant loss drives `CKA→~1` from a corrupted start, yet KL/PPL/top-1 show the function is still wrong. | `tab:cap` (left) | `restore.py` (CKA restoration) + `capability.py` (logit_kl / ppl / top1) |
| **Result 2** — capability is driven by the logit term | Ablation on all-layers-reinit: `L_cka` only → collapse; `L_logit` only → recovered; both → recovered. Logit term must be **full-sequence**. | `tab:ablation`, `tab:cap` (right) | `restore.py` (adds logit term) + `capability.py` (function metrics) |
| **Result 3** — restoration confined to the corpus subspace | In-domain restores far more than out-domain; mixing out-domain text into the corpus closes the gap (`CKA 0.53→0.996`). | `tab:cap` | `restore.py` (in_domain vs out_domain probe sets, corpus mixing) |
| **Result 4** — the `L_abs` control degrades under stronger corruption | `L_abs` matches `L_cka` when surviving layers preserve the frame, but the basis-invariant advantage grows as corruption destroys the frame. | (in-text, from `tab:cap` sweep) | `restore.py --kind abs` vs `--kind cka` across strengths |
| **Result 5** — scale, and the role of weight tying | Tied 0.5B: `L_logit` suffices (head = embedding, auto-fixed). Untied 7B/8B: the **output head must be trained** to select the representative. | `tab:models` | `restore.py` (7B/8B, head in trainable set) + `capability.py` (head-frozen vs head-trained ablation) |
| **Result 6** — decoder-from-scratch reconstruction | Re-initialize **all** decoder layers, keep only embedding/unembedding; distillation rebuilds the decoder when the corpus covers the probe domains. | (in-text) | `restore.py` (all-layers reinit mode) + `capability.py` |
| **Result 7** — cross-width restoration + teacher-forced/generation gap | `T=1.5B → S=0.5B`, unequal width. (ii) logit only vs (iii) W-then-logit: W buys nothing; teacher-forced top-1 ≈0.98 while free-running generation drifts (rollout KL ≈1.25 nats). | (in-text) | `qwen_crosswidth.py` (real Qwen, 3 conditions + generation metrics); logic pre-validated by `surrogate_graft.py` |
| **Result 8** — feature-only collapse under genuine distillation | Pristine `S=0.5B`, `T=1.5B`, CE-free. (i) `L_fm` only → PPL >1e6 (collapse); (ii) `L_logit` only → 20.77; (iii) `L_fm`+`L_logit` → 20.94 (mildly worse). | `tab:fmonly` (real, 3 seeds), `tab:surrogate` (5-seed controlled) | `train_kd.py` (`--mode logit` / `--mode fm` / `--fm_only`) + `eval_ppl.py` (wikitext PPL) + `seed_sweep_fmonly.sh` (seeds 42/123/7) + `surrogate_multiseed.py` (`tab:surrogate`) |

---

## Section 5 (graft-success prediction) — cited by Result 7 context

| Paper object | Code |
|---|---|
| **Prediction 6.1** (`pred:mono`, Monotonicity): success rises monotonically with overlap (CKA / ρ) | `graft_experiment.py` (overlap spectrum), `invariants.py` (CKA / ρ / k_eff), `analyze_grafts.py` (regression) |
| **Prediction 6.2** (`pred:mediate`, Domain effect mediated by overlap): domain match → CKA → success | `analyze_grafts.py` (mediation steps), `extract.py` (donor output-space repr via API logprobs; host hidden states) |
| **Prediction 6.3** (`pred:align`, Alignment helps most under mismatch): low-capacity `W` helps when overlap is low | `graft_experiment.py` (W ablation, overlap×alignment interaction), `analyze_grafts.py` |

## Downstream SFT (Discussion support, not a numbered Result)

| Question | Code |
|---|---|
| Does basis-invariant + logit restoration help subsequent SFT approach the teacher ceiling? | `sft_eval.py` (single config), `sft_sweep.py` (sweep) |

## Utilities

| File | Role |
|---|---|
| `corrupt.py` | Corruption utility (reinit / noise on middle decoder layers) used by all restoration drivers. |
| `check_local_cache.py` | Verify offline HF cache + emit the exact `load_dataset` call before launching `qwen_crosswidth.py`. |
| `corpus.jsonl`, `corpus.txt` | WikiText-derived training/probe text for the distillation and restoration runs. |
| `fig2_quotient.pdf` | Figure 2 asset (produced by `make_fig2.py`). |


## Not code-backed (no omission)

`fig:spine` (the Prop/Result spine diagram) and `fig:summary` (the one-figure
overview) are pure TikZ concept diagrams drawn inline in the `.tex`; they present
no data and therefore have no accompanying experiment script. `make_fig2.py`
(the only figure with a rendered asset) is not in this archive's core experiment
set but is referenced above for completeness.
