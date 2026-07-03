# PAPER: Loss defs (tab:losses), Sec 4.1 -- L_rel/L_cka (well-posed) vs L_abs (ill-posed). See RESULTS_TO_CODE.md
"""
Differentiable representation-matching losses for basis-invariant restoration.

L_rel  (basis-invariant): match the CENTERED GRAM structure of student to
        teacher. Invariant to orthogonal reparametrization of either side
        (paper Sec 4.1) -> well-posed.
L_cka  (basis-invariant): 1 - linear CKA; same invariance, scale-normalized.
L_abs  (control, ILL-POSED per paper Prop 2.1): raw coordinate matching
        ||H_S - H_T||_F^2. Included only to demonstrate it fails / is unstable.

All operate on a batch of token representations H of shape (B, d), where B is
the number of pooled tokens in the minibatch (one vector per sequence, or per
token if you flatten). Teacher tensors must be detached (no grad).

These are torch functions; this file is import-safe without torch only for
syntax checking. Numerical correctness is cross-checked against numpy in
test_losses.py (same formulas).
"""


def _center(H):
    return H - H.mean(dim=0, keepdim=True)


def gram_centered(H):
    Hc = _center(H)
    return Hc @ Hc.t()                      # (B, B)


def l_rel(H_s, H_t):
    """
    Normalized centered-Gram matching (paper L_rel, eq. in Sec 4.1):

        L_rel = || Gs/||Gs||_F  -  Gt/||Gt||_F ||_F^2

    Each centered Gram is divided by ITS OWN Frobenius norm before differencing.
    This is what makes the loss invariant to isotropic rescaling of EITHER side
    (H -> cH scales G by c^2, which cancels in G/||G||_F). A one-sided
    normalization (dividing only by ||Gt||) is NOT student-scale invariant and
    would break the basis-/scale-invariance the paper relies on. Teacher detached
    by caller.
    """
    Gs = gram_centered(H_s)
    Gt = gram_centered(H_t)
    Gs = Gs / Gs.norm(p="fro").clamp_min(1e-12)
    Gt = Gt / Gt.norm(p="fro").clamp_min(1e-12)
    return ((Gs - Gt) ** 2).sum()


def l_cka(H_s, H_t):
    """1 - linear CKA. Basis- and isotropic-scale invariant; in [0, 1]."""
    Gs = gram_centered(H_s)
    Gt = gram_centered(H_t)
    hsic_st = (Gs * Gt).sum()
    hsic_ss = (Gs * Gs).sum().clamp_min(1e-12)
    hsic_tt = (Gt * Gt).sum().clamp_min(1e-12)
    cka = hsic_st / (hsic_ss.sqrt() * hsic_tt.sqrt())
    return 1.0 - cka


def l_abs(H_s, H_t):
    """Raw coordinate matching -- ILL-POSED control (paper Prop 2.1)."""
    return ((H_s - H_t) ** 2).mean()


def multilayer_loss(hs_student, hs_teacher, kind="rel", layers=None,
                    weights=None):
    """
    Sum a per-layer representation loss across selected layers.

    hs_student, hs_teacher : tuples/lists of (B, d) tensors, one per layer
                             (e.g. model output.hidden_states). Teacher must be
                             detached by the caller.
    kind   : "rel" | "cka" | "abs"
    layers : iterable of layer indices to match (default: all)
    weights: optional per-layer weights (default: uniform)
    """
    fn = {"rel": l_rel, "cka": l_cka, "abs": l_abs}[kind]
    L = len(hs_student)
    idxs = list(range(L)) if layers is None else list(layers)
    if weights is None:
        weights = [1.0] * len(idxs)
    total = None
    for w, i in zip(weights, idxs):
        term = w * fn(hs_student[i], hs_teacher[i])
        total = term if total is None else total + term
    return total / max(len(idxs), 1)
