# PAPER: Basis-invariant invariants, Sec 2-5 (CKA, rho, k_eff). Used by Results 1-7 & Sec 5. See RESULTS_TO_CODE.md
"""
Basis-invariant representation invariants (paper Sections 2-5).

All functions take representation matrices H of shape (N, d): N probe samples,
d hidden dims. Different arguments may have different d. Everything here is
invariant to orthogonal reparametrization (and, where noted, isotropic scale),
so it is well-defined on the equivalence class [H] = {H Q Λ}.

No torch dependency: pass in numpy arrays. Convert hidden states to numpy
once at extraction time.
"""
import numpy as np


# ----------------------------------------------------------------------
# centering / gram
# ----------------------------------------------------------------------
def center_rows(H):
    """Subtract the mean sample (center the N points)."""
    return H - H.mean(axis=0, keepdims=True)


def gram(H, centered=True):
    """N x N Gram matrix H H^T (centered by default). Orthogonal-invariant."""
    Hc = center_rows(H) if centered else H
    return Hc @ Hc.T


# ----------------------------------------------------------------------
# CKA (linear) -- paper Def 3.3 / Cor 3.1
# ----------------------------------------------------------------------
def linear_cka(H_a, H_b):
    """
    Centered Kernel Alignment with linear kernel, in [0, 1].
    Invariant to orthogonal transform and isotropic scaling of either arg.
    Uses the HSIC form; computed via Gram matrices (works for d_a != d_b).

    NOTE: with finite N the CKA of *independent* reps is not 0 but a positive
    baseline ~ O(d/N). Always compare against cka_null (permutation baseline)
    rather than against 0.
    """
    Ga = gram(H_a, centered=True)
    Gb = gram(H_b, centered=True)
    hsic_ab = np.sum(Ga * Gb)
    hsic_aa = np.sum(Ga * Ga)
    hsic_bb = np.sum(Gb * Gb)
    denom = np.sqrt(hsic_aa * hsic_bb)
    if denom == 0:
        return 0.0
    return float(hsic_ab / denom)


def cka_null(H_a, H_b, n_perm=200, seed=0):
    """
    Permutation null for CKA: shuffle the row-correspondence of H_b so the two
    reps describe unrelated point sets, recompute CKA. Returns (mean, std) of
    the null. An observed CKA is 'real overlap' only if it exceeds mean+2std.
    """
    rng = np.random.default_rng(seed)
    vals = []
    N = H_b.shape[0]
    for _ in range(n_perm):
        perm = rng.permutation(N)
        vals.append(linear_cka(H_a, H_b[perm]))
    vals = np.asarray(vals)
    return float(vals.mean()), float(vals.std())


# ----------------------------------------------------------------------
# Orthogonal Procrustes residual -- paper Prop 4.1
# ----------------------------------------------------------------------
def procrustes_residual(H_a, H_b):
    """
    Align H_b to H_a by the optimal orthogonal Q* (requires same d), return
    Q*, the normalized residual rho = ||H_a - H_b Q*||_F / ||H_a||_F, and the
    aligned H_b. rho is invariant to separate orthogonal reparametrizations.
    """
    assert H_a.shape == H_b.shape, "Procrustes needs equal shapes (same d)"
    Ha = center_rows(H_a)
    Hb = center_rows(H_b)
    # H_b^T H_a = U Σ V^T ; Q* = U V^T  (maximizes tr(Q^T H_b^T H_a))
    M = Hb.T @ Ha
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    Q = U @ Vt
    Hb_aligned = Hb @ Q
    num = np.linalg.norm(Ha - Hb_aligned, ord="fro")
    den = np.linalg.norm(Ha, ord="fro")
    rho = float(num / den) if den != 0 else float("inf")
    return Q, rho, Hb_aligned


def linear_map_residual(H_a, H_b, ridge=1e-3):
    """
    Unequal-dim alignment: least-squares W (ridge) with H_b W ~ H_a, return
    normalized residual. Low ridge keeps W a coordinate correction, not new
    learning (paper remark after Prop 4.1). Use when d_a != d_b.
    """
    Ha = center_rows(H_a)
    Hb = center_rows(H_b)
    d_b = Hb.shape[1]
    # W = (Hb^T Hb + ridge I)^-1 Hb^T Ha
    A = Hb.T @ Hb + ridge * np.eye(d_b)
    W = np.linalg.solve(A, Hb.T @ Ha)
    resid = Ha - Hb @ W
    num = np.linalg.norm(resid, ord="fro")
    den = np.linalg.norm(Ha, ord="fro")
    rho = float(num / den) if den != 0 else float("inf")
    return W, rho


# ----------------------------------------------------------------------
# Principal angles / shared dimension -- paper Def 3.4
# ----------------------------------------------------------------------
def principal_angles(H_a, H_b, energy=0.99):
    """
    Principal angles between the column spaces of H_a and H_b.
    Each H is reduced to an orthonormal basis of its dominant subspace
    (capturing `energy` fraction of variance) before computing cos(theta_i)
    as singular values of U_a^T U_b.
    Returns cos2 = cos^2(theta_i) sorted desc, and k_eff = sum(cos2).
    """
    def ortho_basis(H):
        Hc = center_rows(H)
        U, S, _ = np.linalg.svd(Hc, full_matrices=False)
        if S.sum() == 0:
            return U[:, :0]
        cum = np.cumsum(S**2) / np.sum(S**2)
        k = int(np.searchsorted(cum, energy) + 1)
        return U[:, :k]            # N x k orthonormal columns

    Ua = ortho_basis(H_a)
    Ub = ortho_basis(H_b)
    if Ua.shape[1] == 0 or Ub.shape[1] == 0:
        return np.array([]), 0.0
    s = np.linalg.svd(Ua.T @ Ub, compute_uv=False)
    cos2 = np.clip(s, 0, 1) ** 2
    cos2 = np.sort(cos2)[::-1]
    return cos2, float(cos2.sum())


# ----------------------------------------------------------------------
# Effective rank / participation ratio -- paper Section 6 (collapse)
# ----------------------------------------------------------------------
def participation_ratio(H):
    """
    PR = (sum mu_i)^2 / sum mu_i^2 over Gram eigenvalues mu_i.
    Basis-invariant effective rank; track across self-training rounds to
    detect representational variance collapse.
    """
    G = gram(H, centered=True)
    mu = np.linalg.eigvalsh(G)
    mu = mu[mu > 0]
    if mu.size == 0:
        return 0.0
    return float((mu.sum() ** 2) / (mu ** 2).sum())


# ----------------------------------------------------------------------
# convenience: all pairwise invariants for a donor/host pair
# ----------------------------------------------------------------------
def pair_invariants(H_donor, H_host, energy=0.99):
    """Compute the full invariant set for one graft boundary pair."""
    out = {
        "cka": linear_cka(H_donor, H_host),
        "pr_donor": participation_ratio(H_donor),
        "pr_host": participation_ratio(H_host),
    }
    cos2, keff = principal_angles(H_donor, H_host, energy=energy)
    out["k_eff"] = keff
    out["n_shared_strong"] = int((cos2 > 0.5).sum())  # angles < 45 deg
    if H_donor.shape[1] == H_host.shape[1]:
        _, rho, _ = procrustes_residual(H_donor, H_host)
        out["procrustes_rho"] = rho
    else:
        _, rho = linear_map_residual(H_donor, H_host)
        out["linear_rho"] = rho
    return out
