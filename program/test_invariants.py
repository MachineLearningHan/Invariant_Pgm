# PAPER: Self-tests for invariants.py (Sec 2-5 invariance). See RESULTS_TO_CODE.md
"""
Self-tests for invariants.py — verify the theory holds numerically before
trusting these on real grafting logs. Run: python -u test_invariants.py
Live per-check output (flush) per standing rule.
"""
import numpy as np
from invariants import (linear_cka, procrustes_residual, principal_angles,
                        participation_ratio, pair_invariants)

rng = np.random.default_rng(0)
N, d = 200, 64


def rand_orth(d):
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    return Q


def check(name, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}", flush=True)
    return cond


ok = True

# 1. CKA orthogonal-invariance: CKA(H, H Q) == 1
H = rng.standard_normal((N, d))
Q = rand_orth(d)
c = linear_cka(H, H @ Q)
ok &= check("CKA(H, HQ) == 1", abs(c - 1.0) < 1e-8, f"(got {c:.6f})")

# 2. CKA isotropic-scale invariance: CKA(H, 5H) == 1
c = linear_cka(H, 5.0 * H)
ok &= check("CKA(H, 5H) == 1", abs(c - 1.0) < 1e-8, f"(got {c:.6f})")

# 3. CKA of independent reps sits at the permutation-null baseline
#    (finite-N CKA of unrelated reps is ~d/N, not 0; judge against null)
from invariants import cka_null
H2 = rng.standard_normal((N, d))
c = linear_cka(H, H2)
null_mean, null_std = cka_null(H, H2, n_perm=100)
ok &= check("CKA(H, indep) within null band",
            abs(c - null_mean) < 4 * null_std + 0.02,
            f"(got {c:.3f}, null {null_mean:.3f}±{null_std:.3f})")

# 4. Procrustes residual ~ 0 for a pure rotation
_, rho, _ = procrustes_residual(H, H @ Q)
ok &= check("Procrustes rho(H, HQ) ~ 0", rho < 1e-6, f"(got {rho:.2e})")

# 5. Procrustes residual grows with added noise (monotone)
rhos = []
for eps in [0.0, 0.25, 0.5, 1.0, 2.0]:
    Hn = H @ Q + eps * rng.standard_normal((N, d))
    _, r, _ = procrustes_residual(H, Hn)
    rhos.append(r)
mono = all(rhos[i] <= rhos[i+1] + 1e-9 for i in range(len(rhos)-1))
ok &= check("Procrustes rho monotone in noise", mono, f"{[round(x,3) for x in rhos]}")

# 6. Principal-angle k_eff: identical subspace -> k_eff == rank; orthogonal
#    subspaces -> k_eff ~ 0
cos2_same, keff_same = principal_angles(H, H @ Q)
ok &= check("k_eff(H, HQ) ~ full", keff_same > 0.99 * len(cos2_same),
            f"(keff={keff_same:.2f}, k={len(cos2_same)})")

# build two reps sharing only a partial subspace
shared = rng.standard_normal((N, 10))
A_only = rng.standard_normal((N, 20))
B_only = rng.standard_normal((N, 20))
H_A = np.hstack([shared, A_only])         # spans shared + A
H_B = np.hstack([shared, B_only])         # spans shared + B
_, keff_partial = principal_angles(H_A, H_B)
ok &= check("k_eff partial-overlap ~ shared dim (10)",
            7 < keff_partial < 13, f"(got {keff_partial:.2f})")

# 7. CKA monotone with overlap fraction
def mix(frac):
    k = int(frac * d)
    sh = rng.standard_normal((N, k))
    a = rng.standard_normal((N, d - k))
    b = rng.standard_normal((N, d - k))
    return np.hstack([sh, a]), np.hstack([sh, b])
ckas = []
for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
    Ha, Hb = mix(frac)
    ckas.append(linear_cka(Ha, Hb))
mono = all(ckas[i] <= ckas[i+1] + 0.05 for i in range(len(ckas)-1))
ok &= check("CKA monotone in shared fraction", mono,
            f"{[round(x,3) for x in ckas]}")

# 8. participation ratio: full-rank iid ~ d; rank-1 -> ~1
pr_full = participation_ratio(rng.standard_normal((N, d)))
v = rng.standard_normal((N, 1))
pr_rank1 = participation_ratio(v @ rng.standard_normal((1, d)))
ok &= check("PR(iid) near d", pr_full > 0.6 * d, f"(got {pr_full:.1f} / {d})")
ok &= check("PR(rank-1) ~ 1", pr_rank1 < 1.5, f"(got {pr_rank1:.3f})")

# 9. unequal-dim path runs (linear_rho) without crashing
inv = pair_invariants(rng.standard_normal((N, 48)),
                      rng.standard_normal((N, 64)))
ok &= check("unequal-dim pair_invariants returns linear_rho",
            "linear_rho" in inv, f"(keys={sorted(inv)})")

print(f"\n=== {'ALL PASS' if ok else 'SOME FAILED'} ===", flush=True)
