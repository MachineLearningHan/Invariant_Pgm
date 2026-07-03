# PAPER: Section 5 (Predictions 5.1-5.3) -- overlap->success AUC + domain->CKA->success mediation. See RESULTS_TO_CODE.md
"""
Graft-overlap analysis (paper Section 5 / Predictions 5.1-5.3).

Input: a list of graft records, each
    {
      "pair_id": str,
      "domain_match": 0/1,         # donor/host pretrain domain aligned?
      "success": 0/1,              # did the graft preserve function?
      "H_donor": np.ndarray (N,da),
      "H_host":  np.ndarray (N,dh),
      "aligned": 0/1 (optional)    # was a linear alignment map applied?
    }

Outputs:
  - per-pair invariants (CKA, rho, k_eff)
  - Prediction 1: success vs CKA / rho (logistic regression, AUC)
  - Prediction 2: domain -> CKA -> success mediation (simple causal steps)
  - Prediction 3: overlap x alignment interaction

No sklearn dependency: logistic regression by Newton steps; AUC by rank.
Live per-step output (flush) per standing rule.

Run self-test:  python -u analyze_grafts.py --selftest
Run on data:    python -u analyze_grafts.py --data grafts.pkl
"""
import sys, argparse, pickle
import numpy as np
from invariants import linear_cka, cka_null, pair_invariants


# ----------------------------------------------------------------------
# tiny logistic regression (Newton-Raphson) + AUC
# ----------------------------------------------------------------------
def logreg(X, y, iters=50, l2=1e-3):
    X = np.asarray(X, float)
    X = np.hstack([np.ones((X.shape[0], 1)), X])   # bias
    y = np.asarray(y, float)
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = X @ w
        p = 1 / (1 + np.exp(-z))
        W = p * (1 - p) + 1e-9
        grad = X.T @ (p - y) + l2 * w
        H = X.T @ (X * W[:, None]) + l2 * np.eye(X.shape[1])
        w -= np.linalg.solve(H, grad)
    return w


def auc(scores, y):
    y = np.asarray(y)
    s = np.asarray(scores)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U / (n_pos n_neg)
    order = np.argsort(s)
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(s) + 1)
    R_pos = ranks[y == 1].sum()
    n1, n0 = len(pos), len(neg)
    return float((R_pos - n1 * (n1 + 1) / 2) / (n1 * n0))


# ----------------------------------------------------------------------
# main analysis
# ----------------------------------------------------------------------
def analyze(records, verbose=True):
    rows = []
    for r in records:
        inv = pair_invariants(r["H_donor"], r["H_host"])
        rho = inv.get("procrustes_rho", inv.get("linear_rho"))
        nm, ns = cka_null(r["H_donor"], r["H_host"], n_perm=50)
        row = {
            "pair_id": r.get("pair_id", "?"),
            "success": int(r["success"]),
            "domain_match": int(r.get("domain_match", -1)),
            "aligned": int(r.get("aligned", 0)),
            "cka": inv["cka"],
            "cka_z": (inv["cka"] - nm) / (ns + 1e-9),   # over null
            "rho": rho,
            "k_eff": inv["k_eff"],
        }
        rows.append(row)
        if verbose:
            print(f"[pair {row['pair_id']}] success={row['success']} "
                  f"cka={row['cka']:.3f} (z={row['cka_z']:+.1f}) "
                  f"rho={rho:.3f} k_eff={row['k_eff']:.1f} "
                  f"dom={row['domain_match']} align={row['aligned']}",
                  flush=True)

    y = np.array([r["success"] for r in rows])
    cka = np.array([r["cka"] for r in rows])
    rho = np.array([r["rho"] for r in rows])

    print("\n=== Prediction 1: success vs overlap ===", flush=True)
    w_cka = logreg(cka[:, None], y)
    auc_cka = auc(cka, y)
    print(f"  CKA  -> success : logit coef {w_cka[1]:+.2f} "
          f"(expect >0), AUC={auc_cka:.3f}", flush=True)
    w_rho = logreg(rho[:, None], y)
    auc_rho = auc(-rho, y)   # lower rho should mean success
    print(f"  rho  -> success : logit coef {w_rho[1]:+.2f} "
          f"(expect <0), AUC(-rho)={auc_rho:.3f}", flush=True)

    # Prediction 2: mediation domain -> cka -> success
    dom = np.array([r["domain_match"] for r in rows])
    if (dom >= 0).all() and len(np.unique(dom)) > 1:
        print("\n=== Prediction 2: domain -> CKA -> success mediation ===",
              flush=True)
        # step a: domain -> cka (mean diff)
        ca = cka[dom == 1].mean() - cka[dom == 0].mean()
        # step b: total effect domain -> success
        w_tot = logreg(dom[:, None], y)
        # step c: direct effect controlling cka
        w_dir = logreg(np.column_stack([dom, cka]), y)
        print(f"  a) domain->CKA  : ΔCKA={ca:+.3f} (expect >0)", flush=True)
        print(f"  b) total domain->success coef {w_tot[1]:+.2f}", flush=True)
        print(f"  c) direct (|CKA) domain coef  {w_dir[1]:+.2f} "
              f"(expect shrink toward 0 => mediated)", flush=True)

    # Prediction 3: overlap x alignment interaction
    al = np.array([r["aligned"] for r in rows])
    if len(np.unique(al)) > 1:
        print("\n=== Prediction 3: overlap x alignment interaction ===",
              flush=True)
        inter = cka * al
        w = logreg(np.column_stack([cka, al, inter]), y)
        print(f"  cka coef {w[1]:+.2f}, align coef {w[2]:+.2f}, "
              f"interaction coef {w[3]:+.2f} (expect <0: align helps "
              f"most at low overlap)", flush=True)

    return rows


# ----------------------------------------------------------------------
# synthetic self-test: build records where the theory holds by construction
# ----------------------------------------------------------------------
def make_synthetic(n_pairs=60, N=150, seed=1):
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n_pairs):
        domain = int(rng.random() < 0.5)
        # domain match -> higher shared fraction
        frac = (0.55 if domain else 0.2) + 0.15 * rng.standard_normal()
        frac = float(np.clip(frac, 0.02, 0.95))
        d = 64
        k = max(1, int(frac * d))
        shared = rng.standard_normal((N, k))
        Hd = np.hstack([shared, rng.standard_normal((N, d - k))])
        Hh = np.hstack([shared @ rng.standard_normal((k, k)),
                        rng.standard_normal((N, d - k))])
        # success prob rises with shared fraction; alignment helps when low
        aligned = int(rng.random() < 0.5)
        base = frac + (0.25 * (1 - frac) if aligned else 0.0)
        success = int(rng.random() < np.clip(base, 0.02, 0.98))
        recs.append({"pair_id": f"S{i:02d}", "domain_match": domain,
                     "success": success, "aligned": aligned,
                     "H_donor": Hd, "H_host": Hh})
    return recs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data", default="")
    args = ap.parse_args()
    if args.selftest:
        print("[selftest] synthetic graft records where theory holds\n",
              flush=True)
        analyze(make_synthetic())
    elif args.data:
        with open(args.data, "rb") as f:
            recs = pickle.load(f)
        analyze(recs)
    else:
        print("use --selftest or --data path.pkl", flush=True)
