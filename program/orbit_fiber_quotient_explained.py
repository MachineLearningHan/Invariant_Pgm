# PAPER: Prop 3.1-3.2, Figure 2 (orbit/fiber/quotient, annotated). See RESULTS_TO_CODE.md
"""
orbit / fiber / quotient, as runnable code.
Companion to "Teacher Supervision over Representation Equivalence Classes."
NumPy only; deterministic (seed 0); prints every intermediate result.
Run:  python -u orbit_fiber_quotient.py

Reading order (familiar -> unfamiliar):
  (i)   a FEATURE is an IR      -- h is the network's SSA/intermediate form
  (ii)  the INVARIANT is output -- rewrite the IR, the returned value is fixed
  (iii) an ORBIT is the set of IR-rewrites sharing one output
  (iv)  a QUOTIENT collapses each orbit to one object
        (two of them: IR-geometry vs output; capability lives on the latter)
"""
import numpy as np, hashlib
rng = np.random.default_rng(0)

# ---- one hidden layer h and the reader W2 that consumes it -----------------
N, d, V = 6, 8, 4
x  = rng.standard_normal((N, d))
W1 = rng.standard_normal((d, d))
W2 = rng.standard_normal((d, V))
h  = x @ W1                       # a FEATURE: the network's IR (like SSA)
y  = h @ W2                       # the output it produces   (N x V)

def rotation(n):                  # a random orthogonal change of basis
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return Q

def clone(h, W2, Q, c):           # the joint rewrite: absorb (Q, c)
    return c * (h @ Q), (1.0 / c) * (Q.T @ W2)

def out_hash(h, W2):              # a 'behavior' = hash of the output function
    return hashlib.sha256(np.round(h @ W2, 6).tobytes()).hexdigest()[:12]

def gram_id(h):                   # canonical id of the REPRESENTATION quotient
    G = h @ h.T                   # h h^T is invariant to h -> c h Q ...
    return hashlib.sha256(np.round(G / np.linalg.norm(G), 6).tobytes()).hexdigest()[:12]

# ===========================================================================
# (i)+(ii)  FEATURE is an IR; the INVARIANT is the output.
# Renaming an SSA temporary (t1 -> x17) leaves the executable unchanged;
# here 'renaming' the feature (h -> h @ Q, with the reader updated) leaves
# the output unchanged. The IR moved; the invariant (output) did not.
# ===========================================================================
print("=== FEATURE-as-IR: rewrite the IR, the output is invariant ===", flush=True)
_r = np.random.default_rng(12345)                 # isolated: no effect downstream
_Q, _ = np.linalg.qr(_r.standard_normal((d, d))); _c = 1.7
h_ir, W2_ir = clone(h, W2, _Q, _c)                # a re-coordinatized IR
print(f"IR changed?  ||h_ir - h|| = {np.linalg.norm(h_ir - h):.3e}", flush=True)
print(f"output same? {np.allclose(h_ir @ W2_ir, y)}  "
      f"(hash {out_hash(h, W2)} -> {out_hash(h_ir, W2_ir)})", flush=True)

# ===========================================================================
# ORBIT  = all implementations of ONE program (different internal coordinates)
# ===========================================================================
print("=== ORBIT: same program, different internal coordinates ===", flush=True)
print(f"{'variant':<20}{'||h_var - h||':>15}{'output hash':>15}{'==y?':>7}", flush=True)
base = out_hash(h, W2)
print(f"{'original':<20}{0.0:>15.3e}{base:>15}{'yes':>7}", flush=True)
for k in range(1, 4):
    Q, c = rotation(d), float(rng.uniform(0.5, 2.0))
    hk, W2k = clone(h, W2, Q, c)
    ok = np.allclose(hk @ W2k, y)
    print(f"{'clone %d'%k:<20}{np.linalg.norm(hk-h):>15.3e}"
          f"{out_hash(hk,W2k):>15}{('yes' if ok else 'NO'):>7}", flush=True)

# ===========================================================================
# FIBER  = pi^-1(behavior): everything mapping to the same output
# ===========================================================================
print("\n=== FIBER: everything that maps to one output (same-hash set) ===", flush=True)
inside = sum(out_hash(*clone(h, W2, rotation(d), float(rng.uniform(0.5, 2.0)))) == base
             for _ in range(200))
print(f"joint rewrites landing in fiber of y : {inside}/200", flush=True)
y_rot = (h @ rotation(d)) @ W2     # rotate h but DON'T fix the reader
rot_hash = hashlib.sha256(np.round(y_rot, 6).tobytes()).hexdigest()[:12]
print(f"rotation without compensating reader : hash {rot_hash} "
      f"({'in' if np.allclose(y_rot, y) else 'OUTSIDE'} the fiber)", flush=True)

# ===========================================================================
# QUOTIENT  = collapse each orbit to one point.
# Two programs A, B share the hidden rep h but have DIFFERENT readers.
#   representation quotient (gram)  -> collapses A,B together (CKA can't tell)
#   joint quotient (output)         -> keeps A,B apart (capability can)
# ===========================================================================
print("\n=== QUOTIENT: representation vs joint, on two programs ===", flush=True)
W2b = rng.standard_normal((d, V))                 # program B: different reader
rows = []
for name, ww in (("A", W2), ("B", W2b)):
    for _ in range(30):
        Q, c = rotation(d), float(rng.uniform(0.5, 2.0))
        rows.append((name, *clone(h, ww, Q, c)))
joint, repq = {}, {}
for name, hk, wk in rows:
    joint.setdefault(out_hash(hk, wk), set()).add(name)
    repq.setdefault(gram_id(hk),      set()).add(name)
print("Input ------------------------------------------------------", flush=True)
print("30 clones of Program A + 30 clones of Program B", flush=True)
print("(A and B share the same hidden FEATURE/IR h, but use different readers W)", flush=True)

print("\nRepresentation Quotient -----------------------------------", flush=True)
print("Identifier : normalized Gram matrix (feature geometry)", flush=True)
print("Meaning    : compare only the hidden representation geometry", flush=True)
print(f"Result     : {len(repq)} class", flush=True)
print(f"A and B collapsed together: {any(len(v) > 1 for v in repq.values())}", flush=True)

print("\nJoint Quotient --------------------------------------------", flush=True)
print("Identifier : output function (observable input -> output behavior)", flush=True)
print("Meaning    : compare program semantics, not hidden features", flush=True)
print(f"Result     : {len(joint)} classes", flush=True)
print(f"Every class contains only one program: {all(len(v) == 1 for v in joint.values())}", flush=True)

print("\nInterpretation --------------------------------------------", flush=True)
print("Gram matrix describes only FEATURE GEOMETRY (what CKA measures).", flush=True)
print("Output function describes PROGRAM BEHAVIOR (observable semantics).", flush=True)
print("Therefore A and B look identical on the representation quotient,", flush=True)
print("but remain different on the joint quotient because they compute", flush=True)
print("different functions.", flush=True)
print("\n=> Capability lives on the JOINT quotient, whereas CKA/Gram live", flush=True)
print("   only on the REPRESENTATION quotient.", flush=True)
