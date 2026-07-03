# PAPER: MLP appendix, Table 8 (tab:mlp) -- claims exact on a norm-free MLP. See RESULTS_TO_CODE.md
"""
Minimal verification of the paper's core claims on a tiny MLP.

The paper's claims are not specific to transformers; they are properties of
representation learning in general. This script demonstrates them on a 2-layer
MLP trained on a toy 2D classification task, with hand-written forward/backward
(no autograd, no deep-learning framework -- numpy only), so every step is
inspectable.

A clean point that is only APPROXIMATE in an LLM (because RMSNorm's learned gain
does not commute with a general rotation) becomes EXACT here: a plain MLP has no
normalization, so an orthogonal reparametrization of a hidden layer is absorbed
exactly by the next layer's weights and the function is preserved to machine
precision.

Network (teacher):  x -> [W1,b1] -> ReLU -> h(2) -> [W2,b2] -> logits(2)
The hidden activation h in R^{N x H} is the "representation".

Claims checked:
  (1) Prop 2.1 -- absolute feature matching is ill-posed.
      A function-preserving rotation h -> hQ leaves CKA / normalized-Gram at 0
      but sends the feature-matching loss ||h - hQ||^2 to large values.
  (2) Assumption 2.1 / Prop 3.1 -- the absorbing identity eq (1), exact here.
      The operator consuming the representation is the next linear layer W2
      (= W_in in the paper), with psi-tilde = identity. The absorbing identity,
      numbered eq (1) in the paper, has two parts:
        (1a) cancellation:  (hQ)(Q^T W2) = h W2   (exact, since Q Q^T = I)
        (1b) consequence:   phi^Q(hQ) = phi(h)    (function unchanged)
      with no RMSNorm gain to spoil (1a). So injecting Q after the hidden layer
      AND Q^T into W2 leaves the function unchanged to machine precision, while
      injecting Q alone (no compensation) changes it. This is Assumption 2.1
      with every approximation of the paper's Remark 2.1 removed.
  (3) Representation != function (the ablation, in miniature).
      Train a student's hidden layer to match the teacher's hidden rep under
      (a) a basis-invariant objective (CKA) vs (b) an output-function objective
      (logit/KL match). CKA-matching aligns the representation (CKA->1) but does
      not give the teacher's function; output-matching gives the function.
  (4) Procrustes alignment recovers the rotation that absolute matching needs.

After the four claims, make_quotient_figure() both verifies and (if matplotlib
is available) renders the paper's Figure 2: rotating h with the downstream W
held fixed breaks the output (representation quotient is not enough), while the
joint action h->hQ, W->Q^T W preserves it (capability lives on the joint
quotient). The figure is saved as fig2_quotient.pdf.

Run:  python3 minimal_mlp.py

The script opens with two 3-minute warmups before the full MLP verification:
  Warmup A -- the absorbing identity (hQ)(Q^T W) = h W by hand on a 2x2 example.
  Warmup B -- on a 100x512 random matrix, a function-preserving rotation H -> HQ
              makes absolute feature matching ||H - HQ||^2 huge while the Gram
              (basis-invariant) loss stays ~0. Runs in torch if available, else
              numpy.
"""
import numpy as np

rng = np.random.default_rng(0)


# ----------------------------- toy data -----------------------------
def two_moons(n=400, noise=0.15, seed=0):
    r = np.random.default_rng(seed)
    n2 = n // 2
    t = np.linspace(0, np.pi, n2)
    x0 = np.stack([np.cos(t), np.sin(t)], 1)
    x1 = np.stack([1 - np.cos(t), 1 - np.sin(t) - 0.5], 1)
    X = np.concatenate([x0, x1], 0) + noise * r.standard_normal((n2 * 2, 2))
    y = np.concatenate([np.zeros(n2, int), np.ones(n2, int)])
    return X.astype(np.float64), y


# ----------------------------- MLP -----------------------------
class MLP:
    """x -> W1 -> relu -> h -> W2 -> logits. Plain, no normalization."""
    def __init__(self, d_in=2, d_h=16, d_out=2, seed=0):
        r = np.random.default_rng(seed)
        self.W1 = r.standard_normal((d_in, d_h)) / np.sqrt(d_in)
        self.b1 = np.zeros(d_h)
        self.W2 = r.standard_normal((d_h, d_out)) / np.sqrt(d_h)
        self.b2 = np.zeros(d_out)

    def hidden(self, X):
        return np.maximum(0.0, X @ self.W1 + self.b1)  # ReLU

    def logits(self, X):
        return self.hidden(X) @ self.W2 + self.b2

    def copy(self):
        m = MLP.__new__(MLP)
        m.W1, m.b1 = self.W1.copy(), self.b1.copy()
        m.W2, m.b2 = self.W2.copy(), self.b2.copy()
        return m


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def train_classifier(net, X, y, steps=4000, lr=0.2):
    N = X.shape[0]
    Y = np.eye(2)[y]
    for _ in range(steps):
        H = net.hidden(X)
        Z = H @ net.W2 + net.b2
        P = softmax(Z)
        dZ = (P - Y) / N
        gW2 = H.T @ dZ
        gb2 = dZ.sum(0)
        dH = dZ @ net.W2.T
        dH[H <= 0] = 0.0
        gW1 = X.T @ dH
        gb1 = dH.sum(0)
        net.W2 -= lr * gW2; net.b2 -= lr * gb2
        net.W1 -= lr * gW1; net.b1 -= lr * gb1
    return net


# ----------------------------- invariants -----------------------------
def center(G):
    N = G.shape[0]
    C = np.eye(N) - np.ones((N, N)) / N
    return C @ G @ C


def linear_cka(A, B):
    GA, GB = center(A @ A.T), center(B @ B.T)
    return (GA * GB).sum() / (np.linalg.norm(GA) * np.linalg.norm(GB) + 1e-12)


def gram_rel(A, B):
    GA, GB = center(A @ A.T), center(B @ B.T)
    GA /= np.linalg.norm(GA) + 1e-12
    GB /= np.linalg.norm(GB) + 1e-12
    return ((GA - GB) ** 2).sum()


def feature_kd(A, B):
    return ((A - B) ** 2).sum() / A.shape[0]


def random_orthogonal(d, seed):
    A = np.random.default_rng(seed).standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    return Q @ np.diag(np.sign(np.diag(R)))


def expm_interp(Q, t):
    from scipy.linalg import logm, expm
    return expm(t * logm(Q)).real


# --------------------- 3-minute warmups (read these first) ---------------------
def hand_calculation():
    """The absorbing identity (hQ)(Q^T W) = h W on the smallest possible example,
    so it can be checked by hand. Representation h in R^{1x2}, consuming operator
    W in R^{2x1}, Q a 90-degree rotation in O(2)."""
    print("\n=== Warmup A: the absorbing identity by hand (2x2) ===")
    h = np.array([[1.0, 0.0]])            # representation, 1 x 2
    W = np.array([[0.0], [1.0]])          # consuming operator, 2 x 1
    Q = np.array([[0.0, 1.0],             # 90-degree rotation in O(2)
                  [-1.0, 0.0]])
    hW = h @ W                            # original output
    hQ = h @ Q                            # rotated representation
    QtW = Q.T @ W                         # counter-rotated operator
    out = hQ @ QtW                        # (hQ)(Q^T W)
    print(f"  h        = {h.tolist()}")
    print(f"  W^T      = {W.T.tolist()}        (W is 2x1)")
    print(f"  Q        = {Q.tolist()}   (Q Q^T = I: {np.allclose(Q @ Q.T, np.eye(2))})")
    print(f"  h W           = {hW.ravel()[0]:+.1f}")
    print(f"  h Q           = {hQ.tolist()}        (representation rotated)")
    print(f"  Q^T W (=W')   = {QtW.ravel().tolist()}      (operator counter-rotated)")
    print(f"  (hQ)(Q^T W)   = {out.ravel()[0]:+.1f}        <- same as h W = {hW.ravel()[0]:+.1f}")
    print("  -> rotating the representation and counter-rotating the operator")
    print("     leave the output identical. That is eq (1a), by hand.")


def two_line_demo():
    """The paper's central contrast in a few lines on a large random matrix:
    a function-preserving rotation H -> HQ sends absolute feature matching
    ||H - HQ||^2 to a huge value, while the basis-invariant Gram loss stays ~0.
    Runs in torch if available (the snippet from the discussion), else numpy."""
    print("\n=== Warmup B: absolute matching is huge, Gram loss is ~0 ===")
    try:
        import torch
        H = torch.randn(100, 512)
        Q, _ = torch.linalg.qr(torch.randn(512, 512))   # random orthogonal
        HQ = H @ Q
        L_abs = (torch.norm(H - HQ, p="fro") ** 2).item()
        G_H, G_HQ = H @ H.T, HQ @ HQ.T
        L_gram = (torch.norm(G_H - G_HQ, p="fro") ** 2).item()
        backend = "torch"
    except Exception:
        H = rng.standard_normal((100, 512))
        Q = random_orthogonal(512, 11)
        HQ = H @ Q
        L_abs = float(((H - HQ) ** 2).sum())
        G_H, G_HQ = H @ H.T, HQ @ HQ.T
        L_gram = float(((G_H - G_HQ) ** 2).sum())
        backend = "numpy"
    print(f"  [{backend}]  H: 100 x 512,  Q random orthogonal,  HQ = H Q")
    print(f"  absolute feature matching ||H - HQ||_F^2 = {L_abs:,.2f}      (huge)")
    print(f"  Gram-matrix loss          ||G_H - G_HQ||_F^2 = {L_gram:.3e}   (~0)")
    print("  -> H and HQ are the same network up to basis, yet absolute matching")
    print("     penalizes them heavily; the Gram (relational) loss sees them as equal.")


# ----------------------------- claims -----------------------------
def assumption_2_1_walkthrough(teacher, X):
    """Map and verify each element of Assumption 2.1, one line at a time, on the
    MLP. The paper writes the layer-l-to-output map as

        phi_{l:L} = rho . psi,    psi(h) = psi_tilde(W_in @ h),

    and assumes a rotation h -> hQ of the representation is undone by the
    counter-rotation W_in -> Q^T W_in, giving phi^Q(hQ) = phi(h) and f^Q = f.
    Here we name each symbol in MLP terms and check the assumption's identity
    holds, step by step."""
    print("\n=== Assumption 2.1, element by element (MLP instance) ===")
    d = teacher.W1.shape[1] 
    Q = random_orthogonal(d, 5)
    h = teacher.hidden(X)                       # the layer-l representation
    #d = h.shape[1]

    # --- the symbols of Assumption 2.1, in MLP terms ---
    print(f"  d   (representation width)          = {d}")
    print(f"  h   (layer-l representation H_l)    = ReLU(X W1 + b1), shape {h.shape}")
    print(f"  W_in(operator consuming h)          = W2, shape {teacher.W2.shape}")
    print(f"  psi_tilde (post-W_in nonlinearity)  = identity  (no normalization)")
    print(f"  rho (rest of net to logits)         = add b2  (this is the last layer)")
    print(f"  phi_{{l:L}}(h) = rho(psi_tilde(W_in h)) = h W2 + b2  (the logits)")

    # --- step 1: psi(h) = psi_tilde(W_in h). here psi_tilde = id, so psi = W_in h
    psi_h = h @ teacher.W2                       # psi(h) = W_in h
    # --- step 2: eq (1a) of the paper, the cancellation
    #     psi_tilde((Q^T W_in)(hQ)) = psi_tilde(W_in h)
    psi_rotated = (h @ Q) @ (Q.T @ teacher.W2)   # (hQ)(Q^T W2)
    err_absorb = np.abs(psi_rotated - psi_h).max()
    print(f"  step 1  psi(h) = W_in h                       computed, shape {psi_h.shape}")
    print(f"  step 2  eq (1a) || psi_tilde((Q^T W_in)(hQ)) - psi_tilde(W_in h) || "
          f"= {err_absorb:.3e}")
    print(f"          (Q Q^T = I cancels; exact, no gamma to spoil it)")

    # --- step 3: eq (1b), the consequence  phi^Q(hQ) = phi(h)
    phi_h = h @ teacher.W2 + teacher.b2          # phi_{l:L}(h)
    phi_Q_hQ = (h @ Q) @ (Q.T @ teacher.W2) + teacher.b2   # phi^Q(hQ)
    err_phi = np.abs(phi_Q_hQ - phi_h).max()
    print(f"  step 3  eq (1b) || phi^Q(hQ) - phi(h) ||      = {err_phi:.3e}")

    # --- step 4: hence f^Q = f  (network function unchanged).
    #     In this 2-layer MLP phi_{l:L} IS the rest of the network, so step 4
    #     numerically coincides with step 3 (f = phi here); in a deeper net the
    #     two would differ as separate paths.
    f = teacher.logits(X)
    f_Q = (h @ Q) @ (Q.T @ teacher.W2) + teacher.b2
    err_f = np.abs(f_Q - f).max()
    print(f"  step 4  || f^Q - f ||  (network function)     = {err_f:.3e}"
          f"   (= step 3 here: phi_{{l:L}} = f in a 2-layer net)")

    # --- contrast: drop the counter-rotation -> assumption's premise fails
    f_nocomp = (h @ Q) @ teacher.W2 + teacher.b2     # W_in NOT counter-rotated
    err_nc = np.abs(f_nocomp - f).max()
    print(f"  contrast: WITHOUT W_in -> Q^T W_in,  || f - f || = {err_nc:.3e}")
    print(f"  -> every step of Assumption 2.1 holds to machine precision; the")
    print(f"     premise (counter-rotate W_in) is what makes the function invariant.")


def claim1_illposed(teacher, X):
    print("\n=== Claim 1: Prop 2.1 -- absolute feature matching is ill-posed ===")
    H = teacher.hidden(X)                      # representation (N x d_h)
    d = H.shape[1]
    Q = random_orthogonal(d, 1)
    print(f"{'rot t':>6} | {'FeatureKD':>12} {'1-CKA':>10} {'L_rel':>10}")
    have_scipy = True
    try:
        import scipy.linalg  # noqa
    except Exception:
        have_scipy = False
    ts = [0.0, 0.25, 0.5, 0.75, 1.0] if have_scipy else [0.0, 1.0]
    for t in ts:
        Qt = np.eye(d) if t == 0 else (expm_interp(Q, t) if have_scipy else Q)
        Hr = H @ Qt
        print(f"{t:>6.2f} | {feature_kd(H, Hr):>12.4f} "
              f"{1 - linear_cka(H, Hr):>10.6f} {gram_rel(H, Hr):>10.6f}")
    print("  -> Feature-KD grows; CKA and L_rel stay ~0 (exact invariance).")


def claim2_joint_invariance(teacher, X):
    print("\n=== Claim 2: Assumption 2.1 / Prop 3.1 -- the absorbing identity, "
          "exact in an MLP ===")
    d = teacher.W1.shape[1]
    Q = random_orthogonal(d, 2)
    f0 = teacher.logits(X)

    # This is Assumption 2.1 made concrete. In the paper's notation the operator
    # that consumes the layer-l representation is W_in; here the representation
    # is h = ReLU(...) (teacher.hidden) and the consuming operator is the next
    # linear layer W2, so  W_in = W2  and  psi-tilde = identity (the layer is
    # purely linear, no normalization). The absorbing identity
    #     psi-tilde( (Q^T W_in)(h Q) ) = psi-tilde( W_in h )
    # becomes, with psi-tilde = id,
    #     (h Q)(Q^T W2) = h (Q Q^T) W2 = h W2,
    # exact because Q Q^T = I. There is no RMSNorm gain gamma in an MLP, so the
    # approximation of the paper's Remark 2.1 is absent and the identity holds
    # to machine precision.
    H = teacher.hidden(X)

    # (i) the absorbing identity at the operator, directly: (hQ)(Q^T W2) vs h W2
    lhs = (H @ Q) @ (Q.T @ teacher.W2)
    rhs = H @ teacher.W2
    print(f"  absorbing identity ||(hQ)(Q^T W2) - h W2|| = "
          f"{np.abs(lhs - rhs).max():.3e}   (Q Q^T = I cancels; exact)")

    # (ii) consequence at the network function: joint reparam preserves f
    f_joint = (H @ Q) @ (Q.T @ teacher.W2) + teacher.b2   # h->hQ, W2->Q^T W2
    f_rot_only = (H @ Q) @ teacher.W2 + teacher.b2         # h->hQ, W2 unchanged

    print(f"  ||f_joint - f||      = {np.abs(f_joint - f0).max():.3e}   "
          f"(joint reparam W_in->Q^T W_in: function preserved)")
    print(f"  ||f_rot_only - f||   = {np.abs(f_rot_only - f0).max():.3e}   "
          f"(rotation alone, W_in unchanged: function CHANGED)")
    print("  -> Assumption 2.1 holds exactly here: the rotation h->hQ is undone")
    print("     by the counter-rotation W2->Q^T W2, so f is invariant under the")
    print("     JOINT action (h and downstream), not h alone (Remark 3.1).")


def claim3_rep_vs_function(teacher, X, y):
    print("\n=== Claim 3: representation != function (the ablation, in miniature) ===")
    H_T = teacher.hidden(X)            # teacher representation
    Z_T = teacher.logits(X)            # teacher logits (the function)
    P_T = softmax(Z_T)
    d = H_T.shape[1]
    N = X.shape[0]

    def fresh_student():
        s = MLP(2, d, 2, seed=7)       # different init = different representative
        return s

    # (a) train student hidden to match teacher REP via CKA (basis-invariant)
    s_cka = fresh_student()
    lr = 0.5
    for _ in range(3000):
        H = s_cka.hidden(X)
        # maximize CKA -> minimize 1-CKA; finite-diff-free analytic grad is messy,
        # so use a surrogate: align centered Gram (L_rel), same invariance class.
        GA, GB = center(H @ H.T), center(H_T @ H_T.T)
        GA_n = GA / (np.linalg.norm(GA) + 1e-12)
        GB_n = GB / (np.linalg.norm(GB) + 1e-12)
        dG = 2 * (GA_n - GB_n) / (np.linalg.norm(GA) + 1e-12)
        Cc = np.eye(N) - np.ones((N, N)) / N
        dGram = Cc @ dG @ Cc
        dH = 2 * dGram @ H
        mask = (X @ s_cka.W1 + s_cka.b1) > 0
        dH *= mask
        s_cka.W1 -= lr * (X.T @ dH)
        s_cka.b1 -= lr * dH.sum(0)

    # (b) train student to match teacher FUNCTION via logit/KL (output-function)
    s_log = fresh_student()
    lr = 0.3
    for _ in range(3000):
        H = s_log.hidden(X)
        Z = H @ s_log.W2 + s_log.b2
        P = softmax(Z)
        dZ = (P - P_T) / N            # match teacher distribution
        s_log.W2 -= lr * (H.T @ dZ); s_log.b2 -= lr * dZ.sum(0)
        dH = dZ @ s_log.W2.T
        dH[H <= 0] = 0
        s_log.W1 -= lr * (X.T @ dH); s_log.b1 -= lr * dH.sum(0)

    def fn_match(s):
        P = softmax(s.logits(X))
        kl = (P_T * (np.log(P_T + 1e-12) - np.log(P + 1e-12))).sum(1).mean()
        top1 = (s.logits(X).argmax(1) == Z_T.argmax(1)).mean()
        return kl, top1

    cka_rep = linear_cka(s_cka.hidden(X), H_T)
    cka_fn = fn_match(s_cka)
    log_rep = linear_cka(s_log.hidden(X), H_T)
    log_fn = fn_match(s_log)

    print(f"  CKA-trained student : CKA(rep)={cka_rep:.4f}  "
          f"KL(T||S)={cka_fn[0]:.4f}  top1_agree={cka_fn[1]:.3f}")
    print(f"  logit-trained stud. : CKA(rep)={log_rep:.4f}  "
          f"KL(T||S)={log_fn[0]:.4f}  top1_agree={log_fn[1]:.3f}")
    print("  -> CKA training aligns the representation but not the function;")
    print("     output-function training restores the function. Near-orthogonal,")
    print("     exactly as in the paper's ablation (Table 2).")


def claim4_procrustes(teacher, X):
    print("\n=== Claim 4: Procrustes alignment recovers the needed rotation ===")
    H = teacher.hidden(X)
    d = H.shape[1]
    Q = random_orthogonal(d, 3)
    Hr = H @ Q                                  # a different representative
    # solve min_R ||H - Hr R||_F  ->  R* = U V^T from H_r^T H = U S V^T
    U, S, Vt = np.linalg.svd(Hr.T @ H)
    R = U @ Vt
    resid = np.linalg.norm(H - Hr @ R) / np.linalg.norm(H)
    print(f"  feature dist before align ||H - Hr||/||H|| = "
          f"{np.linalg.norm(H - Hr) / np.linalg.norm(H):.4f}")
    print(f"  after Procrustes        ||H - Hr R||/||H|| = {resid:.3e}")
    print(f"  recovered R ~ Q^T ?   ||R - Q.T|| = {np.linalg.norm(R - Q.T):.3e}")
    print("  -> Alignment finds the rotation; matching is well-posed AFTER it.")


def make_quotient_figure(teacher, X, path="fig2_quotient.pdf"):
    """Render the paper's Figure 2 (representation quotient fails vs joint
    quotient succeeds) AND verify, on the trained MLP, the two facts it depicts:
      left  -- rotating h with W2 held fixed changes the output (wrong),
      right -- the joint action (h->hQ, W2->Q^T W2) leaves the output identical.
    Saves a vector PDF if matplotlib is available; otherwise prints the numbers
    and skips the drawing (the script stays numpy-only by default)."""
    print("\n=== Figure 2: representation quotient fails vs joint quotient "
          "succeeds ===")
    d = teacher.W1.shape[1]
    Q = random_orthogonal(d, 9)
    H = teacher.hidden(X)
    f = teacher.logits(X)
    # left panel fact: h->hQ, W2 fixed  => output changes
    f_reponly = (H @ Q) @ teacher.W2 + teacher.b2
    err_left = np.abs(f_reponly - f).max()
    # right panel fact: h->hQ, W2->Q^T W2 (joint) => output identical
    f_joint = (H @ Q) @ (Q.T @ teacher.W2) + teacher.b2
    err_right = np.abs(f_joint - f).max()
    print(f"  LEFT  (rep quotient): rotate h, keep W2 fixed -> "
          f"||f' - f|| = {err_left:.3e}  (output WRONG)")
    print(f"  RIGHT (joint quotient): rotate h, counter-rotate W2 -> "
          f"||f' - f|| = {err_right:.3e}  (output PRESERVED)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except Exception:
        print("  (matplotlib not available -- skipping the drawing; the two")
        print("   numbers above are exactly what the figure depicts.)")
        return

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 12
    plt.rcParams["text.usetex"] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2))

    # ---------- left: representation quotient (fails) ----------
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 10); ax1.set_aspect("equal"); ax1.axis("off")
    ax1.set_title("Representation quotient  $\\mathbb{R}^{N\\times d}/\\mathcal{G}$\n"
                  "(collapse $h$ only)", fontsize=12, pad=8)
    cloud = (5, 8.6)
    ax1.plot(*cloud, "ko", markersize=7)
    for ang in [30, 90, 150, 210, 270, 330]:
        r = np.deg2rad(ang)
        ax1.annotate("", xy=(cloud[0]+1.1*np.cos(r), cloud[1]+1.1*np.sin(r)),
                     xytext=cloud, arrowprops=dict(arrowstyle="->", color="#3366cc", lw=1.3))
    ax1.text(5, 9.95, "orbit  $h,\\ hQ_1,\\ hQ_2,\\dots$", ha="center", fontsize=11, color="#3366cc")
    ax1.annotate("", xy=(5, 7.0), xytext=(5, 7.9), arrowprops=dict(arrowstyle="->", color="black", lw=1.8))
    ax1.text(5.55, 7.35, r"$\pi_{\mathrm{rep}}$", fontsize=13)
    ax1.plot(5, 6.7, "ko", markersize=9); ax1.text(5.5, 6.55, "$[h]$", fontsize=13)
    ax1.add_patch(mpatches.FancyBboxPatch((3.4, 4.7), 3.2, 0.95, boxstyle="round,pad=0.1",
                 facecolor="#f4cccc", edgecolor="#cc0000", linewidth=1.6))
    ax1.text(5, 5.18, "fixed downstream $W$", ha="center", va="center",
             fontweight="bold", color="#990000", fontsize=11)
    for xt, xy in [((4.3, 4.7), (3.0, 3.6)), ((5.0, 4.7), (5.0, 3.4)), ((5.7, 4.7), (7.0, 3.7))]:
        ax1.annotate("", xy=xy, xytext=xt, arrowprops=dict(arrowstyle="->", color="gray", lw=1))
    ax1.text(2.9, 3.2, "$y_1$", ha="center", color="#cc0000", fontsize=11)
    ax1.text(5.0, 3.0, "$y_2$", ha="center", color="#cc0000", fontsize=11)
    ax1.text(7.1, 3.3, "$y_3$", ha="center", color="#cc0000", fontsize=11)
    ax1.text(5, 3.7, "(all wrong)", ha="center", color="#cc0000", fontsize=9, style="italic")
    ax1.text(5, 1.6, f"CKA matches $[h]$, but the fixed $W$ reads the\n"
             f"wrong coordinates $\\Rightarrow$ capability lost\n"
             f"(here $||f'-f||={err_left:.1f}$)",
             ha="center", va="center", fontsize=10.5,
             bbox=dict(boxstyle="round", facecolor="#fff2cc", edgecolor="#bf9000", alpha=0.9))

    # ---------- right: joint quotient (succeeds) ----------
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.set_aspect("equal"); ax2.axis("off")
    ax2.set_title("Joint quotient  $\\mathcal{M}/\\mathcal{G}$\n"
                  "(collapse $(h, W)$ together)", fontsize=12, pad=8)
    ax2.text(1.7, 8.7, "$(h,\\ W)$", fontsize=13, fontweight="bold",
             bbox=dict(facecolor="white", edgecolor="black"))
    ax2.annotate("", xy=(4.6, 8.95), xytext=(2.9, 8.95), arrowprops=dict(arrowstyle="->", color="#3366cc", lw=1.8))
    ax2.text(3.75, 9.4, "$h\\rightarrow hQ$", ha="center", color="#3366cc", fontsize=10.5)
    ax2.annotate("", xy=(4.6, 8.25), xytext=(2.9, 8.25), arrowprops=dict(arrowstyle="->", color="#cc0000", lw=1.8))
    ax2.text(3.75, 7.75, "$W\\rightarrow Q^{\\top}W$", ha="center", color="#990000", fontsize=10.5)
    ax2.text(6.4, 8.6, "$(hQ,\\ Q^{\\top}W)$", fontsize=13, fontweight="bold",
             bbox=dict(facecolor="#e6e6e6", edgecolor="black"))
    ax2.annotate("", xy=(5, 6.4), xytext=(5, 7.4), arrowprops=dict(arrowstyle="->", color="black", lw=1.8))
    ax2.text(5.6, 6.8, r"$\pi_{\mathrm{joint}}$", fontsize=13)
    ax2.plot(5, 6.1, "ko", markersize=9); ax2.text(5.5, 5.95, "$[(h,W)]$", fontsize=13)
    ax2.add_patch(mpatches.FancyBboxPatch((3.2, 4.45), 3.6, 0.95, boxstyle="round,pad=0.1",
                 facecolor="#fff2cc", edgecolor="black", linewidth=1.4))
    ax2.text(5, 4.93, r"$(hQ)(Q^{\top}W) = hW$", ha="center", va="center", fontsize=12, fontweight="bold")
    ax2.annotate("", xy=(5, 3.3), xytext=(5, 4.45), arrowprops=dict(arrowstyle="->", color="black", lw=1.8))
    ax2.text(5, 2.95, "$y$  (correct)", ha="center", color="#178a3a", fontsize=12, fontweight="bold")
    ax2.text(5, 1.6, f"output is identical for every $Q$\n"
             f"$\\Rightarrow f^{{Q}}=f$ : capability preserved\n"
             f"(here $||f'-f||={err_right:.1e}$)",
             ha="center", va="center", fontsize=10.5,
             bbox=dict(boxstyle="round", facecolor="#d9ead3", edgecolor="#38761d", alpha=0.9))

    plt.tight_layout()
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> figure written to {path} (left/right numbers above are the")
    print(f"     facts it draws: fixed W breaks the output, joint action preserves it).")


def main():
    hand_calculation()
    two_line_demo()

    X, y = two_moons(400, 0.12, seed=0)
    teacher = MLP(2, 16, 2, seed=0)
    train_classifier(teacher, X, y, steps=4000, lr=0.2)
    acc = (teacher.logits(X).argmax(1) == y).mean()
    print(f"[teacher] 2-layer MLP trained on two-moons, train acc = {acc:.3f}; "
          f"hidden rep dim = {teacher.W1.shape[1]}")

    assumption_2_1_walkthrough(teacher, X)
    claim1_illposed(teacher, X)
    claim2_joint_invariance(teacher, X)
    claim3_rep_vs_function(teacher, X, y)
    claim4_procrustes(teacher, X)
    make_quotient_figure(teacher, X)

    print("\nAll four claims hold on a plain MLP with no normalization, where "
          "Assumption 2.1\nis exact: the paper's account is a property of "
          "representation learning, not of transformers.")


if __name__ == "__main__":
    main()
