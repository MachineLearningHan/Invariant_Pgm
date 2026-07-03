# PAPER: Loss defs (tab:losses) -- toy numpy illustration of L_abs vs L_cka vs L_rel. See RESULTS_TO_CODE.md
import numpy as np
rng = np.random.default_rng(0)
N, d = 64, 16
H_T = rng.standard_normal((N, d))     # teacher representation
W_T = rng.standard_normal((d, 2))     # teacher's downstream reader

def center(G):
    C = np.eye(N) - np.ones((N,N))/N; return C@G@C
def cka(A,B):
    GA,GB = center(A@A.T), center(B@B.T)
    return (GA*GB).sum()/(np.linalg.norm(GA)*np.linalg.norm(GB))
def softmax(z):
    z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)

def L_abs(Hs,Ht):  return ((Hs-Ht)**2).sum()/Hs.shape[0]
def L_rel(Hs,Ht):
    GA,GB=center(Hs@Hs.T),center(Ht@Ht.T); 
    GA/=np.linalg.norm(GA); 
    GB/=np.linalg.norm(GB)
    return ((GA-GB)**2).sum()
def L_cka(Hs,Ht):  return 1-cka(Hs,Ht)
# (C): each network uses ITS OWN reader -> the student that is the teacher in a
# rotated basis carries the counter-rotated reader Q^T W_T, so outputs coincide.
def L_logit(Ps,Pt): return (Pt*(np.log(Pt)-np.log(Ps))).sum(1).mean()

def rand_orth(d,t):
    from scipy.linalg import logm,expm
    A=rng.standard_normal((d,d)); Q,_=np.linalg.qr(A); return expm(t*logm(Q).real).real

P_T = softmax(H_T@W_T)
print(f"{'t':>5} | {'L_abs':>8} | {'L_rel':>9} | {'1-CKA':>9} | {'L_logit':>9}")
print("-"*54)
for t in [0.0,0.25,0.5,0.75,1.0]:
    Q  = rand_orth(d,t)
    Hs = H_T@Q                    # student = teacher in a rotated basis (same class)
    Ws = Q.T@W_T                  # its reader is counter-rotated (joint action)
    Ps = softmax(Hs@Ws)           # student's own output
    print(f"{t:>5.2f} | {L_abs(Hs,H_T):>8.2f} | {L_rel(Hs,H_T):>9.1e} | "
          f"{L_cka(Hs,H_T):>9.1e} | {L_logit(Ps,P_T):>9.1e}")
