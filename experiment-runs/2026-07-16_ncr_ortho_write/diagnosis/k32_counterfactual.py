"""Counterfactual: does forcing |lambda|=c* (orthogonalizing the entity block,
keeping learned phases) restore far-depth recovery? This is an in-silico
preview of the orthogonal-write fix. Uses repo's predicted_cos_curve_far.
Decomposes far-depth death into modulus-decay vs phase-drift."""
import json, sys, os
import numpy as np
sys.path.insert(0, "matrix-thinking/ncr")
sys.path.insert(0, "matrix-thinking/chapter2")
import analyze_zdump as az
import ncr_spectral as ns

def curve_at(A, Pi, keys, h):
    """mean predicted cosine at depth h (repo's direction-exact renorm power vs ideal)."""
    K = keys.shape[0]
    h_ref = h % K
    Ph = np.linalg.matrix_power(Pi, h_ref) if h_ref>0 else np.eye(Pi.shape[0])
    cs=[]
    for i in range(K):
        pa = ns._scaled_power_apply(A, keys[i], h)
        pp = Ph @ keys[i]; n=np.linalg.norm(pp)
        cs.append(0.0 if n<1e-12 else float(np.dot(pa, pp/n)))
    return float(np.mean(cs))

def orthogonalize_keep_phase(A):
    """Rebuild A with every eigenvalue pushed to the unit circle * c* (phases kept,
    moduli set equal). Uses eig; recombines real. c* = mean modulus."""
    ev, W = np.linalg.eig(A)
    cstar = float(np.mean(np.abs(ev)))
    ev_fixed = cstar * ev/np.clip(np.abs(ev),1e-12,None)   # keep phase, unit modulus*c*
    A2 = W @ np.diag(ev_fixed) @ np.linalg.inv(W)
    return np.real(A2)

def polar_orthogonal(A):
    """Nearest orthogonal matrix (polar factor) * c*: the true orthogonal-write target."""
    U,S,Vt = np.linalg.svd(A)
    cstar = float(np.mean(S))
    return cstar*(U@Vt)

CELLS = [
 ("HEALTHY K14@d16", "experiment-runs/2026-07-11_ncr_earlyln_scale/earlyln_K14_s0.json", 8*14-3),
 ("2K K16@d32 DEAD", "experiment-runs/2026-07-11_ncr_earlyln_scale/earlyln_K16_s0.json", 8*16-3),
 ("2K K24@d48 DEAD", "experiment-runs/2026-07-11_ncr_earlyln_scale/earlyln_K24_s0.json", 8*24-3),
 ("K32@d33 4x-best s2", "experiment-runs/2026-07-12_ncr_k32_budget/budget4x_earlyln_K32_s2.json", 8*32-3),
 ("K32@d33 4x s1", "experiment-runs/2026-07-12_ncr_k32_budget/budget4x_earlyln_K32_s1.json", 8*32-3),
]
print(f"{'cell':<22}{'h*':>5} | far-depth cos@h*   [as-is | fix-modulus | polar-orth]   also cos@h=20,40")
for label,path,hstar in CELLS:
    d=json.load(open(path)); K=d["K"]
    Z=d["z_dump"]["Z"]; Zi=d["z_dump"]["z_ideal"]
    a0=[]; a1=[]; a2=[]; a20=[]; a40=[]
    for Zr,Zir in zip(Z,Zi):
        Zm=np.asarray(Zr,float); Zim=np.asarray(Zir,float)
        sub=az.entity_subspace(Zim); U=sub["U"]
        A,_,_,_ = az.block_decompose(Zm,U,sub["V"])
        Pi=U.T@Zim@U
        keys,_,_ = az.synthetic_keys_from_pi(Pi)
        Afix = orthogonalize_keep_phase(A)
        Apol = polar_orthogonal(A)
        a0.append(curve_at(A,Pi,keys,hstar))
        a1.append(curve_at(Afix,Pi,keys,hstar))
        a2.append(curve_at(Apol,Pi,keys,hstar))
        a20.append(curve_at(Apol,Pi,keys,20)); a40.append(curve_at(Apol,Pi,keys,40))
    m=lambda x: float(np.mean(x))
    print(f"{label:<22}{hstar:>5} |   {m(a0):>6.3f} | {m(a1):>6.3f} | {m(a2):>6.3f}          polar@20={m(a20):.3f} @40={m(a40):.3f}")
