"""Quantify non-normality of the written entity operator + find the depth-frontier
that an orthogonal (polar) write would sustain at K=32. Robustness across all
K32 budget seeds."""
import json, glob, sys
import numpy as np
sys.path.insert(0,"matrix-thinking/ncr"); sys.path.insert(0,"matrix-thinking/chapter2")
import analyze_zdump as az, ncr_spectral as ns

def curve_at(A,Pi,keys,h):
    K=keys.shape[0]; hr=h%K
    Ph=np.linalg.matrix_power(Pi,hr) if hr>0 else np.eye(Pi.shape[0])
    cs=[]
    for i in range(K):
        pa=ns._scaled_power_apply(A,keys[i],h); pp=Ph@keys[i]; n=np.linalg.norm(pp)
        cs.append(0.0 if n<1e-12 else float(np.dot(pa,pp/n)))
    return float(np.mean(cs))

def nonnormality(A):
    # departure from normality (Henrici): ||AA^T - A^TA||_F / ||A||_F^2
    comm = A@A.T - A.T@A
    return float(np.linalg.norm(comm)/max(np.linalg.norm(A)**2,1e-12))

def polar(A):
    U,S,Vt=np.linalg.svd(A); return float(np.mean(S))*(U@Vt)

def frontier(A,Pi,keys,bar=0.9,hmax=260):
    last=0
    for h in range(1,hmax+1):
        if curve_at(A,Pi,keys,h)>=bar: last=h
        else: break
    return last

groups={
 "K14@d16 (healthy)":sorted(glob.glob("experiment-runs/2026-07-11_ncr_earlyln_scale/earlyln_K14_s*.json")),
 "K16@d32 2K (dead)":sorted(glob.glob("experiment-runs/2026-07-11_ncr_earlyln_scale/earlyln_K16_s*.json")),
 "K24@d48 2K (dead)":sorted(glob.glob("experiment-runs/2026-07-11_ncr_earlyln_scale/earlyln_K24_s*.json")),
 "K32@d33 2x":sorted(glob.glob("experiment-runs/2026-07-12_ncr_k32_budget/budget2x_earlyln_K32_s*.json")),
 "K32@d33 4x":sorted(glob.glob("experiment-runs/2026-07-12_ncr_k32_budget/budget4x_earlyln_K32_s*.json")),
}
print(f"{'group':<20}{'nonnorm(asis)':>14}{'A_cond':>8}{'front_asis@.9':>14}{'front_polar@.9':>15}{'polar@h*':>9}")
for g,fs in groups.items():
    nn=[];cond=[];fa=[];fp=[];ph=[]
    for f in fs:
        d=json.load(open(f)); K=d["K"]; hstar=8*K-3
        if "z_dump" not in d or "Z" not in d.get("z_dump",{}):
            continue
        for Zr,Zir in zip(d["z_dump"]["Z"],d["z_dump"]["z_ideal"]):
            Zm=np.asarray(Zr,float); Zim=np.asarray(Zir,float)
            sub=az.entity_subspace(Zim); U=sub["U"]
            A,_,_,_=az.block_decompose(Zm,U,sub["V"]); Pi=U.T@Zim@U
            keys,_,_=az.synthetic_keys_from_pi(Pi)
            nn.append(nonnormality(A));
            s=np.linalg.svd(A,compute_uv=False); cond.append(s[0]/max(s[-1],1e-12))
            fa.append(frontier(A,Pi,keys)); fp.append(frontier(polar(A),Pi,keys,hmax=min(hstar,260)))
            ph.append(curve_at(polar(A),Pi,keys,hstar))
    m=lambda x: float(np.mean(x))
    print(f"{g:<20}{m(nn):>14.4f}{m(cond):>8.1f}{m(fa):>14.1f}{m(fp):>15.1f}{m(ph):>9.3f}")
