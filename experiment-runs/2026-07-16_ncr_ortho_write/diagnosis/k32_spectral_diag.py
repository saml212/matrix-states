"""K=32 write-failure spectral/leakage diagnosis (the never-done §11.6 analysis).
CPU-only, on EXISTING archived z_dump.Z. Reuses repo's audited analyzer +
independent eigen-analysis. Decomposes far-depth death into modulus-decay vs
eigenphase-drift vs cross-term-leakage."""
import json, glob, sys, os
import numpy as np
sys.path.insert(0, "matrix-thinking/ncr")
sys.path.insert(0, "matrix-thinking/chapter2")
import ncr_spectral as ns
import analyze_zdump as az

def analyze_cell(path, hstar):
    d = json.load(open(path))
    K, dd = d["K"], d["d"]
    Z = d["z_dump"]["Z"]; Zi = d["z_dump"]["z_ideal"]
    rows = []
    for Zr, Zir in zip(Z, Zi):
        Zm = np.asarray(Zr, float); Zim = np.asarray(Zir, float)
        sub = az.entity_subspace(Zim)
        U, V, k_eff = sub["U"], sub["V"], sub["k_eff"]
        A, B, C, Dblk = az.block_decompose(Zm, U, V)
        Pi = U.T @ Zim @ U
        c_star = float(np.sum(A*Pi)/max(np.sum(Pi*Pi),1e-12))
        # eigenspectrum of the restricted operator A (KxK, entity subspace)
        ev = np.linalg.eigvals(A)
        mod = np.abs(ev)                       # modulus: decay axis
        ev_unit = ev/np.clip(mod,1e-12,None)
        roots = np.exp(2j*np.pi*np.arange(A.shape[0])/A.shape[0])
        _, phres = az.match_eigenvalues(ev_unit, roots)   # phase/chord resid
        # normalize modulus by c_star (the ideal isotropic scale): entity modes
        # should have |lambda| == c_star exactly for a clean scaled cycle
        mod_rel = mod / max(abs(c_star),1e-12)
        sig = np.linalg.svd(A, compute_uv=False)
        rows.append(dict(
            k_eff=k_eff, c_star=c_star,
            mod_rel_min=float(mod_rel.min()), mod_rel_max=float(mod_rel.max()),
            mod_rel_mean=float(mod_rel.mean()), mod_rel_std=float(mod_rel.std()),
            phres_max=float(phres.max()), phres_mean=float(phres.mean()),
            A_cond=float(sig[0]/max(sig[-1],1e-12)),
            normB=float(np.linalg.norm(B)), normC=float(np.linalg.norm(C)),
            normD=float(np.linalg.norm(Dblk)),
            # decomposition of far-depth death:
            # (a) geometric modulus decay over h*: (min entity |lambda|/c*)^h*
            decay_hstar=float(mod_rel.min()**hstar) if mod_rel.min()<1 else 1.0,
            # (b) phase drift accumulated over h*: h* * mean phase resid (rad approx)
            phasedrift_hstar=float(hstar*phres.mean()),
            # root spacing at this K (rad); resolution needs phres < half spacing
            root_spacing=float(2*np.pi/A.shape[0]),
        ))
    def agg(k):
        v=[r[k] for r in rows]; return float(np.mean(v))
    keys=rows[0].keys()
    return K, dd, {k:agg(k) for k in keys}

CELLS = [
 ("HEALTHY K14@d16 (conv, far-depth works)","experiment-runs/2026-07-11_ncr_earlyln_scale/earlyln_K14_s0.json", 8*14-3),
 ("SPARE-COLLAPSE K15@d16 (DEAD)","experiment-runs/2026-07-11_ncr_earlyln_scale/earlyln_K15_s0.json", 8*15-3),
 ("2K K16@d32 (DEAD)","experiment-runs/2026-07-11_ncr_earlyln_scale/earlyln_K16_s0.json", 8*16-3),
 ("2K K24@d48 (DEAD)","experiment-runs/2026-07-11_ncr_earlyln_scale/earlyln_K24_s0.json", 8*24-3),
 ("TIGHT-SPARE WALL K32@d33 4x-best s2 (in-dist conv, far DEAD)","experiment-runs/2026-07-12_ncr_k32_budget/budget4x_earlyln_K32_s2.json", 8*32-3),
 ("TIGHT-SPARE WALL K32@d33 4x s1","experiment-runs/2026-07-12_ncr_k32_budget/budget4x_earlyln_K32_s1.json", 8*32-3),
 ("TIGHT-SPARE WALL K32@d33 1x s3","experiment-runs/2026-07-12_ncr_mappinglaw_wave1/dratio_earlyln_K32_s3.json", 8*32-3),
]
print(f"{'cell':<52}{'K':>4}{'d':>4}{'c*':>7}{'|λ|/c* min':>11}{'|λ|/c* mean':>12}{'phres_max':>10}{'phres_mn':>9}{'A_cond':>9}{'decay^h*':>10}{'drift·h*':>10}{'spacing':>8}")
for label,path,hstar in CELLS:
    if not os.path.exists(path):
        print(f"{label:<52} MISSING {path}"); continue
    K,dd,a = analyze_cell(path,hstar)
    print(f"{label:<52}{K:>4}{dd:>4}{a['c_star']:>7.2f}{a['mod_rel_min']:>11.3f}{a['mod_rel_mean']:>12.3f}{a['phres_max']:>10.3f}{a['phres_mean']:>9.3f}{a['A_cond']:>9.1f}{a['decay_hstar']:>10.2e}{a['phasedrift_hstar']:>10.1f}{a['root_spacing']:>8.3f}")
