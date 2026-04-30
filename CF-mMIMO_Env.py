# ============================================================
# UC CF-mMIMO with Reduced Pilot Contamination (Orthogonal pilots)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, solve
rng = np.random.default_rng(7)

# -------------------- System parameters --------------------
L, K, N = 100, 35, 4
AREA, AP_HEIGHT, UE_HEIGHT = 1000.0, 10.0, 1.5

# Noise
BW, NF_dB, N0_dBm_perHz = 20e6, 7.0, -174.0
noise_power_dBm = N0_dBm_perHz + 10*np.log10(BW) + NF_dB
sigma2 = 10**(noise_power_dBm/10)/1000.0

# Powers
P_UL_user, P_pilot, P_DL_AP_max = 0.1, 0.1, 0.2

# Coherence block
tau_c, tau_u = 200, 95
tau_p_ortho = K
tau_d = tau_c - tau_p_ortho - tau_u
assert tau_d > 0, "Increase tau_c or reduce tau_u for orthogonal pilots."

# Cluster
CLUSTER_SIZE, rho_corr = 10, 0.7

# Pathloss
PL_intercept_dB, PL_exponent, shadow_sigma_dB = -30.5, 36.7, 3.0

# Simulation realizations
NUM_REALIZATIONS = 200

# -------------------- Helpers --------------------
def toeplitz_exp(rho, N):
    idx = np.arange(N)
    return rho ** np.abs(np.subtract.outer(idx, idx))

def gen_positions(L, K, area):
    ap_xy = rng.uniform(0, area, size=(L, 2))
    ue_xy = rng.uniform(0, area, size=(K, 2))
    return ap_xy, ue_xy

def lsfc_matrix(ap_xy, ue_xy):
    d2d = np.linalg.norm(ap_xy[:, None, :] - ue_xy[None, :, :], axis=-1)
    d3d = np.sqrt(d2d**2 + (AP_HEIGHT-UE_HEIGHT)**2)
    F = rng.normal(0.0, shadow_sigma_dB, size=(L, K))
    beta_dB = PL_intercept_dB - PL_exponent * np.log10(d3d) + F
    return 10**(beta_dB/10.0)

def assign_pilots_orthogonal(K):
    return np.arange(K)

def sample_channel(Rcorr, beta_lk):
    chol = np.linalg.cholesky(Rcorr)
    Z = (rng.standard_normal((L, K, N)) + 1j*rng.standard_normal((L, K, N)))/np.sqrt(2.0)
    h = Z @ chol.T
    h *= np.sqrt(beta_lk[..., None])
    return h

def lp_mmse_estimation(h, Rcorr, beta_lk, pilots, P_pilot, tau_p):
    I = np.eye(N, dtype=np.complex128)
    groups = [np.where(pilots == t)[0] for t in range(tau_p)]
    hhat = np.zeros_like(h, dtype=np.complex128)
    Cerr_trace = np.zeros((L, K), dtype=np.float64)
    sqrt_pt = np.sqrt(P_pilot * tau_p)
    for l in range(L):
        for idxs in groups:
            if len(idxs) == 0:
                continue
            y_lt = sqrt_pt * np.sum(h[l, idxs, :], axis=0)
            n = np.sqrt(sigma2/2)*(rng.standard_normal(N) + 1j*rng.standard_normal(N))
            y_lt = y_lt + n
            S = (P_pilot * tau_p) * (np.sum(beta_lk[l, idxs]) * Rcorr) + sigma2 * I
            S_inv = inv(S)
            for k in idxs:
                R_kl = beta_lk[l, k] * Rcorr
                hhat[l, k, :] = sqrt_pt * (R_kl @ (S_inv @ y_lt))
                C = R_kl - (P_pilot * tau_p) * (R_kl @ (S_inv @ R_kl))
                Cerr_trace[l, k] = float(np.real(np.trace(C)))
    return hhat, Cerr_trace

def local_lpmmse_combiner(hhat, Cerr_trace, rho_ul_users):
    I = np.eye(N, dtype=np.complex128)
    v = np.zeros_like(hhat, dtype=np.complex128)
    for l in range(L):
        G = np.zeros((N, N), dtype=np.complex128)
        trC = 0.0
        for i in range(K):
            hh = hhat[l, i, :][:, None]
            G += rho_ul_users[i] * (hh @ hh.conj().T)
            trC += rho_ul_users[i] * Cerr_trace[l, i]
        A_inv = inv(G + (trC/N + sigma2) * I)
        v[l, :, :] = (A_inv @ hhat[l, :, :].T).T
    return v

def form_user_clusters(beta_lk, cluster_size):
    M_k = []
    for k in range(K):
        top = np.argsort(beta_lk[:, k])[-cluster_size:]
        M_k.append(np.sort(top))
    D_l = [set() for _ in range(L)]
    for k in range(K):
        for l in M_k[k]:
            D_l[l].add(k)
    D_l = [sorted(list(s)) for s in D_l]
    return M_k, D_l

def ul_sinr(h, v, M_k, rho_ul_users):
    sinr = np.zeros(K)
    for k in range(K):
        cluster = M_k[k]
        gk = np.zeros((len(cluster),), dtype=np.complex128)
        G = np.zeros((len(cluster), K), dtype=np.complex128)
        vnorm = np.zeros((len(cluster),), dtype=np.float64)
        for idx, l in enumerate(cluster):
            vkl = v[l, k, :]
            vnorm[idx] = float(np.real(np.vdot(vkl, vkl)))
            G[idx, :] = vkl.conj().T @ h[l, :, :].T
            gk[idx] = G[idx, k]
        R_i = (G * rho_ul_users).dot(G.conj().T) + sigma2 * np.diag(vnorm)
        try: alpha = solve(R_i, gk)
        except np.linalg.LinAlgError: alpha = np.linalg.pinv(R_i) @ gk
        num = rho_ul_users[k] * np.abs(alpha.conj().T @ gk)**2
        den = (alpha.conj().T @ (R_i - rho_ul_users[k] * np.outer(gk, gk.conj())) @ alpha).real
        sinr[k] = float((num/den).real) if den > 0 else 0.0
    return sinr

def dl_sinr(h, v, M_k, D_l, P_DL_AP_max, rho_ul_users):
    norms = np.linalg.norm(v, axis=2) + 1e-12
    f = v / norms[:, :, None]
    sinr = np.zeros(K)
    for k in range(K):
        desired, interf = 0+0j, 0.0
        for i in range(K):
            ls = [l for l in range(L) if i in D_l[l]]
            if not ls: continue
            Hl = h[ls, k, :]
            fl = f[ls, i, :]
            eff = np.sum(np.sum(Hl.conj() * fl, axis=1))
            if i == k: desired = eff
            else: interf += np.abs(eff)**2
        sinr[k] = float((np.abs(desired)**2 / (interf + sigma2)).real)
    return sinr

def ecdf(x):
    xs = np.sort(x); ys = np.linspace(0, 1, len(xs), endpoint=False); return xs, ys

# -------------------- Simulation --------------------
ap_xy, ue_xy = gen_positions(L, K, AREA)
beta_lk = lsfc_matrix(ap_xy, ue_xy)
Rcorr = toeplitz_exp(rho_corr, N)
M_k, D_l = form_user_clusters(beta_lk, CLUSTER_SIZE)
rho_ul_users = np.full(K, P_UL_user)

UL_SINR_new, DL_SINR_new = [], []
for r in range(NUM_REALIZATIONS):
    h = sample_channel(Rcorr, beta_lk)
    pilots = assign_pilots_orthogonal(K)
    hhat, Cerr_tr = lp_mmse_estimation(h, Rcorr, beta_lk, pilots, P_pilot, tau_p_ortho)
    v = local_lpmmse_combiner(hhat, Cerr_tr, rho_ul_users)
    UL_SINR_new.append(ul_sinr(h, v, M_k, rho_ul_users))
    DL_SINR_new.append(dl_sinr(h, v, M_k, D_l, P_DL_AP_max, rho_ul_users))

UL_SINR_new_dB = 10*np.log10(np.maximum(np.vstack(UL_SINR_new).flatten(), 1e-12))
DL_SINR_new_dB = 10*np.log10(np.maximum(np.vstack(DL_SINR_new).flatten(), 1e-12))

# -------------------- Plots --------------------
plt.figure(figsize=(6,4))
xN,yN = ecdf(UL_SINR_new_dB)
plt.plot(xN,yN,label="UL SINR (dB) - orthogonal pilots")
plt.xlabel("SINR (dB)"); plt.ylabel("CDF"); plt.title("UL SINR CDF")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

plt.figure(figsize=(6,4))
xN,yN = ecdf(DL_SINR_new_dB)
plt.plot(xN,yN,label="DL SINR (dB) - orthogonal pilots")
plt.xlabel("SINR (dB)"); plt.ylabel("CDF"); plt.title("DL SINR CDF")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# -------------------- Percentiles --------------------
def pct(a, p): return float(np.percentile(a, p))
print("=== UL SINR (dB) with ORTHOGONAL pilots ===")
print(f"p5={pct(UL_SINR_new_dB,5):.2f}, p50={pct(UL_SINR_new_dB,50):.2f}, p95={pct(UL_SINR_new_dB,95):.2f}")
print("=== DL SINR (dB) with ORTHOGONAL pilots ===")
print(f"p5={pct(DL_SINR_new_dB,5):.2f}, p50={pct(DL_SINR_new_dB,50):.2f}, p95={pct(DL_SINR_new_dB,95):.2f}")
