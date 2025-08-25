#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# ===================== Toy-only utils =====================
def is_prime(n: int) -> bool:
    if n < 2: return False
    small = [2,3,5,7,11,13,17,19,23,29]
    for sp in small:
        if n % sp == 0: return n == sp
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2; s += 1
    for a in [2,3,5,7,11,13,17]:
        if a % n == 0: continue
        x = pow(a, d, n)
        if x in (1, n-1): continue
        for _ in range(s-1):
            x = (x*x) % n
            if x == n-1:
                break
        else:
            return False
    return True

# ===================== ECC over F_p (toy) =====================
O = (None, None)

def inv_mod(x, p): return pow(x % p, p-2, p)

def is_on_curve(Pt, p, a, b):
    if Pt == O: return True
    x, y = Pt
    return (y*y - (x*x*x + a*x + b)) % p == 0

def ec_add(P, Q, p, a):
    if P == O: return Q
    if Q == O: return P
    x1,y1 = P; x2,y2 = Q
    if x1 == x2 and (y1 + y2) % p == 0: return O
    if P != Q:
        lam = ((y2 - y1) * inv_mod((x2 - x1), p)) % p
    else:
        if y1 % p == 0: return O
        lam = ((3*x1*x1 + a) * inv_mod((2*y1), p)) % p
    x3 = (lam*lam - x1 - x2) % p
    y3 = (lam*(x1 - x3) - y1) % p
    return (x3, y3)

def ec_mul(P, n, p, a):
    R, Q = O, P
    while n:
        if n & 1: R = ec_add(R, Q, p, a)
        Q = ec_add(Q, Q, p, a)
        n >>= 1
    return R

def point_order(P, p, a, max_iters=10000):
    if P == O: return 1
    R = O
    for k in range(1, max_iters+1):
        R = ec_add(R, P, p, a)
        if R == O: return k
    return None

# ===================== Envelopes =====================
def long_env(N, eps=0.30, f1=7, f2=11, phi1=0.0, phi2=0.0, mode="sum"):
    k = np.arange(N, dtype=float)
    if mode == "sum":
        A = 1 + 0.5*eps*(np.cos(2*np.pi*f1*k/N + phi1) + np.cos(2*np.pi*f2*k/N + phi2))
    else:
        A = (1 + eps*np.cos(2*np.pi*f1*k/N + phi1))*(1 + eps*np.cos(2*np.pi*f2*k/N + phi2))
    A = np.maximum(A, 1e-9)
    return A / A.mean()

def log_clock(N, beta=0.35, f_tau=1.2, phi_tau=0.0, T_c=32.0):
    k = np.arange(N, dtype=float)
    tau = T_c * np.log1p(k / T_c)
    tau = (tau - tau.min()) / (tau.max() - tau.min() + 1e-12)
    M = 1 + beta*np.cos(2*np.pi*f_tau*tau + phi_tau)
    M = np.maximum(M, 1e-9)
    return M / M.mean()

def comp_amp(N, f_tau, T_c, eps=0.3, f1=7, f2=11, beta=0.35, env_mode="sum"):
    A = long_env(N, eps=eps, f1=f1, f2=f2, mode=env_mode) * log_clock(N, beta=beta, f_tau=f_tau, T_c=T_c)
    return A / A.mean()

# ===================== Spectrum helpers (high-pass + DC mask) =====================
def highpass(z, method="diff"):
    if method == "diff":
        return np.diff(z, prepend=z[0])   # discrete derivative
    if method == "demean":
        return z - z.mean()
    return z

def detect_bins(prob, K=5, guard=1):
    N = len(prob)
    p = prob.copy(); p[0] = 0.0  # never select DC
    idx_sorted = np.argsort(p)[::-1]
    chosen, taken = [], np.zeros(N, bool)
    taken[0] = True
    for k in idx_sorted:
        if taken[k]: continue
        chosen.append(int(k))
        for d in range(-guard, guard+1):
            taken[(k+d) % N] = True
        if len(chosen) >= K: break
    return sorted(chosen)

def sharpness(prob, bins, bw=1):
    N = len(prob)
    mask = np.zeros(N, bool)
    for b in bins:
        for d in range(-bw, bw+1):
            mask[(b+d) % N] = True
    sig = prob[mask].sum()
    noi = np.median(prob[~mask]) * (~mask).sum()
    return sig - 0.5*noi

def near_zero_from_global(prob):
    N = len(prob)
    p = prob.copy(); p[0] = 0.0
    k_star = int(np.argmax(p))
    k_twin = (-k_star) % N
    return min(k_star, k_twin), k_star, k_twin

# ===================== Two-stage search =====================
def coarse(u, N, f_rng, T_rng, nf=120, nT=90, eps=0.3, f1=7, f2=11, beta=0.35,
           env_mode="sum", K=5, guard=1, bw=1, use_window=True):
    fvals = np.linspace(*f_rng, nf)
    Tvals = np.linspace(*T_rng, nT)
    best = (-np.inf, None, None, None, None)
    w = 0.5*(1 - np.cos(2*np.pi*np.arange(N)/(N-1))) if use_window else np.ones(N)
    for Tc in Tvals:
        for f in fvals:
            A = comp_amp(N, f, Tc, eps=eps, f1=f1, f2=f2, beta=beta, env_mode=env_mode)
            s = (A * w) * u
            S = np.fft.fft(s, n=N)
            prob = np.abs(S)**2
            prob[0] = 0.0                 # DC mask
            prob /= prob.sum() + 1e-18
            bins = detect_bins(prob, K=K, guard=guard)
            sc = sharpness(prob, bins, bw=bw)
            if sc > best[0]:
                best = (sc, f, Tc, bins, prob)
    return best  # (score, f*, T*, bins*, prob*)

def refine(u, N, f0, T0, df=0.25, dT=10.0, nf=140, nT=100, **kwargs):
    f_rng = (max(0.001, f0 - df), f0 + df)
    T_rng = (max(1.0, T0 - dT), T0 + dT)
    return coarse(u, N, f_rng, T_rng, nf=nf, nT=nT, **kwargs)

# ===================== Main =====================
if __name__ == "__main__":
    # Toy curve & base point
    p, a, b = 97, 2, 3
    P = (3, 6)
    assert is_on_curve(P, p, a, b)
    ordP = point_order(P, p, a)

    # FFT grid
    N = 256

    # ECC-driven phasor sequence (unknown bins) + HIGH-PASS
    x = np.array([0 if (ec_mul(P, k, p, a) == O) else ec_mul(P, k, p, a)[0] for k in range(N)])
    u = np.exp(2j*np.pi * x / p)
    u = highpass(u, method="diff")          # remove DC bias at source

    # Two-stage search
    c_score, c_f, c_T, c_bins, _ = coarse(
        u, N, f_rng=(0.5, 5.0), T_rng=(16.0, 64.0),
        nf=120, nT=90, eps=0.30, f1=7, f2=11, beta=0.35,
        env_mode="sum", K=5, guard=1, bw=1, use_window=True
    )

    b_score, b_f, b_T, bins_found, prob_final = refine(
        u, N, c_f, c_T, df=0.25, dT=10.0, nf=140, nT=100,
        eps=0.30, f1=7, f2=11, beta=0.35,
        env_mode="sum", K=5, guard=1, bw=1, use_window=True
    )

    # Final spectrum (with DC masked)
    A = comp_amp(N, b_f, b_T, eps=0.30, f1=7, f2=11, beta=0.35, env_mode="sum")
    w = 0.5*(1 - np.cos(2*np.pi*np.arange(N)/(N-1)))
    s = (A * w) * u
    S = np.fft.fft(s, n=N)
    prob = np.abs(S)**2
    prob[0] = 0.0                          # mask DC
    prob /= prob.sum() + 1e-18

    # Spectral readout + residues
    k_ans, k_star, k_twin = near_zero_from_global(prob)
    ans_mod  = k_ans  % p
    star_mod = k_star % p
    twin_mod = k_twin % p
    ans_mod_ord = None if ordP is None else (k_ans % ordP)

    # Prints
    print("=== Toy ECC / Spectrum Lock (Unknown Bins, DC-suppressed) ===")
    print(f"p={p} | prime? {is_prime(p)} | Curve: y^2 = x^3 + {a}x + {b} (mod p)")
    print(f"P on curve? {is_on_curve(P, p, a, b)} | P={P} | order(P)≈{ordP}")
    print(f"Coarse best:  f_tau={c_f:.6f}, T_c={c_T:.6f}, score={c_score:.6f}, bins={c_bins}")
    print(f"Refined best: f_tau={b_f:.6f}, T_c={b_T:.6f}, score={b_score:.6f}, bins={bins_found}")
    print(f"Global peak bin: {k_star} (≡ {star_mod} mod {p})")
    print(f"Symmetric twin:  {k_twin} (≡ {twin_mod} mod {p})")
    if ans_mod_ord is None:
        print(f"Near-zero answer bin: {k_ans} (≡ {ans_mod} mod {p})")
    else:
        print(f"Near-zero answer bin: {k_ans} (≡ {ans_mod} mod {p}, ≡ {ans_mod_ord} mod ord(P)={ordP})")

    # Plot
    fig = plt.figure(figsize=(12,6))

    # (1) Spectrum with auto-annotation at the peak
    ax1 = fig.add_subplot(2,1,1)
    ax1.bar(np.arange(N), prob, label="Spectrum")
    for b in bins_found:
        ax1.axvline(b, color='r', ls='--', alpha=0.7)
    ax1.plot([k_star], [prob[k_star]], 'o', label=f"Global peak {k_star}")
    ax1.plot([k_twin], [prob[k_twin]], 'o', label=f"Twin {k_twin}")
    ax1.plot([k_ans], [prob[k_ans]], 's', label=f"Near-zero {k_ans}")

    # --- Auto-annotation with residues on the chosen peak ---
    ann_text = (fr"$k \equiv {ans_mod}\ (\mathrm{{mod}}\ {p}),\ "
                fr"k \equiv {ans_mod_ord}\ (\mathrm{{mod}}\ \mathrm{{ord}}(P)={ordP})$")
    ax1.annotate(
        ann_text,
        xy=(k_ans, prob[k_ans]),
        xytext=(k_ans + 8, prob[k_ans] + prob.max()*0.08),
        ha='left', va='bottom',
        arrowprops=dict(arrowstyle='->', lw=1)
    )

    ax1.set_xlabel("FFT bin"); ax1.set_ylabel("Probability")
    ax1.legend(loc="upper right")
    ax1.set_title(fr"Locked spectrum — best $f_\tau={b_f:.3f}$, $T_c={b_T:.2f}$ (DC suppressed)")

    # (2) Envelopes
    ax2 = fig.add_subplot(2,1,2)
    A_long = long_env(N, eps=0.30, f1=7, f2=11)
    A_tau  = log_clock(N, beta=0.35, f_tau=b_f, T_c=b_T)
    ax2.plot(A * w, label="Final envelope × window")
    ax2.plot(A_long, '--', label="Longitudinal")
    ax2.plot(A_tau,  ':', label="Log clock")
    ax2.set_xlabel("k"); ax2.set_ylabel("Amplitude"); ax2.legend(loc="upper right")
    ax2.set_title("Envelopes")
    # Show the reduced EC multiple (51P ≡ 1P = P)
    Q = ec_mul(P, 51, p, a)
    Q_red = ec_mul(P, 51 % ordP, p, a)
    print("51*P =", Q, "| reduced =", Q_red, "| equals P?", Q_red == P)

    # Save the annotated plot
    plt.savefig("locked_spectrum_annotated.png", dpi=160)
    print("Saved figure -> locked_spectrum_annotated.png")
    plt.tight_layout()
    plt.show()