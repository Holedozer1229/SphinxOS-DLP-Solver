#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scalar_waze_guided.py
---------------------
secp256k1 spectrum-lock + guided solve135 scan (with stop-on-hit) and rich spectrum reporting.

• Curve: y^2 = x^3 + 7 over F_p (secp256k1), p ≡ 3 (mod 4)
• Two-stage spectrum lock (coarse → refine), DC-masked, λ-vertex angle fold (λ=0.33333333326)
• Guided subranges: top-K spectral bins → prioritized windows over the scalar range
• Address target: Base58Check P2PKH or explicit HASH160 (hex)
• Pubkey hashing: compressed or uncompressed
• Strided iterator seeded by --seed; checkpoint JSON; ETA with EMA smoothing; stop-on-hit

One-liners:
  spectrum only (print + files):
    python3 scalar_waze_guided.py spectrum-lock --grid 256 --topk 8 \
      --spectrum-base spec

  guided solve (uncompressed pubkey, P135 window):
    python3 scalar_waze_guided.py solve135 \
      --address 16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v \
      --range '0x4000000000000000000000000000000000:0x7fffffffffffffffffffffffffffffffff' \
      --hash-pub uncompressed \
      --grid 256 --topk 8 --bin-span 2 --bin-halo-frac 0.22 \
      --progress-sec 5 --eta-alpha 0.45 \
      --checkpoint-file jadon_dump_p135_guided.json --checkpoint-sec 300 \
      --stop-on-hit 1
"""

import argparse, os, sys, time, json, math, csv, hashlib, struct, random
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt

# ===================== secp256k1 constants =====================
K1_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
K1_A = 0
K1_B = 7
K1_Gx = 55066263022277343669578718895168534326250603453777594175500187360389116729240
K1_Gy = 32670510020758816978083085130507043184471273380659243275938904335757337482424
K1_G  = (K1_Gx, K1_Gy)
K1_N  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
O = (None, None)

# ===================== EC arithmetic =====================
def inv_mod(x: int, p: int) -> int:
    return pow(x % p, p - 2, p)

def is_on_curve(Pt: Tuple[Optional[int], Optional[int]], p: int, a: int, b: int) -> bool:
    if Pt == O: return True
    x, y = Pt
    return (y*y - (x*x*x + a*x + b)) % p == 0

def ec_add(P, Q, p, a):
    if P == O: return Q
    if Q == O: return P
    x1,y1 = P; x2,y2 = Q
    if x1 == x2 and (y1 + y2) % p == 0: 
        return O
    if P != Q:
        lam = ((y2 - y1) * inv_mod((x2 - x1), p)) % p
    else:
        if y1 % p == 0: 
            return O
        lam = ((3*x1*x1 + a) * inv_mod((2*y1), p)) % p
    x3 = (lam*lam - x1 - x2) % p
    y3 = (lam*(x1 - x3) - y1) % p
    return (x3, y3)

def ec_mul(P, n, p, a):
    R, Q = O, P
    while n:
        if n & 1: 
            R = ec_add(R, Q, p, a)
        Q = ec_add(Q, Q, p, a)
        n >>= 1
    return R

# ===================== λ-vertex angle (triangular fold) =====================
def _rep(a: int, p: int) -> int:
    return a if a <= (p >> 1) else a - p

def angle_feature(x: int, y: int, p: int, lam: float = 0.33333333326, fold3: bool = True):
    xs = _rep(x, p); ys = _rep(y, p)
    fx = xs / float(p); fy = ys / float(p)
    th = 2.0 * math.pi * lam
    c, s = math.cos(th), math.sin(th)
    xr = c*fx - s*fy
    yr = s*fx + c*fy
    ang = math.atan2(yr, xr)  # (-π, π]
    if fold3:
        period = 2.0 * math.pi / 3.0
        k = round(ang / period)
        folded = ang - k * period
        score = 1.0 - abs(folded) / (period / 2.0)
        if score < 0.0: score = 0.0
        return folded, score
    else:
        return ang, 1.0 - abs(ang) / math.pi

# ===================== FFT envelopes (unchanged logic) =====================
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

def highpass(z, method="diff"):
    if method == "diff":
        return np.diff(z, prepend=z[0])
    if method == "demean":
        return z - z.mean()
    return z

def detect_bins(prob, K=5, guard=1):
    N = len(prob)
    p = prob.copy(); p[0] = 0.0
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
            prob[0] = 0.0
            prob /= prob.sum() + 1e-18
            bins = detect_bins(prob, K=K, guard=guard)
            sc = sharpness(prob, bins, bw=bw)
            if sc > best[0]:
                best = (sc, f, Tc, bins, prob)
    return best

def refine(u, N, f0, T0, df=0.25, dT=10.0, nf=140, nT=100, **kwargs):
    f_rng = (max(0.001, f0 - df), f0 + df)
    T_rng = (max(1.0, T0 - dT), T0 + dT)
    return coarse(u, N, f_rng, T_rng, nf=nf, nT=nT, **kwargs)

# ===================== Spectrum reporting =====================
def _bin_residues(k: int):
    mod12 = k % 12
    mod256 = k % 256
    heavy = mod12 in {0,3,6,9}
    return mod12, mod256, heavy

def _ascii_hist(prob, width=50, max_bins=80):
    N = len(prob)
    mx = float(prob.max()) if prob.size else 1.0
    lines = []
    step = max(1, N // max_bins)
    for i in range(0, N, step):
        p = float(prob[i])
        bar = "#" * int((p / (mx + 1e-18)) * width)
        lines.append(f"{i:4d} | {bar} {p:.6g}")
    return "\n".join(lines)

def spectrum_report(
    title: str,
    prob: np.ndarray,
    bins: list,
    grid_N: int,
    f_tau: float,
    T_c: float,
    score: float,
    lam: float,
    print_ascii: bool = True,
    topk: int = 10,
    save_png: Optional[str] = None,
    save_json: Optional[str] = None,
    save_csv: Optional[str] = None,
):
    probs = prob.copy().astype(float)
    if probs.size == 0:
        print(f"[{title}] empty spectrum."); return
    total = probs.sum() + 1e-18
    order = np.argsort(probs)[::-1]
    top = [(int(k), float(probs[k]/total)) for k in order[:topk]]

    print(f"\n=== {title} spectrum ===")
    print(f"grid={grid_N}  fτ={f_tau:.6f}  Tc={T_c:.6f}  score={score:.6f}  DC-masked=True  λ={lam}")
    print(f"Top-{topk} bins (k, weight): {[(k, round(w,12)) for (k,w) in top]}")
    details = []
    for k, w in top:
        m12, m256, heavy = _bin_residues(k)
        details.append({"k":k, "weight":w, "k_mod_12":m12, "k_mod_256":m256, "heavy_mod12":heavy})
    print("Top bin details:")
    for d in details:
        print(f"  k={d['k']:4d}  w={d['weight']:.12f}  mod12={d['k_mod_12']:2d}  "
              f"mod256={d['k_mod_256']:3d}  heavy={str(d['heavy_mod12']).lower()}")
    print(f"Guided-bin seeds (from detector): {bins}")

    if print_ascii:
        print("\nASCII histogram (thinned):")
        print(_ascii_hist(probs, width=40, max_bins=96))

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    if save_json:
        payload = {
            "title": title, "stamp_utc": stamp, "grid": grid_N,
            "f_tau": f_tau, "T_c": T_c, "score": score, "lambda": lam,
            "top": details, "guided_bins": list(map(int, bins)),
            "prob_len": int(len(probs))
        }
        with open(save_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[{title}] wrote JSON -> {save_json}")

    if save_csv:
        with open(save_csv, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["bin","prob_norm"])
            for k in range(len(probs)):
                wtr.writerow([k, probs[k]/total])
        print(f"[{title}] wrote CSV  -> {save_csv}")

    if save_png:
        fig = plt.figure(figsize=(10,3))
        ax = fig.add_subplot(1,1,1)
        ax.bar(np.arange(len(probs)), probs/total)
        ax.set_title(f"{title}: N={grid_N}, fτ={f_tau:.3f}, Tc={T_c:.2f}")
        ax.set_xlabel("bin"); ax.set_ylabel("weight")
        for b in bins:
            ax.axvline(b, color='r', ls='--', alpha=0.6)
        fig.tight_layout()
        plt.savefig(save_png, dpi=140)
        plt.close(fig)
        print(f"[{title}] wrote PNG  -> {save_png}")

# ===================== BTC address / HASH160 utils =====================
_B58 = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

def _b58check_decode(s: str) -> bytes:
    n = 0
    for ch in s.encode():
        n *= 58
        if ch not in _B58:
            raise ValueError("invalid base58 character")
        n += _B58.index(ch)
    full = n.to_bytes(25, "big")
    data, cksum = full[:-4], full[-4:]
    if hashlib.sha256(hashlib.sha256(data).digest()).digest()[:4] != cksum:
        raise ValueError("bad base58 checksum")
    return data  # version(1) + payload(20)

def _b58check_encode(version: int, payload20: bytes) -> str:
    data = bytes([version]) + payload20
    cksum = hashlib.sha256(hashlib.sha256(data).digest()).digest()[:4]
    full = data + cksum
    n = int.from_bytes(full, "big")
    out = b""
    while n > 0:
        n, r = divmod(n, 58)
        out = _B58[r] + out
    # leading zeros
    for b in full:
        if b == 0:
            out = b"1" + out
        else:
            break
    return out.decode()

def _hash160(pubkey_bytes: bytes) -> bytes:
    h1 = hashlib.sha256(pubkey_bytes).digest()
    try:
        h2 = hashlib.new("ripemd160", h1).digest()
    except Exception as e:
        raise RuntimeError("RIPEMD160 not available in this Python build") from e
    return h2  # 20 bytes

def _ser_pubkey(P: Tuple[int,int], compressed: bool) -> bytes:
    x, y = P
    x32 = x.to_bytes(32, "big")
    if not compressed:
        return b"\x04" + x32 + y.to_bytes(32, "big")
    prefix = b"\x02" if (y % 2 == 0) else b"\x03"
    return prefix + x32

def target_from_args(args) -> Tuple[str, bytes, bool]:
    """
    Returns (what, hash20, compressed_flag)
      what: "addr" or "hash160"
    """
    if args.address:
        data = _b58check_decode(args.address)
        ver, h160 = data[0], data[1:]
        if ver not in (0x00, ):
            raise ValueError("Only P2PKH mainnet supported")
        return "addr", h160, (args.hash_pub == "compressed")
    if args.hash160:
        h = bytes.fromhex(args.hash160)
        if len(h) != 20:
            raise ValueError("--hash160 must be 20 bytes hex")
        return "hash160", h, (args.hash_pub == "compressed")
    raise ValueError("Provide --address or --hash160")

# ===================== Range / stride / windows =====================
def parse_int_maybe_hex(s: str) -> int:
    s = s.strip().lower()
    if s.startswith("0x"): return int(s, 16)
    # detect bare hex (no 'x', but contains a-f):
    if all(c in "0123456789abcdef" for c in s) and any(c in "abcdef" for c in s):
        return int(s, 16)
    return int(s, 10)

def parse_range(r: str) -> Tuple[int,int]:
    a, b = r.split(":")
    start = parse_int_maybe_hex(a)
    end   = parse_int_maybe_hex(b)
    if not (0 <= start < end):
        raise ValueError("bad --range")
    return start, end

def derive_stride_and_offset(seed_hex: str, length: int) -> Tuple[int,int]:
    if seed_hex is None or seed_hex == "":
        seed_hex = "01"
    seed = int(seed_hex, 16)
    # derive stride from seed via SHA256 -> int
    sbytes = hashlib.sha256(seed.to_bytes((seed.bit_length()+7)//8 or 1, "big")).digest()
    stride = (int.from_bytes(sbytes, "big") % (length - 1)) + 1
    # ensure co-prime to length
    if math.gcd(stride, length) != 1:
        stride += 1
        if stride >= length: stride = 1
    # derive offset similarly
    obytes = hashlib.sha256(b"off" + sbytes).digest()
    offset = int.from_bytes(obytes, "big") % length
    return stride, offset

def bins_to_windows(bins: List[int], weights: List[float], N: int,
                    start: int, end: int, bin_span: int, halo_frac: float) -> List[Tuple[int,int,float,int]]:
    """
    Map spectral bin indices to prioritized scalar windows.
    Returns list of (wstart, wend, weight, bin_index), sorted by weight desc.
    """
    L = end - start
    windows = []
    base_width = max(1, L // N)  # per-bin slice
    span_bins = max(1, int(bin_span))
    halo = int(base_width * halo_frac)

    for k, w in zip(bins, weights):
        # center slice around bin k (including neighbors within span_bins)
        lo_bin = (k - span_bins) % N
        hi_bin = (k + span_bins) % N
        # Convert to scalar indices (wrap-aware): approximate continuous mapping
        centers = []
        for kk in range(k - span_bins, k + span_bins + 1):
            frac = ((kk % N) + 0.5) / N
            centers.append(int(start + frac * L))
        if not centers:
            continue
        cmin, cmax = min(centers), max(centers)
        w0 = max(start, cmin - halo)
        w1 = min(end,   cmax + halo)
        if w1 > w0:
            windows.append((w0, w1, float(w), int(k)))
    # sort by weight desc, then narrower first
    windows.sort(key=lambda t: (-t[2], (t[1]-t[0])))
    # de-duplicate / merge overlaps preserving order
    merged: List[Tuple[int,int,float,int]] = []
    for ws,we,w,bk in windows:
        if not merged:
            merged.append((ws,we,w,bk)); continue
        ps,pe,pw,pb = merged[-1]
        if ws <= pe:
            merged[-1] = (ps, max(pe, we), max(pw,w), pb)  # merge to larger end; keep first bin id
        else:
            merged.append((ws,we,w,bk))
    return merged

# ===================== Hamming (optional guidance, low-cost) =====================
def hamming_distance(a: bytes, b: bytes) -> int:
    x = int.from_bytes(a, "big") ^ int.from_bytes(b, "big")
    return x.bit_count()

# ===================== Spectrum lock (build ECC phasor) =====================
def build_phasor(N: int) -> np.ndarray:
    p, a = K1_P, K1_A
    G = K1_G
    xs = []
    for k in range(N):
        if k == 0:
            xs.append(0)
        else:
            Q = ec_mul(G, k, p, a)
            xs.append(Q[0])
    xs_f = np.array(xs, dtype=np.float64) / float(p)
    u = np.exp(2j*np.pi * xs_f)
    u = highpass(u, method="diff")
    return u

def spectrum_lock(N: int, topk: int, print_spectrum: bool, base: Optional[str]):
    lam = 0.33333333326
    u = build_phasor(N)
    c_score, c_f, c_T, c_bins, c_prob = coarse(
        u, N, f_rng=(0.5, 5.0), T_rng=(16.0, 64.0),
        nf=120, nT=90, eps=0.30, f1=7, f2=11, beta=0.35,
        env_mode="sum", K=5, guard=1, bw=1, use_window=True
    )
    b_score, b_f, b_T, bins_found, prob_final = refine(
        u, N, c_f, c_T, df=0.25, dT=10.0, nf=140, nT=100,
        eps=0.30, f1=7, f2=11, beta=0.35,
        env_mode="sum", K=5, guard=1, bw=1, use_window=True
    )

    # weights normalized
    pf = prob_final.copy()
    pf[0] = 0.0
    s = pf.sum() + 1e-18
    order = np.argsort(pf)[::-1]
    top_bins = [int(k) for k in order[:topk]]
    top_wts  = [float(pf[k]/s) for k in top_bins]

    print("=== spectrum-lock (secp256k1 / DC-suppressed) ===")
    print(f"Coarse:  fτ={c_f:.6f}  Tc={c_T:.6f}  score={c_score:.6f}  bins={c_bins}")
    print(f"Refined: fτ={b_f:.6f}  Tc={b_T:.6f}  score={b_score:.6f}  bins={bins_found}")
    print(f"Top-K with weights:", list(zip(top_bins, top_wts)))

    if print_spectrum:
        spectrum_report(
            "Coarse",
            c_prob, c_bins, N, c_f, c_T, c_score, lam,
            print_ascii=True, topk=topk,
            save_png=(f"{base}-coarse.png" if base else None),
            save_json=(f"{base}-coarse.json" if base else None),
            save_csv=(f"{base}-coarse.csv" if base else None),
        )
        spectrum_report(
            "Refined",
            prob_final, bins_found, N, b_f, b_T, b_score, lam,
            print_ascii=True, topk=topk,
            save_png=(f"{base}-refined.png" if base else None),
            save_json=(f"{base}-refined.json" if base else None),
            save_csv=(f"{base}-refined.csv" if base else None),
        )

    return top_bins, top_wts, (b_f, b_T, b_score, bins_found, prob_final)

# ===================== Scanning / verification =====================
@dataclass
class Progress:
    last_ts: float
    last_tested: int
    ema_rate: float

def maybe_load_checkpoint(path: Optional[str]):
    if not path or not os.path.exists(path): return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def maybe_write_checkpoint(path: Optional[str], visited: int, tested: int, last_k: int, params: dict):
    if not path: return
    payload = {
        "visited": visited,
        "tested": tested,
        "last_k": last_k,
        "params": params,
        "ts_utc": datetime.utcnow().isoformat()
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)

def verify_k(k: int, compressed: bool, target_h160: bytes) -> bool:
    P = ec_mul(K1_G, k % K1_N, K1_P, K1_A)
    if P == O: 
        return False
    pub = _ser_pubkey(P, compressed=compressed)
    h160 = _hash160(pub)
    return h160 == target_h160

def scan_window(start: int, end: int, stride: int, offset: int,
                compressed: bool, target_h160: bytes,
                progress_sec: float, eta_alpha: float,
                stop_on_hit: bool,
                ckpt_file: Optional[str], ckpt_sec: float,
                worker_mod: int = 1, worker_id: int = 0) -> Optional[int]:
    """
    Iterates exactly (end-start) candidates in a strided, full-cycle walk.
    Splits by worker_mod/worker_id using k % worker_mod == worker_id.
    Returns the winning k if found, else None.
    """
    L = end - start
    if L <= 0: return None
    k = start + offset
    tested = 0
    visited = 0
    t0 = time.time()
    prog = Progress(last_ts=t0, last_tested=0, ema_rate=0.0)
    next_ckpt = t0 + ckpt_sec if ckpt_file and ckpt_sec > 0 else float("inf")
    next_log  = t0 + max(1.0, progress_sec)

    params = {
        "start": start, "end": end, "stride": stride, "offset": offset,
        "compressed": compressed, "worker_mod": worker_mod, "worker_id": worker_id
    }

    while visited < L:
        # worker filter
        if (k % worker_mod) == worker_id:
            visited += 1
            tested += 1
            if verify_k(k, compressed, target_h160):
                print(f"[hit] k=0x{k:064x}")
                maybe_write_checkpoint(ckpt_file, visited, tested, k, params)
                if stop_on_hit:
                    return k
        else:
            visited += 1

        k = start + ((k - start + stride) % L)

        now = time.time()
        if now >= next_log:
            dt = now - prog.last_ts
            inc = tested - prog.last_tested
            rate = inc / (dt + 1e-9)
            if prog.ema_rate == 0.0:
                prog.ema_rate = rate
            else:
                prog.ema_rate = eta_alpha * rate + (1.0 - eta_alpha) * prog.ema_rate
            remain = max(0, L - visited)
            eta_s = remain / max(1e-9, prog.ema_rate)
            print(f"[scan] tested={tested:,} visited={visited:,} rate={int(prog.ema_rate)}/s "
                  f"(ETA ~ {int(eta_s)}s) last_k=0x{k:064x}")
            prog.last_ts = now
            prog.last_tested = tested
            next_log = now + max(1.0, progress_sec)

        if now >= next_ckpt:
            maybe_write_checkpoint(ckpt_file, visited, tested, k, params)
            next_ckpt = now + ckpt_sec

    maybe_write_checkpoint(ckpt_file, visited, tested, k, params)
    return None

# ===================== CLI subcommands =====================
def cmd_spectrum_lock(args):
    bins, wts, _ = spectrum_lock(
        N=args.grid,
        topk=args.topk,
        print_spectrum=bool(args.print_spectrum),
        base=args.spectrum_base
    )

def cmd_solve135(args):
    # target
    what, target_h160, compressed = target_from_args(args)
    # range
    start, end = parse_range(args.range)
    L = end - start
    if L <= 0:
        raise ValueError("empty range")

    # spectrum lock first (to get guided bins)
    bins, wts, _ = spectrum_lock(
        N=args.grid, topk=args.topk,
        print_spectrum=bool(args.print_spectrum),
        base=(args.spectrum_base if args.spectrum_base else None)
    )

    # map bins → windows
    windows = bins_to_windows(bins, wts, args.grid, start, end,
                              bin_span=args.bin_span, halo_frac=args.bin_halo_frac)
    print("\n[guide] prioritized windows:")
    for i,(ws,we,w,bk) in enumerate(windows, 1):
        print(f"  {i:2d}. bin={bk:3d} weight={w:.12f}  "
              f"[{hex(ws)}, {hex(we)})  size={we-ws:,}")

    # stride/offset
    stride, offset = derive_stride_and_offset(args.seed or "01", L)
    print(f"\n[stride] length={L:,} stride={stride} offset={offset}")

    # scan windows in order
    params = {
        "start": start, "end": end, "grid": args.grid,
        "bins": bins, "wts": wts, "bin_span": args.bin_span, "halo": args.bin_halo_frac
    }
    visited_total = 0
    tested_total = 0
    hit_k = None
    t_all = time.time()
    ckpt_base = args.checkpoint_file or "jadon_dump_guided.json"

    for idx,(ws,we,w,bk) in enumerate(windows, 1):
        print(f"\n[window {idx}/{len(windows)}] bin={bk} weight={w:.12f} "
              f"range=[{hex(ws)}, {hex(we)}) size={we-ws:,}")
        ckpt_this = f"{os.path.splitext(ckpt_base)[0]}__bin{bk}_idx{idx}.json"
        k = scan_window(ws, we, stride, offset,
                        compressed=compressed, target_h160=target_h160,
                        progress_sec=args.progress_sec, eta_alpha=args.eta_alpha,
                        stop_on_hit=bool(args.stop_on_hit),
                        ckpt_file=ckpt_this, ckpt_sec=args.checkpoint_sec,
                        worker_mod=max(1, args.workers), worker_id=max(0, args.worker_id))
        # accumulate lite stats
        # (we could merge from checkpoint if desired)
        if k is not None:
            hit_k = k
            if args.stop_on_hit:
                break

    if hit_k is None:
        print("\n[result] no hit found in guided windows.")
    else:
        print(f"\n[result] FOUND: k=0x{hit_k:064x}")

def main():
    ap = argparse.ArgumentParser(prog="scalar_waze_guided.py")
    sp = ap.add_subparsers(dest="cmd", required=True)

    # spectrum-lock
    p_sl = sp.add_parser("spectrum-lock", help="Compute spectrum lock (coarse+refined) and print/report.")
    p_sl.add_argument("--grid", type=int, default=256, help="FFT grid size (e.g., 256).")
    p_sl.add_argument("--topk", type=int, default=10, help="How many bins to print/guide.")
    p_sl.add_argument("--print-spectrum", type=int, default=1, help="1 to print spectrum sections.")
    p_sl.add_argument("--spectrum-base", type=str, default=None, help="Base path to save PNG/JSON/CSV.")
    p_sl.set_defaults(func=cmd_spectrum_lock)

    # solve135 (guided)
    p_sv = sp.add_parser("solve135", help="Run guided scan using spectrum-lock top-K bins.")
    p_sv.add_argument("--address", type=str, default=None, help="P2PKH address (Base58).")
    p_sv.add_argument("--hash160", type=str, default=None, help="Target HASH160 (40-hex).")
    p_sv.add_argument("--hash-pub", type=str, choices=["compressed","uncompressed"], default="compressed")
    p_sv.add_argument("--range", type=str, required=True, help="start:end (hex or dec).")
    p_sv.add_argument("--grid", type=int, default=256)
    p_sv.add_argument("--topk", type=int, default=8)
    p_sv.add_argument("--bin-span", type=int, default=2, help="Neighbor bins on each side to include.")
    p_sv.add_argument("--bin-halo-frac", type=float, default=0.22, help="Scalar halo expansion per window.")
    p_sv.add_argument("--seed", type=str, default="01", help="Hex seed for stride/offset.")
    p_sv.add_argument("--workers", type=int, default=1)
    p_sv.add_argument("--worker-id", type=int, default=0)
    p_sv.add_argument("--progress-sec", type=float, default=5.0)
    p_sv.add_argument("--eta-alpha", type=float, default=0.45)
    p_sv.add_argument("--checkpoint-file", type=str, default="jadon_dump_guided.json")
    p_sv.add_argument("--checkpoint-sec", type=float, default=300.0)
    p_sv.add_argument("--stop-on-hit", type=int, default=1)
    p_sv.add_argument("--print-spectrum", type=int, default=1)
    p_sv.add_argument("--spectrum-base", type=str, default=None)
    p_sv.set_defaults(func=cmd_solve135)

    args = ap.parse_args()
    # sanity
    assert is_on_curve(K1_G, K1_P, K1_A, K1_B)
    args.func(args)

if __name__ == "__main__":
    main()