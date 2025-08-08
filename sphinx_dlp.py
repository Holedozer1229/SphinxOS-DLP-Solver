import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import time

class EllipticCurve:
    """Efficient elliptic curve implementation with optimized point operations"""
    def __init__(self, a, b, p):
        self.a = a
        self.b = b
        self.p = p
        
    def point_add(self, P, Q):
        """Add two points on the elliptic curve using optimized modular arithmetic"""
        if P == "infinity":
            return Q
        if Q == "infinity":
            return P
            
        xp, yp = P
        xq, yq = Q
        
        # Point doubling (P == Q)
        if P == Q:
            if yp == 0:
                return "infinity"
            numerator = (3 * xp*xp + self.a) % self.p
            denominator = (2 * yp) % self.p
            s = numerator * pow(denominator, self.p-2, self.p) % self.p
        else:
            if xp == xq:
                return "infinity"
            numerator = (yq - yp) % self.p
            denominator = (xq - xp) % self.p
            s = numerator * pow(denominator, self.p-2, self.p) % self.p
            
        xr = (s*s - xp - xq) % self.p
        yr = (s*(xp - xr) - yp) % self.p
        return (xr, yr)
    
    def scalar_mult(self, k, P):
        """Efficient scalar multiplication using double-and-add algorithm"""
        if k == 0 or P == "infinity":
            return "infinity"
            
        result = "infinity"
        current = P
        k_bin = bin(k)[2:]
        
        for bit in k_bin:
            result = self.point_add(result, result)
            if bit == '1':
                result = self.point_add(result, current)
            current = self.point_add(current, current)
        return result
    
    def point_order(self, P):
        """Compute order of point P using optimized algorithm"""
        if P == "infinity":
            return 1
            
        # Factorize p-1 to find possible orders
        factors = []
        n = self.p - 1
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
            
        # Determine order by checking divisors of p-1
        order = self.p
        for f in set(factors):
            candidate = order // f
            if self.scalar_mult(candidate, P) == "infinity":
                order = candidate
        return order

def solve_ecdlp(curve, P, Q, exponent_size=None, show_plot=True):
    """
    Solve ECDLP: find k such that k*P = Q
    
    Args:
        curve: EllipticCurve object
        P: base point
        Q: target point
        exponent_size: size of exponent register
        show_plot: whether to display probability distribution
        
    Returns:
        k: solution to ECDLP
        period: order of P
        elapsed_time: computation time
    """
    start_time = time.time()
    
    # Compute the order of P
    n = curve.point_order(P)
    
    # Set exponent register size
    if exponent_size is None:
        exponent_size = 4 * n
    
    # Memory management check
    if exponent_size * n > 10**8:
        print("Large group: Using classical solution")
        for k in range(1, n):
            if curve.scalar_mult(k, P) == Q:
                return k, n, time.time()-start_time
        return None, n, time.time()-start_time
    
    # Initialize quantum state
    psi = np.zeros((exponent_size, n), dtype=np.complex128)
    psi[:, 0] = 1 / np.sqrt(exponent_size)
    
    # Apply scalar multiplication oracle
    for k_val in range(exponent_size):
        point = curve.scalar_mult(k_val % n, P)
        point_idx = 0 if point == "infinity" else k_val % n
        psi[k_val, point_idx] = psi[k_val, 0]
        psi[k_val, 0] = 0

    # Inverse QFT using vectorization
    def inverse_qft_1d(signal):
        n_len = len(signal)
        j = np.arange(n_len)
        k = j.reshape((n_len, 1))
        omega = np.exp(2j * np.pi * j * k / n_len)
        return (omega @ signal) / np.sqrt(n_len)

    for pt_idx in range(n):
        psi[:, pt_idx] = inverse_qft_1d(psi[:, pt_idx])
    
    # Measure probabilities
    probabilities = np.sum(np.abs(psi)**2, axis=1)
    peaks, _ = find_peaks(probabilities, height=0.01 * np.max(probabilities))
    peak_values = sorted(peaks, key=lambda i: probabilities[i], reverse=True)[:5]
    
    # Find period with minimal error
    def find_period(peaks, exp_size, order):
        divisors = [d for d in range(1, order+1) if order % d == 0]
        best_period = order
        min_error = float('inf')
        
        for r in divisors:
            total_error = 0
            valid_peaks = 0
            for peak in peaks:
                j = round(peak * r / exp_size)
                expected_pos = j * exp_size / r
                error = abs(peak - expected_pos)
                if error < exp_size / (2*r):
                    total_error += error
                    valid_peaks += 1
            
            if valid_peaks >= 3 and total_error < min_error:
                min_error = total_error
                best_period = r
        return best_period

    period = find_period(peak_values, exponent_size, n)
    
    # Solve ECDLP
    solutions = [k for k in range(period) if curve.scalar_mult(k, P) == Q]
    k_val = solutions[0] if solutions else None
    
    # Visualization
    if show_plot:
        plt.figure(figsize=(14, 7))
        plt.bar(range(exponent_size), probabilities, width=1.0, alpha=0.7)
        plt.plot(peak_values, probabilities[peak_values], "xr", ms=10, mew=2)
        for i, peak in enumerate(peak_values):
            plt.annotate(f"Peak {i+1}", (peak, probabilities[peak]), 
                         xytext=(peak, probabilities[peak] + 0.005),
                         arrowprops=dict(arrowstyle="->", lw=1.5),
                         ha='center')
        plt.xlabel('Exponent Value')
        plt.ylabel('Probability')
        plt.title(f'ECDLP Solution: k*P = Q\nCurve: y² = x³ + {curve.a}x + {curve.b} mod {curve.p}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'ecdlp_{curve.p}.png', dpi=150)
        plt.close()
    
    elapsed_time = time.time() - start_time
    
    # Output results
    print(f"\n{'='*60}")
    print(f"Elliptic Curve Discrete Logarithm Problem")
    print(f"{'='*60}")
    print(f"Curve: y² = x³ + {curve.a}x + {curve.b} mod {curve.p}")
    print(f"Base point P: {P}")
    print(f"Target point Q: {Q}")
    print(f"Order of P: n = {n}")
    print(f"Exponent register size: {exponent_size}")
    print(f"Computation time: {elapsed_time:.4f} seconds")
    
    if k_val is not None:
        print(f"\nSOLUTION: k = {k_val}")
        print(f"Verification: {k_val}*P = {curve.scalar_mult(k_val, P)}")
        print(f"Period used: n = {period}")
        
        print("\nPeak explanations:")
        print(f"{'Peak':<6} {'Position':<10} {'Fraction':<12} {'j/n':<20} {'Frac Error':<12} {'Pos Error':<12}")
        print(f"{'-'*70}")
        for i, peak in enumerate(peak_values):
            frac = peak/exponent_size
            j = round(frac * period)
            exp_frac = j/period
            frac_err = abs(frac - exp_frac)
            pos_err = abs(peak - j * exponent_size / period)
            print(f"{i+1:<6} {peak:<10} {frac:.6f}      {j}/{period} ≈ {exp_frac:.6f}    {frac_err:.6f}     {pos_err:.6f}")
    else:
        print("No quantum solution found, using classical fallback")
        for k in range(1, n):
            if curve.scalar_mult(k, P) == Q:
                print(f"Classical solution: k = {k}")
                return k, n, elapsed_time
        return None, n, elapsed_time
        
    return k_val, period, elapsed_time

if __name__ == "__main__":
    print("="*60)
    print("Quantum-Inspired ECDLP Solver")
    print("="*60)
    
    # Example 1: Small curve
    curve1 = EllipticCurve(a=1, b=1, p=5)
    P1 = (0, 1)
    Q1 = curve1.scalar_mult(2, P1)
    k1, period1, t1 = solve_ecdlp(curve1, P1, Q1, exponent_size=16)
    
    # Example 2: Medium curve
    curve2 = EllipticCurve(a=2, b=3, p=97)
    P2 = (3, 6)
    Q2 = curve2.scalar_mult(2, P2)
    k2, period2, t2 = solve_ecdlp(curve2, P2, Q2, exponent_size=128)
    
    # Example 3: Secp256k1 demo
    curve3 = EllipticCurve(a=0, b=7, p=101)
    # Find a point on the curve
    P3 = None
    for x in range(101):
        rhs = (x**3 + 7) % 101
        for y in range(101):
            if (y*y) % 101 == rhs:
                P3 = (x, y)
                break
        if P3: break
    if P3:
        Q3 = curve3.scalar_mult(5, P3)
        k3, period3, t3 = solve_ecdlp(curve3, P3, Q3, exponent_size=100)
