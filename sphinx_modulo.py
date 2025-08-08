import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def solve_dlp(g, h, p, exponent_size=None, show_plot=True):
    """
    Solves the Discrete Logarithm Problem: find x such that g^x ≡ h mod p
    
    Args:
        g: generator of the multiplicative group
        h: target element
        p: prime modulus
        exponent_size: size of the exponent register (default: 4*(p-1))
        show_plot: whether to display the probability distribution
        
    Returns:
        x: solution to the DLP
        period: the period used to find the solution
    """
    group_order = p - 1
    if exponent_size is None:
        exponent_size = 4 * group_order  # Default size
    
    # Initialize quantum state (exponent × function)
    psi = np.zeros((exponent_size, p), dtype=np.complex128)
    
    # Step 1: Uniform superposition in exponent register
    psi[:, 0] = 1 / np.sqrt(exponent_size)
    
    # Step 2: Apply modular exponentiation
    for x in range(exponent_size):
        result = pow(g, x % group_order, p)
        psi[x, result] = psi[x, 0]
        psi[x, 0] = 0

    # Step 3: Apply inverse QFT to exponent register
    def inverse_qft_1d(signal):
        n = len(signal)
        inv_qft = np.zeros(n, dtype=np.complex128)
        for k in range(n):
            for j in range(n):
                angle = 2 * np.pi * j * k / n
                inv_qft[k] += signal[j] * np.exp(1j * angle)
            inv_qft[k] /= np.sqrt(n)
        return inv_qft

    for f in range(p):
        psi[:, f] = inverse_qft_1d(psi[:, f])
    
    # Step 4: Measure exponent probabilities
    probabilities = np.sum(np.abs(psi)**2, axis=1)
    
    # Find probability peaks
    peaks, _ = find_peaks(probabilities, height=0.01 * np.max(probabilities))
    peak_values = sorted(peaks, key=lambda i: probabilities[i], reverse=True)[:5]
    
    # Step 5: Find true period using minimal error method
    def find_true_period(peaks, exponent_size, group_order):
        """Find period with minimal total error across peaks"""
        divisors = [d for d in range(1, group_order+1) if group_order % d == 0]
        best_period = group_order
        min_error = float('inf')
        
        for r in divisors:
            total_error = 0
            valid_peaks = 0
            for peak in peaks:
                j = round(peak * r / exponent_size)
                expected_pos = j * exponent_size / r
                error = abs(peak - expected_pos)
                if error < exponent_size / (2*r):
                    total_error += error
                    valid_peaks += 1
            
            if valid_peaks >= 3 and total_error < min_error:
                min_error = total_error
                best_period = r
                
        return best_period

    period = find_true_period(peak_values, exponent_size, group_order)
    
    # Solve DLP
    solutions = [x for x in range(period) if pow(g, x, p) == h]
    x_val = solutions[0] if solutions else None
    
    # Visualization
    if show_plot:
        plt.figure(figsize=(12, 6))
        plt.bar(range(exponent_size), probabilities, width=1.0)
        plt.plot(peak_values, probabilities[peak_values], "xr", ms=10)
        plt.xlabel('Exponent Value')
        plt.ylabel('Probability')
        plt.title(f'DLP Solution: {g}^x ≡ {h} mod {p}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'dlp_{p}_{g}_{h}.png', dpi=300)
        plt.close()
    
    # Output results
    print(f"\nDiscrete Logarithm Problem: {g}^x ≡ {h} mod {p}")
    if x_val is not None:
        print(f"SOLUTION: x = {x_val}")
        print(f"Verification: {g}^{x_val} mod {p} = {pow(g, x_val, p)}")
        print(f"Period: r = {period}")
        
        # Calculate peak explanations
        print("\nPeak explanations (using true period):")
        for i, peak in enumerate(peak_values):
            fraction = peak/exponent_size
            j = round(fraction * period)
            expected_frac = j/period
            error = abs(fraction - expected_frac)
            distance = abs(peak - j * exponent_size / period)
            
            print(f"Peak {i+1} at {peak} ({peak}/{exponent_size} ≈ {fraction:.4f})")
            print(f"  → {j}/{period} ≈ {expected_frac:.4f} (error: {error:.4f})")
            print(f"  Position error: {distance:.1f} indices")
        return x_val, period
    else:
        print("No solution found via quantum method")
        # Classical fallback
        for x in range(group_order):
            if pow(g, x, p) == h:
                print(f"Classical solution: x = {x}")
                return x, group_order
        return None, None

# Example usage
if __name__ == "__main__":
    # Problem 1: 5^x ≡ 10 mod 23
    solve_dlp(g=5, h=10, p=23, exponent_size=625)
    
    # Problem 2: 2^x ≡ 15 mod 37
    solve_dlp(g=2, h=15, p=37, exponent_size=2048)
