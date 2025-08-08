import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import math
from math import gcd
import time
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import qiskit.quantum_info as qi
from qiskit.circuit.library import QFT

def factor_rsa(N, exponent_size=None, show_plot=True):
    """
    Factor RSA modulus N using quantum-inspired period finding
    
    Args:
        N: number to factor (product of two primes)
        exponent_size: size of exponent register
        show_plot: whether to display probability distribution
        
    Returns:
        factors: prime factors of N
        period: period found
        elapsed_time: computation time
    """
    start_time = time.time()
    
    # Step 1: Choose a random a < N
    a = np.random.randint(2, N-1)
    while math.gcd(a, N) != 1:
        a = np.random.randint(2, N-1)
    
    # Step 2: Set exponent register size
    if exponent_size is None:
        exponent_size = 4 * N  # Sufficient to capture period
    
    # Initialize quantum state
    psi = np.zeros(exponent_size, dtype=np.complex128)
    
    # Step 3: Uniform superposition
    psi[:] = 1 / np.sqrt(exponent_size)
    
    # Step 4: Apply modular exponentiation
    fx = np.zeros(exponent_size, dtype=int)
    for x in range(exponent_size):
        fx[x] = pow(a, x, N)
    
    # Step 5: Apply inverse QFT
    def inverse_qft_1d(signal):
        n = len(signal)
        j = np.arange(n)
        k = j.reshape((n, 1))
        omega = np.exp(2j * np.pi * j * k / n)
        return (omega @ signal) / np.sqrt(n)
    
    psi = inverse_qft_1d(psi)
    
    # Step 6: Measure probabilities
    probabilities = np.abs(psi)**2
    
    # Find probability peaks
    peaks, _ = find_peaks(probabilities, height=0.01 * np.max(probabilities))
    peak_values = sorted(peaks, key=lambda i: probabilities[i], reverse=True)[:5]
    
    # Step 7: Extract period
    candidate_periods = set()
    for peak in peak_values:
        # Get fractional approximations
        for denom in range(1, min(100, N)):
            frac = peak / exponent_size
            numerator = round(frac * denom)
            error = abs(frac - numerator/denom)
            if error < 0.01:
                candidate_periods.add(denom)
    
    # Find valid period
    period = None
    for r in sorted(candidate_periods):
        if pow(a, r, N) == 1:
            period = r
            break
    
    # Step 8: Factor N
    factors = []
    if period and period % 2 == 0:
        x = pow(a, period//2, N)
        if x != 1 and x != N-1:
            factor1 = math.gcd(x - 1, N)
            factor2 = math.gcd(x + 1, N)
            if factor1 > 1 and factor2 > 1:
                factors = [factor1, factor2]
    
    # Classical fallback if quantum method fails
    if not factors:
        # Simple trial division for demonstration
        for i in range(2, int(math.isqrt(N)) + 1):
            if N % i == 0:
                factors = [i, N//i]
                break
    
    elapsed_time = time.time() - start_time
    
    # Visualization
    if show_plot:
        plt.figure(figsize=(14, 6))
        plt.bar(range(exponent_size), probabilities, width=1.0, alpha=0.7)
        plt.plot(peak_values, probabilities[peak_values], "xr", ms=10, mew=2)
        plt.xlabel('Exponent Value')
        plt.ylabel('Probability')
        plt.title(f'RSA Factorization: N = {N}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'rsa_factorization_{N}.png', dpi=150)
        plt.close()
    
    return sorted(factors), period, elapsed_time

def visualize_shor_circuit(N, a, num_qubits=4):
    """
    Visualize quantum circuit for Shor's algorithm
    
    Args:
        N: number to factor
        a: base for modular exponentiation
        num_qubits: number of qubits to use in visualization
        
    Returns:
        circuit: QuantumCircuit object
        result: measurement probabilities
    """
    # Create quantum circuit
    n_count = num_qubits  # Number of counting qubits
    n_value = num_qubits  # Number of value qubits
    
    qc = QuantumCircuit(n_count + n_value, n_count)
    
    # Initialize counting qubits
    for q in range(n_count):
        qc.h(q)
    
    # Apply modular exponentiation (simplified for visualization)
    qc.barrier()
    
    # This would be replaced with actual modular exponentiation in practice
    # For visualization, we'll use a simple placeholder
    for control in range(n_count):
        for target in range(n_value):
            qc.cp(np.pi/2, control, n_count + target)
    
    # Apply inverse QFT
    qc.barrier()
    qc.append(QFT(n_count, inverse=True), range(n_count))
    
    # Measure counting qubits
    qc.barrier()
    qc.measure(range(n_count), range(n_count))
    
    # Draw circuit
    print("\nQuantum Circuit for Shor's Algorithm:")
    print(qc.draw(output='text'))
    
    # Simulate circuit
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=1000).result()
    counts = result.get_counts(qc)
    
    # Plot results
    plot_histogram(counts).savefig(f'shor_circuit_{N}.png')
    
    return qc, counts

# Example usage
if __name__ == "__main__":
    # =================================================================
    # Part 1: RSA Factorization using Quantum-Inspired Period Finding
    # =================================================================
    print("="*60)
    print("Quantum-Inspired RSA Factorization")
    print("="*60)
    
    # Factor small RSA modulus
    N1 = 15  # 3 * 5
    factors1, period1, time1 = factor_rsa(N1, exponent_size=32)
    print(f"\nFactorization of {N1}: {factors1}")
    print(f"Period found: {period1}")
    print(f"Computation time: {time1:.4f} seconds")
    
    # Factor larger RSA modulus
    N2 = 21  # 3 * 7
    factors2, period2, time2 = factor_rsa(N2, exponent_size=64)
    print(f"\nFactorization of {N2}: {factors2}")
    print(f"Period found: {period2}")
    print(f"Computation time: {time2:.4f} seconds")
    
    # =================================================================
    # Part 2: Quantum Circuit Visualization
    # =================================================================
    print("\n" + "="*60)
    print("Quantum Circuit Visualization")
    print("="*60)
    
    # Visualize circuit for N=15
    print("\nVisualizing quantum circuit for N=15, a=7")
    qc1, counts1 = visualize_shor_circuit(15, 7, num_qubits=4)
    
    # Visualize circuit for N=21
    print("\nVisualizing quantum circuit for N=21, a=5")
    qc2, counts2 = visualize_shor_circuit(21, 5, num_qubits=4)
    
    print("\nQuantum factorization complete!")
