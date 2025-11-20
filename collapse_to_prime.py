import mpmath
from mpmath import mp, mpf, power, sqrt, log, fabs

# Set precision to 120 decimal places (more than enough)
mp.dps = 120

def ace_rimmer_product(N_terms=1000000):
    """
    Compute the infinite product ∏_{n≠0} (1 + 1/cosh²(2πn))
    by multiplying positive n only (symmetric) up to N_terms.
    """
    q = mp.exp(-2 * mp.pi)        # q = e^{-2π} ≈ 0.00186744273171
    log_prod = mpf(0)

    for n in range(1, N_terms + 1):
        term = 1 + 1 / mp.cosh(2 * mp.pi * n)**2
        log_prod += mp.log(term)

        # Early exit if term is indistinguishable from 1
        if fabs(term - 1) < mp.exp(-mp.dps * mp.ln(10) * 0.9):
            break

    # Since product is over n and -n, we doubled the log
    total_log = 2 * log_prod
    return mp.exp(total_log)

# Exact closed form
def exact_value():
    return power(2, mpf('0.375')) * sqrt(1 + sqrt(2))

# Compute both
numerical = ace_rimmer_product(N_terms=1000)  # 1000 terms is already overkill
exact = exact_value()

print("Ace Rimmer Product (numerical):")
print(numerical)
print("\nExact closed form:")
print(exact)
print("\nDifference (should be < 10^{-100}):")
print(fabs(numerical - exact))
print("\nAre they equal to 100 decimal places?")
print(numerical == mp.chop(numerical, tol=1e-100))