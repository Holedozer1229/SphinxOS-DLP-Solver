# This Code is for educational purpose only!!

# SphinxOS Cryptanalysis Suite

Quantum-inspired solutions for cryptographic vulnerabilities:

- **Modular Discrete Logarithm Problem**
- **Elliptic Curve Discrete Logarithm Problem**
- **RSA Factorization with Quantum Circuit Visualization**

## Installation

```bash
git clone https://github.com/Holedozer1229/SphinxOS-DLP-Solver.git
cd SphinxOS-DLP-Solver
pip install -r requirements.txt



# SphinxOS-DLP-Solver
SphinxOS Elliptical Curve Discrete Logarithm Problem Solver

# Results from sphinx_dlp.py

python3
============================================================
Quantum-Inspired ECDLP Solver
============================================================

============================================================
Elliptic Curve Discrete Logarithm Problem
============================================================
Curve: y² = x³ + 1x + 1 mod 5
Base point P: (0, 1)
Target point Q: (4, 2)
Order of P: n = 5
Exponent register size: 16
Computation time: 0.9764 seconds

SOLUTION: k = 2
Verification: 2*P = (4, 2)
Period used: n = 5

Peak explanations:
Peak   Position   Fraction     j/n                  Frac Error   Pos Error   
----------------------------------------------------------------------
1      13         0.812500      4/5 ≈ 0.800000    0.012500     0.200000
2      3          0.187500      1/5 ≈ 0.200000    0.012500     0.200000
3      10         0.625000      3/5 ≈ 0.600000    0.025000     0.400000
4      6          0.375000      2/5 ≈ 0.400000    0.025000     0.400000

============================================================
Elliptic Curve Discrete Logarithm Problem
============================================================
Curve: y² = x³ + 2x + 3 mod 97
Base point P: (3, 6)
Target point Q: (80, 10)
Order of P: n = 97
Exponent register size: 128
Computation time: 0.2849 seconds

SOLUTION: k = 2
Verification: 2*P = (80, 10)
Period used: n = 97

Peak explanations:
Peak   Position   Fraction     j/n                  Frac Error   Pos Error   
----------------------------------------------------------------------
1      33         0.257812      25/97 ≈ 0.257732    0.000081     0.010309
2      95         0.742188      72/97 ≈ 0.742268    0.000081     0.010309
3      66         0.515625      50/97 ≈ 0.515464    0.000161     0.020619
4      62         0.484375      47/97 ≈ 0.484536    0.000161     0.020619
5      99         0.773438      75/97 ≈ 0.773196    0.000242     0.030928

============================================================
Elliptic Curve Discrete Logarithm Problem
============================================================
Curve: y² = x³ + 0x + 7 mod 101
Base point P: (4, 24)
Target point Q: (77, 90)
Order of P: n = 101
Exponent register size: 100
Computation time: 0.2766 seconds

SOLUTION: k = 5
Verification: 5*P = (77, 90)
Period used: n = 101

Peak explanations:
Peak   Position   Fraction     j/n                  Frac Error   Pos Error   
----------------------------------------------------------------------
1      5          0.050000      5/101 ≈ 0.049505    0.000495     0.049505
2      7          0.070000      7/101 ≈ 0.069307    0.000693     0.069307
3      12         0.120000      12/101 ≈ 0.118812    0.001188     0.118812
4      17         0.170000      17/101 ≈ 0.168317    0.001683     0.168317
5      27         0.270000      27/101 ≈ 0.267327    0.002673     0.267327
