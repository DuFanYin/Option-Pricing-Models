from math import exp, floor
import numpy as np
from numpy.linalg import inv

def calculate(x, isRound=False, dp=4):
    return round(x, dp) if isRound else x

def FDImplicit_Euro_Call(S0, K, r, sigma, T, M, N, isRound=False, dp=4):
    dt = calculate(T / N, isRound, dp)
    print(f'dt: {dt}')
    
    Smax = calculate(2 * K, isRound, dp)
    print(f'Smax: {Smax}')
    
    dS = calculate(Smax / M, isRound, dp)
    print(f'dS: {dS}\n')
    
    # Initialize payoff matrix
    f = np.zeros((M + 1, N + 1))
    print("Initial Payoff (at T):")
    for j in range(M + 1):
        f[j, N] = calculate(max(j * dS - K, 0), isRound, dp)
        print(f"f[{j},{N}] = {f[j, N]}")
    print()
    
    # Set up matrix A
    A = np.zeros((M + 1, M + 1))
    A[0, 0] = 1
    A[M, M] = 1
    for j in range(1, M):
        A[j, j - 1] = calculate(0.5 * dt * (r * j - sigma**2 * j**2), isRound, dp)   # aj
        A[j, j]     = calculate(1 + dt * (sigma**2 * j**2 + r), isRound, dp)          # bj
        A[j, j + 1] = calculate(-0.5 * dt * (r * j + sigma**2 * j**2), isRound, dp)   # cj
    
    print("Matrix A:")
    print(np.array2string(A, formatter={'float_kind': lambda x: f"{x:>8.4f}"}))
    
    # Compute inverse of A
    Ainv = inv(A).round(dp)
    print("\nInverse of A:")
    print(np.array2string(Ainv, formatter={'float_kind': lambda x: f"{x:>8.4f}"}))
    print()
    print("-" * 50)

    # Backward in time iteration
    for i in range(N - 1, -1, -1):
        bound = round(Smax - K * exp(-r * (N - i) * dt),4)
        print(f"\ni = {i}, bound = {bound}")
        print()
        Fhat = f[:, [i + 1]]
        Fhat[0, 0] = 0
        Fhat[M, 0] = Smax - K * exp(-r * (N - i) * dt)
        print("Adjusted Fhat:")
        print(np.array2string(Fhat, formatter={'float_kind': lambda x: f"{x:>8.4f}"}))
        
        # Update f for this time step
        f[:, i] = (Ainv @ Fhat).flatten().round(dp)  # Flatten to get a 1D array for this column
        print(f"\nf")
        print(np.array2string(f, formatter={'float_kind': lambda x: f"{x:>8.4f}"}))
        print()
        print("-" * 50)
        
    # Interpolate result for S0
    k = floor(S0 / dS)
    print(f"\nk: {k}")
    result = calculate(f[k, 0] + (f[k + 1, 0] - f[k, 0]) / dS * (S0 - k * dS), isRound, dp)
    print(f"Final interpolated result: {result}\n")
    return result

# Example call
S0, K, r, sigma, T, M, N = 50, 50, 0.1, 0.4, 1, 4, 3
FDImplicit_Euro_Call(S0, K, r, sigma, T, M, N, isRound=True)