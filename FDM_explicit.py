import math
import numpy as np

def normal_round(n, ndigit=0):
    n = n * 10**ndigit
    if n - math.floor(n) < 0.499995:  # Q: Why isn't it 0.5? A: Machine error
        return math.floor(n) / (10**ndigit)
    return math.ceil(n) / (10**ndigit)

def calculate(x, isRound=False, dp=4):
    return normal_round(x, dp) if isRound else x

def FDExplicit_Euro_Call(S0, K, r, sigma, T, M, N, isRound=False, dp=4):
    dt = calculate(T / N, isRound, dp)
    print('dt:', dt)
    Smax = calculate(2 * K, isRound, dp)
    print('Smax:', Smax)
    dS = calculate(Smax / M, isRound, dp)
    print('dS:', dS)
    print()

    f = np.zeros((N + 1, M + 1))  # Use NumPy array for easier printing

    for j in range(M + 1):
        f[N, j] = calculate(max(j * dS - K, 0), isRound, dp)
        print(f'f({N},{j}):', f[N, j])

    for i in range(N - 1, -1, -1):
        print()
        f[i, 0] = 0
        f[i, M] = calculate(Smax - K * math.exp(-r * (N - i) * dt), isRound, dp)

        for j in range(1, M):
            aj = 0.5 * dt * (sigma**2 * j**2 - r * j)
            bj = (1 - (sigma**2 * j**2 + r) * dt)
            cj = 0.5 * dt * (sigma**2 * j**2 + r * j)
            f[i, j] = calculate(
                aj * f[i + 1, j - 1]
                + bj * f[i + 1, j]
                + cj * f[i + 1, j + 1],
                isRound,
                dp,
            )
         

        # Print the entire matrix at each step
        print("Matrix f at step i = ", i)
        ft = f.T
        print(np.array2string(ft, precision=dp, suppress_small=True))

    k = math.floor(S0 / dS)
    print()
    print('k:', k)
    return calculate(
        f[0, k] + (f[0, k + 1] - f[0, k]) / dS * (S0 - k * dS), isRound, dp
    )

# Example Parameters
S0, K, r, sigma, T, M, N = 50, 50, 0.1, 0.4, 1, 4, 3
print(f'V:', FDExplicit_Euro_Call(S0, K, r, sigma, T, M, N, isRound=True))