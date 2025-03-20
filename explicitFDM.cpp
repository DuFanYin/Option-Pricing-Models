#include <iostream>
#include <cmath>
#include <vector>

double FDExplicit_Euro_Call(double S0, double K, double r, double sigma, double T, int M, int N) {
    double dt = T / N;
    double Smax = 2 * K;
    double dS = Smax / M;

    std::vector<std::vector<double>> f(N + 1, std::vector<double>(M + 1, 0.0)); // Initialize a 2D array of zeros

    // Fill the final row (payoff at maturity)
    for (int j = 0; j <= M; ++j) {
        f[N][j] = std::max(j * dS - K, 0.0);
    }

    // Perform the backward induction
    for (int i = N - 1; i >= 0; --i) {
        f[i][0] = 0;
        f[i][M] = Smax - K * std::exp(-r * (N - i) * dt);

        for (int j = 1; j < M; ++j) {
            double aj = 0.5 * dt * (sigma * sigma * j * j - r * j);
            double bj = (1 - (sigma * sigma * j * j + r) * dt);
            double cj = 0.5 * dt * (sigma * sigma * j * j + r * j);
            f[i][j] = aj * f[i + 1][j - 1] + bj * f[i + 1][j] + cj * f[i + 1][j + 1];
        }
    }

    int k = std::floor(S0 / dS);
    return f[0][k] + (f[0][k + 1] - f[0][k]) / dS * (S0 - k * dS);
}

int main() {
    // Example Parameters
    double S0 = 50, K = 50, r = 0.1, sigma = 0.4, T = 1;
    int M = 4, N = 3;

    // Output the final result
    std::cout << "V: " << FDExplicit_Euro_Call(S0, K, r, sigma, T, M, N) << std::endl;

    return 0;
}

// clang++ explicitFDM.cpp -o efdm && ./efdm