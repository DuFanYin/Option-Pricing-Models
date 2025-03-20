#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// Function to perform matrix inversion using Gaussian elimination
std::vector<std::vector<double>> invertMatrix(const std::vector<std::vector<double>>& matrix) {
    int n = matrix.size();

    // Create augmented matrix [A | I]
    std::vector<std::vector<double>> augmented(n, std::vector<double>(2 * n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            augmented[i][j] = matrix[i][j];
        }
        augmented[i][n + i] = 1;  // Identity matrix
    }

    // Perform Gaussian elimination
    for (int i = 0; i < n; ++i) {
        // Find the pivot row and swap
        int maxRow = i;
        for (int k = i + 1; k < n; ++k) {
            if (std::abs(augmented[k][i]) > std::abs(augmented[maxRow][i])) {
                maxRow = k;
            }
        }
        std::swap(augmented[i], augmented[maxRow]);

        // Check if the matrix is singular
        if (augmented[i][i] == 0) {
            throw std::invalid_argument("Matrix is singular and cannot be inverted.");
        }

        // Normalize the pivot row
        double pivot = augmented[i][i];
        for (int j = 0; j < 2 * n; ++j) {
            augmented[i][j] /= pivot;
        }

        // Eliminate all other entries in column i
        for (int k = 0; k < n; ++k) {
            if (k != i) {
                double factor = augmented[k][i];
                for (int j = 0; j < 2 * n; ++j) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }

    // Extract the inverse matrix from the augmented matrix
    std::vector<std::vector<double>> inverse(n, std::vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            inverse[i][j] = augmented[i][n + j];
        }
    }

    return inverse;
}

// Matrix-vector multiplication
std::vector<double> multiplyMatrixVector(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vec) {
    int n = matrix.size();
    std::vector<double> result(n, 0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }

    return result;
}

std::vector<std::vector<double>> FDImplicit_Euro_Call(double S0, double K, double r, double sigma, double T, int M, int N, bool isRound = false, int dp = 4) {
    double dt = T / N;
    double Smax = 2 * K;
    double dS = Smax / M;

    // Initialize payoff matrix
    std::vector<std::vector<double>> f(M + 1, std::vector<double>(N + 1, 0));

    for (int j = 0; j <= M; ++j) {
        f[j][N] = std::max(j * dS - K, 0.0);
    }

    // Initialize matrix A
    std::vector<std::vector<double>> A(M + 1, std::vector<double>(M + 1, 0));
    A[0][0] = 1;
    A[M][M] = 1;
    for (int j = 1; j < M; ++j) {
        A[j][j - 1] = 0.5 * dt * (r * j - sigma * sigma * j * j);
        A[j][j] = 1 + dt * (sigma * sigma * j * j + r);
        A[j][j + 1] = -0.5 * dt * (r * j + sigma * sigma * j * j);
    }

    // Compute inverse of A using the invertMatrix function
    std::vector<std::vector<double>> Ainv = invertMatrix(A);

    // Backward in time iteration
    for (int i = N - 1; i >= 0; --i) {
        double bound = Smax - K * std::exp(-r * (N - i) * dt);
        std::vector<double> Fhat(M + 1, 0);
        Fhat[0] = 0;
        Fhat[M] = bound;

        // Update f for this time step by solving Ainv * Fhat
        std::vector<double> result = multiplyMatrixVector(Ainv, Fhat);
        for (int j = 0; j <= M; ++j) {
            f[j][i] = result[j];
        }
    }

    // Interpolate result for S0
    int k = std::floor(S0 / dS);
    double result = f[k][0] + (f[k + 1][0] - f[k][0]) / dS * (S0 - k * dS);
    return { { result } };
}

int main() {
    double S0 = 50, K = 50, r = 0.1, sigma = 0.4, T = 1;
    int M = 4, N = 3;
    auto result = FDImplicit_Euro_Call(S0, K, r, sigma, T, M, N, true);

    std::cout << "Option price: " << result[0][0] << std::endl;

    return 0;
}

// clang++ implicitFDM.cpp -o efdm && ./efdm