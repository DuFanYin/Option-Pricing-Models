#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <random>
#include <chrono>
#include <sys/resource.h> 

long getMemoryUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;  // in kilobytes (on Linux/Unix)
}

// Function to solve a linear system of equations using Gaussian elimination
std::vector<double> solve_linear_system(std::vector<std::vector<double>> A, std::vector<double> B) {
    int n = 3; // 3x3 system
    for (int i = 0; i < n; ++i) {
        // Pivoting (for numerical stability)
        int pivot = i;
        for (int j = i + 1; j < n; ++j) {
            if (abs(A[j][i]) > abs(A[pivot][i])) {
                pivot = j;
            }
        }
        std::swap(A[i], A[pivot]);
        std::swap(B[i], B[pivot]);

        // Make leading coefficient 1
        double diag = A[i][i];
        if (diag == 0) throw std::runtime_error("Singular matrix, cannot solve.");
        for (int j = 0; j < n; ++j) A[i][j] /= diag;
        B[i] /= diag;

        // Eliminate below
        for (int j = i + 1; j < n; ++j) {
            double factor = A[j][i];
            for (int k = 0; k < n; ++k) A[j][k] -= factor * A[i][k];
            B[j] -= factor * B[i];
        }
    }

    // Back substitution
    std::vector<double> X(n);
    for (int i = n - 1; i >= 0; --i) {
        X[i] = B[i];
        for (int j = i + 1; j < n; ++j) {
            X[i] -= A[i][j] * X[j];
        }
    }
    return X;
}

// Ordinary Least Squares (OLS) to fit a quadratic function
std::vector<double> ordinary_least_squares(const std::vector<double>& X, const std::vector<double>& Y) {
    if (X.size() != Y.size() || X.empty()) throw std::invalid_argument("Vectors must be non-empty and same size.");

    int n = X.size();
    double sumX = 0, sumX2 = 0, sumX3 = 0, sumX4 = 0;
    double sumY = 0, sumXY = 0, sumX2Y = 0;

    for (int i = 0; i < n; ++i) {
        double x = X[i], y = Y[i];
        double x2 = x * x, x3 = x2 * x, x4 = x3 * x;

        sumX += x;
        sumX2 += x2;
        sumX3 += x3;
        sumX4 += x4;

        sumY += y;
        sumXY += x * y;
        sumX2Y += x2 * y;
    }

    // Construct the system Ax = B
    std::vector<std::vector<double>> A = {
        {static_cast<double>(n), sumX, sumX2},
        {sumX, sumX2, sumX3},
        {sumX2, sumX3, sumX4}
    };

    std::vector<double> B = {sumY, sumXY, sumX2Y};

    return solve_linear_system(A, B); // Returns {c, b, a}
}


double monteCarlo(int numberOfStockPath, int steps, double S0, double K, double r, double T, double sigma) {
    double dt = T / steps;

    // Dynamically allocate memory using malloc for stockPath
    double** stockPath = (double**)malloc(numberOfStockPath * sizeof(double*));
    for (int i = 0; i < numberOfStockPath; ++i) {
        stockPath[i] = (double*)malloc((steps + 1) * sizeof(double));  // Each row is a vector of doubles
    }

    // Dynamically allocate memory for optionPath and p
    double* optionPath = (double*)malloc(numberOfStockPath * sizeof(double));
    int* p = (int*)malloc(numberOfStockPath * sizeof(int));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    // Simulate stock paths
    for (int j = 0; j < numberOfStockPath; ++j) {  
        stockPath[j][0] = S0; 

        for (int i = 1; i <= steps; ++i) {  
            double Z = dist(gen);  
            double dS = (r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z;
            stockPath[j][i] = stockPath[j][i - 1] * std::exp(dS);  
        }
    }

    // Calculate option path values
    for (int j = 0; j < numberOfStockPath; ++j) { 
        optionPath[j] = std::max(K - stockPath[j][steps], 0.0);
    }

    // Backward induction for option pricing
    for (int i = steps - 1; i > 0; --i) { 
        std::vector<double> x, y;
        double* optionPriceAtStep = (double*)malloc(numberOfStockPath * sizeof(double)); // Dynamically allocate memory for option prices

        for (int j = 0; j < numberOfStockPath; ++j) { 
            optionPriceAtStep[j] = std::max(K - stockPath[j][i], 0.0);
            if (optionPriceAtStep[j] > 0) {
                x.push_back(stockPath[j][i]);
                y.push_back(optionPath[j] * std::exp(-r * (p[j] - i) * dt));
            }
        }

        if (x.empty() || y.empty() || x.size() != y.size()) {
            free(optionPriceAtStep); // Cleanup memory for optionPriceAtStep
            continue; 
        }

        // Perform ordinary least squares regression
        std::vector<double> c = ordinary_least_squares(x, y);
        if (c.size() < 3) {
            free(optionPriceAtStep); // Cleanup memory for optionPriceAtStep
            continue;
        }

        for (int j = 0; j < numberOfStockPath; ++j) {
            if (optionPriceAtStep[j] > 0) {
                double fS = c[2] * stockPath[j][i] * stockPath[j][i] + c[1] * stockPath[j][i] + c[0];
                if (optionPriceAtStep[j] > fS) {
                    p[j] = i;  
                    optionPath[j] = optionPriceAtStep[j];
                }
            }
        }

        free(optionPriceAtStep); // Cleanup memory for optionPriceAtStep
    }

    // Calculate the final option price
    double s = 0;
    for (int j = 0; j < numberOfStockPath; ++j) {
        double discount_factor = std::exp(-r * p[j] * dt);
        double payoff = optionPath[j];  
        s += discount_factor * payoff;
    }

    // Free dynamically allocated memory
    for (int i = 0; i < numberOfStockPath; ++i) {
        free(stockPath[i]);  // Free each row
    }
    free(stockPath);  // Free the main array
    free(optionPath);  // Free the optionPath array
    free(p);  // Free the p array

    return s / numberOfStockPath;
}

int main() {
    double S0 = 50;
    double K = 50;
    double r = 0.1;
    double T = 1;
    double sigma = 0.4;

    int numberOfStockPath = 10000;
    int steps = 500;

    auto start = std::chrono::high_resolution_clock::now();
    long memoryBefore = getMemoryUsage();  // Measure memory usage before function call

    double optionPrice = monteCarlo(numberOfStockPath, steps, S0, K, r, T, sigma);

    long memoryAfter = getMemoryUsage();  // Measure memory usage after function call
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    long memoryUsed = memoryAfter - memoryBefore;  // Memory used by monteCarlo

    // Output the estimated option price
    std::cout << "Estimated Option Price: " << optionPrice << std::endl;
    std::cout << "Execution Time (ms): " << duration.count() * 1e6 << std::endl;
    std::cout << "Memory Usage (MB): " << memoryUsed / 1024 << "\n";

    return 0;
}

// clang++ monteCarlo_mem.cpp -o mc && ./mc