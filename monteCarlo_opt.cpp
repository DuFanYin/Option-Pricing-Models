#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <random>
#include <thread>
#include <chrono>

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

void generatePaths(std::vector<std::vector<double>>& stockPath, std::vector<std::vector<double>>& optionPath, int startRow, int endRow,
    double S0, double K, double r, double sigma, double dt, int steps) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int j = startRow; j < endRow; ++j) {  
        stockPath[j][0] = S0;  // First column (step 0) is the initial stock price
        optionPath[j][0] =  std::max( K - S0, 0.0);

        for (size_t i = 1; i < steps; ++i) {  // Start from step 1
            double Z = dist(gen);  
            double dS = (r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z;
            stockPath[j][i] = stockPath[j][i - 1] * std::exp(dS);  // Use the previous step price
            optionPath[j][i] = std::max(K - stockPath[j][i], 0.0);
    }
}
}


void processBackwardInduction(int start, int end, int i, const std::vector<std::vector<double>>& stockPath,
    std::vector<std::vector<double>>& optionPath, std::vector<int>& p, double r, double dt,
    std::vector<double>& x, std::vector<double>& y, std::mutex& xy_mutex) {

    std::vector<double> local_x, local_y; // Thread-local storage to reduce lock contention

    for (int j = start; j < end; ++j) {
        if (optionPath[j][i] > 0) {
            local_x.push_back(stockPath[j][i]);
            local_y.push_back(optionPath[j][p[j]] * std::exp(-r * (p[j] - i) * dt));
        }
    }

    // Merge local_x and local_y into shared x, y with a lock
    {
        std::lock_guard<std::mutex> lock(xy_mutex);
        x.insert(x.end(), local_x.begin(), local_x.end());
        y.insert(y.end(), local_y.begin(), local_y.end());
    }
}

// Monte Carlo simulation for American option pricing
double monteCarlo(int numberOfstockPath, int steps, double S0, double K, double r, double T, double sigma) {
    double dt = T / steps;

    std::vector<std::vector<double>> stockPath(numberOfstockPath, std::vector<double>(steps+1, 0));
    std::vector<std::vector<double>> optionPath(numberOfstockPath, std::vector<double>(steps+1, 0));
    std::vector<int> p(numberOfstockPath, steps); 

    int numThreads = 8;  
    int batchSize = numberOfstockPath / numThreads;  

    std::vector<std::thread> threads;

    // populate stock and option path
    for (int i = 0; i < numThreads; ++i) {
        int startRow = i * batchSize;
        int endRow = (i == numThreads - 1) ? numberOfstockPath : startRow + batchSize;  // Last thread gets remaining rows
        threads.emplace_back(generatePaths, std::ref(stockPath), std::ref(optionPath), startRow, endRow, S0, K, r, sigma, dt, steps);
    }

    // Join threads
    for (auto& t : threads) {
        t.join();
    }

    std::mutex xy_mutex;

    // backward induction
    for (int i = steps - 1; i > 0; --i) {
        std::vector<double> x, y;
        std::vector<std::thread> threads;
        int batch_size = numberOfstockPath / 8;

        for (int t = 0; t < 8; ++t) {
            int start = t * batch_size;
            int end = (t == 8 - 1) ? numberOfstockPath : start + batch_size;
            threads.emplace_back(processBackwardInduction, start, end, i, std::cref(stockPath), std::ref(optionPath),
                                 std::ref(p), r, dt, std::ref(x), std::ref(y), std::ref(xy_mutex));
        }
        
        for (auto& thread : threads) {
            thread.join();
        }

        if (x.empty() || y.empty() || x.size() != y.size()) continue;
        std::vector<double> c = ordinary_least_squares(x, y);
        if (c.size() < 3) continue;

        for (int j = 0; j < numberOfstockPath; ++j) {
            if (optionPath[j][i] > 0) {
                double fS = c[2] * stockPath[j][i] * stockPath[j][i] + c[1] * stockPath[j][i] + c[0];
                if (optionPath[j][i] > fS) {
                    p[j] = i;
                } else if (p[j] > i) {
                    optionPath[j][i] = 0;
                } else {
                    optionPath[j][i] = optionPath[j][i + 1] * std::exp(-r * dt);
                }
            }
        }
    }

    // Compute Monte Carlo estimate of option price
    double s = 0;
    for (int j = 0; j < numberOfstockPath; ++j) {
        double discount_factor = std::exp(-r * p[j] * dt);
        double payoff = optionPath[j][p[j]];  // Directly use p[j] as the exercise time index
        s += discount_factor * payoff;
    }

    // Return the estimated option price
    return s / numberOfstockPath;
}

int main() {
    double S0 = 50;
    double K = 50;
    double r = 0.1;
    double T = 1;
    double sigma = 0.4;

    int numberOfstockPath = 10000;
    int steps = 500;

    // Run Monte Carlo simulation
    auto start = std::chrono::high_resolution_clock::now();
    double optionPrice = monteCarlo(numberOfstockPath, steps, S0, K, r, T, sigma);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;


    // Output the estimated option price
    std::cout << "Estimated Option Price: " << optionPrice << std::endl;
    std::cout << "Execution Time (ms): " << duration.count() * 1e6 << std::endl;

    return 0;
}


// clang++ mc_opt.cpp -o mc && ./mc