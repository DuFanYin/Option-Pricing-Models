#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>
#include <sys/resource.h>  // For memory usage (Linux/Unix)

long getMemoryUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;  // in kilobytes (on Linux/Unix)
}

double trinomialTree(int steps, double S0, double K, double r, double T, double sigma) {
    double dt = T / steps;
    double u = exp(sigma * sqrt(dt));
    double d = 1 / u;
    double m = 1;  
    double p_u = (exp(r * dt) - d) / (u - d);
    double p_d = (u - exp(r * dt)) / (u - d);
    double discount = exp(-r * dt);

    std::vector<std::vector<std::vector<double>>> option(steps + 1, std::vector<std::vector<double>>(steps + 1, std::vector<double>(steps + 1, 0.0)));

    for (int j = 0; j <= steps; ++j) {
        for (int k = 0; k <= steps; ++k) {
            double stockPrice = S0 * pow(u, j) * pow(d, k);
            option[steps][j][k] = std::max(K - stockPrice, 0.0);
            // option[steps][j][k] = std::max(stockPrice - K, 0.0);
        }
    }

    for (int i = steps - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            for (int k = 0; k <= i - j; ++k) {
                double stockPrice = S0 * pow(u, j) * pow(d, k) * pow(m, i - j - k);
                double holdValue = discount * (p_u * option[i + 1][j + 1][k] + p_d * option[i + 1][j][k + 1] + (1 - p_u - p_d) * option[i + 1][j][k]);
                double exerciseValue = std::max(K - stockPrice, 0.0);
                option[i][j][k] = std::max(holdValue, exerciseValue);
            }
        }
    }

    return option[0][0][0];
}

double trinomialTree_v1(int steps, double S0, double K, double r, double T, double sigma) {
    double dt = T / steps;
    double u = exp(sigma * sqrt(dt));
    double d = 1 / u;
    double m = 1;  
    double p_u = (exp(r * dt) - d) / (u - d);
    double p_d = (u - exp(r * dt)) / (u - d);
    double discount = exp(-r * dt);

    // Two layers for rolling storage
    std::vector<std::vector<double>> optionCurrent(steps + 1, std::vector<double>(steps + 1, 0.0));
    std::vector<std::vector<double>> optionNext(steps + 1, std::vector<double>(steps + 1, 0.0));

    // Initialize the terminal condition (at the last step)
    for (int j = 0; j <= steps; ++j) {
        for (int k = 0; k <= steps; ++k) {
            double stockPrice = S0 * pow(u, j) * pow(d, k);
            optionCurrent[j][k] = std::max(K - stockPrice, 0.0);
            // optionCurrent[j][k] = std::max(stockPrice - K, 0.0);  // for call option
        }
    }

    // Roll backwards from the second-to-last step to the first step
    for (int i = steps - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            for (int k = 0; k <= i - j; ++k) {
                double stockPrice = S0 * pow(u, j) * pow(d, k) * pow(m, i - j - k);
                double holdValue = discount * (p_u * optionCurrent[j + 1][k] + p_d * optionCurrent[j][k + 1] + (1 - p_u - p_d) * optionCurrent[j][k]);
                double exerciseValue = std::max(K - stockPrice, 0.0);
                optionNext[j][k] = std::max(holdValue, exerciseValue);
            }
        }
        
        // Swap the current and next layers
        std::swap(optionCurrent, optionNext);
    }

    return optionCurrent[0][0];
}

int main() {
    int steps = 100;
    double S0 = 100.0;
    double K = 100.0;
    double r = 0.05;
    double T = 1.0;
    double sigma = 0.2;

    auto start = std::chrono::high_resolution_clock::now();
    long memoryBefore = getMemoryUsage();
    double optionPrice = trinomialTree(steps, S0, K, r, T, sigma);
    long memoryAfter = getMemoryUsage();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    long memoryUsed = memoryAfter - memoryBefore;

    start = std::chrono::high_resolution_clock::now();
    memoryBefore = getMemoryUsage();
    double optionPrice_v1 = trinomialTree_v1(steps, S0, K, r, T, sigma);
    memoryAfter = getMemoryUsage();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_v1 = end - start;
    long memoryUsed_v1 = memoryAfter - memoryBefore;

    std::cout << "Option Price (v1): " << optionPrice << "\n";
    std::cout << "Option Price (v2): " << optionPrice_v1 << "\n";
    std::cout << "Execution Time (ms): " 
            << duration.count() * 1e3 << " | " 
            << duration_v1.count() * 1e3 << "\n";
    std::cout << "Memory Usage (MB): " 
            << memoryUsed / 1024.0 << " | " 
            << memoryUsed_v1 << "\n";

    return 0;
}

// clang++ trinomialTree.cpp -o tt && ./tt
