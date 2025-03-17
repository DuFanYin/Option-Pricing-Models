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

double binomialTree_v1(int steps, double S0, double K, double r, double T, double sigma) {
    double dt = T / steps;
    double u = exp(sigma * sqrt(dt));
    double d = 1 / u;
    double p = (exp(r * dt) - d) / (u - d);
    double discount = exp(-r * dt);

    std::vector<std::vector<double>> option(steps + 1, std::vector<double>(steps + 1, 0.0));

    for (int j = 0; j <= steps; ++j) {
        double stockPrice = S0 * pow(u, j) * pow(d, steps - j);
        option[steps][j] = std::max(K - stockPrice, 0.0);
        // option[steps][j] = std::max(stockPrice - K, 0.0);
    }

    for (int i = steps - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            double stockPrice = S0 * pow(u, j) * pow(d, i - j);
            double holdValue = discount * (p * option[i + 1][j + 1] + (1 - p) * option[i + 1][j]);
            double exerciseValue = std::max(K - stockPrice, 0.0);
            option[i][j] = std::max(holdValue, exerciseValue);
        }
    }

    return option[0][0];
}

double binomialTree_v2(int steps, double S0, double K, double r, double T, double sigma) {
    double dt = T / steps;
    double u = exp(sigma * sqrt(dt));
    double d = 1 / u;
    double p = (exp(r * dt) - d) / (u - d);
    double discount = exp(-r * dt);

    std::vector<double> layer1(steps + 1, 0.0);

    for (int j = 0; j <= steps; ++j) {
        double stockPrice = S0 * pow(u, j) * pow(d, steps - j);
        layer1[j] = std::max(K - stockPrice, 0.0);
    }

    int startIdx = 0, endIdx = steps;
    while (startIdx <= endIdx && layer1[startIdx] == 0.0) {
        ++startIdx;
    }
    while (endIdx >= startIdx && layer1[endIdx] == 0.0) {
        --endIdx;
    }

    layer1.resize(endIdx - startIdx + 1, 0.0);
    std::vector<double> layer2(layer1.size(), 0.0);

    double* prev = layer1.data();
    double* curr = layer2.data();

    for (int i = steps - 1; i >= 0; --i) {

        for (int j = startIdx; j <= endIdx; ++j) {
            double stockPrice = S0 * pow(u, j) * pow(d, i - j);
            double holdValue = discount * (p * prev[j + 1] + (1 - p) * prev[j]);
            double exerciseValue = std::max(K - stockPrice, 0.0);
            curr[j] = std::max(holdValue, exerciseValue);
        }

        std::swap(prev, curr);
    }

    return prev[0];
}

int main() {
    int steps = 1000; 
    double S0 = 100.0;
    double K = 100.0;
    double r = 0.05;
    double T = 1.0;
    double sigma = 0.2;

    auto start = std::chrono::high_resolution_clock::now();
    long memoryBefore = getMemoryUsage();
    double optionPrice = binomialTree_v1(steps, S0, K, r, T, sigma);
    long memoryAfter = getMemoryUsage();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    long memoryUsed = memoryAfter - memoryBefore;

    start = std::chrono::high_resolution_clock::now();
    memoryBefore = getMemoryUsage();
    double optionPrice_v1 = binomialTree_v2(steps, S0, K, r, T, sigma);
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
            << memoryUsed_v1 / 1024.0 << "\n";

    return 0;
}

// clang++ binomialTree.cpp -o bt && ./bt                 