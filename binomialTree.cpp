#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

double binomialTree(int steps, double S0, double K, double r, double T, double sigma) {
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

// rolling storage
double binomialTree_v1(int steps, double S0, double K, double r, double T, double sigma) {
    double dt = T / steps;
    double u = exp(sigma * sqrt(dt));
    double d = 1 / u;
    double p = (exp(r * dt) - d) / (u - d);
    double discount = exp(-r * dt);

    std::vector<double> current(steps + 1, 0.0);
    std::vector<double> next(steps + 1, 0.0);

    for (int j = 0; j <= steps; ++j) {
        double stockPrice = S0 * pow(u, j) * pow(d, steps - j);
        next[j] = std::max(K - stockPrice, 0.0);
        // next[steps][j] = std::max(stockPrice - K, 0.0);
    }

    for (int i = steps - 1; i >= 0; --i) {
        std::swap(current, next);
        for (int j = 0; j <= i; ++j) {
            double stockPrice = S0 * pow(u, j) * pow(d, i - j);
            double holdValue = discount * (p * current[j + 1] + (1 - p) * current[j]);
            double exerciseValue = std::max(K - stockPrice, 0.0);
            next[j] = std::max(holdValue, exerciseValue);
        }
    }

    return next[0];
}

int main() {
    int steps = 100;
    double S0 = 100.0;
    double K = 100.0;
    double r = 0.05;
    double T = 1.0;
    double sigma = 0.2;

    double optionPrice = binomialTree(steps, S0, K, r, T, sigma);
    std::cout << "Option Price (Binomial Tree): " << optionPrice << std::endl;

    double optionPrice_v1 = binomialTree_v1(steps, S0, K, r, T, sigma);
    std::cout << "Option Price (Binomial Tree): " << optionPrice << std::endl;

    return 0;
}

// clang++ binomialTree.cpp -o bt && ./bt                 