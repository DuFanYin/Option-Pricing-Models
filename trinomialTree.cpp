#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

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

    std::vector<std::vector<double>> current(steps + 1, std::vector<double>(steps + 1, 0.0));
    std::vector<std::vector<double>> next(steps + 1, std::vector<double>(steps + 1, 0.0));

    for (int j = 0; j <= steps; ++j) {
        for (int k = 0; k <= steps; ++k) {
            double stockPrice = S0 * pow(u, j) * pow(d, k);
            next[j][k] = std::max(K - stockPrice, 0.0);;
            // next[j][k] = std::max(stockPrice - K, 0.0);
        }
    }

    for (int i = steps - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            for (int k = 0; k <= i - j; ++k) {
                double stockPrice = S0 * pow(u, j) * pow(d, k) * pow(m, i - j - k);
                double holdValue = discount * (p_u * current[j + 1][k] + p_d * current[j][k + 1] + (1 - p_u - p_d) * current[j][k]);
                double exerciseValue = std::max(K - stockPrice, 0.0);
                next[j][k] = std::max(holdValue, exerciseValue);
            }
        }
    }

    return next[0][0]; 
}

int main() {
    int steps = 100;
    double S0 = 100.0;
    double K = 100.0;
    double r = 0.05;
    double T = 1.0;
    double sigma = 0.2;

    double optionPrice = trinomialTree(steps, S0, K, r, T, sigma);
    std::cout << "Option Price (Trinomial Tree): " << optionPrice << std::endl;

    double optionPrice_v1 = trinomialTree_v1(steps, S0, K, r, T, sigma);
    std::cout << "Option Price (Trinomial Tree): " << optionPrice << std::endl;

    return 0;
}

// clang++ trinomialTree.cpp -o tt && ./tt
