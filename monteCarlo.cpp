#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>

double monteCarlo(int steps, double S0, double K, double r, double T, double sigma, int numSimulations) {
    double dt = T / steps;
    double totalPayoff = 0.0;
    double discountFactor = exp(-r * T);

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (int sim = 0; sim < numSimulations; ++sim) {
        double stockPrice = S0;
        double optionValue = 0.0;
        
        for (int i = 0; i < steps; ++i) {
            double randomFactor = distribution(generator);
            stockPrice *= exp((r - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * randomFactor);
            optionValue = std::max(K - stockPrice, optionValue);
            // optionValue = std::max(stockPrice - K, optionValue);
        }

        totalPayoff += optionValue;
    }

    return discountFactor * (totalPayoff / numSimulations);
}

int main() {
    int steps = 100;
    double S0 = 100.0;
    double K = 100.0;
    double r = 0.05;
    double T = 1.0;
    double sigma = 0.2;
    int numSimulations = 10000;

    double optionPrice = monteCarlo(steps, S0, K, r, T, sigma, numSimulations);
    std::cout << "Option Price (Monte Carlo): " << optionPrice << std::endl;

    return 0;
}