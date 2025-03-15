# Option_Pricing

# Option Pricing Methods in C++

This is a C++ practice project implementing option pricing methods taught in SMU QF101 by Dr. Z, covering:

- Binomial Tree
- Trinomial Tree
- Monte Carlo Simulation
- Explicit Finite Difference
- Implicit Finite Difference

The algorithms are adapted from lecture slides. While option prices converge as time steps and simulation counts increase, excessively high parameters do not necessarily improve accuracy. Some techniques may also introduce more computational overhead and complexity than actual performance gains.

However, for learning purposes, this project explores fine-grained control over multithreading and memory management. The primary focus is on `C++ performance optimization`, `multithreading`, and `memory efficiency`.

## Change Log

The following optimization techniques were applied to each model (in sequence):

### 1. Binomial Tree
- **Rolling Storage:** Retain only two layers, achieving O(N) space complexity.
- **Pointer Swap:** Swap pointers instead of using vectors for better performance.
- **Pruning:** Eliminate leading and trailing zeros to avoid unnecessary calculations.

### 2. Trinomial Tree
- **Memory Optimization:** Applied Rolling Storage to optimize memory usage.

### 3. Monte Carlo Simulation
- **Multithreading:** Parallelized computations for StockPath, OptionPath, and backward induction, enabling batch processing. 50% speed increase compared to single-threaded execution.
- **Thread Pool:** Utilized smaller batches to prevent the overhead of deleting and recreating 8 threads. This change had a minor impact on performance.
- **Memory Optimization:** Stored only final and current step of the option price, reducing memory usage by 55% compared to the original approach.
- **Memory Optimization:** Malloc for manual mem allocation. Reduce container overhead, 10% speed boost.

### 4. Implicit Finite Difference
- 

### 5. Explicit Finite Difference
- 