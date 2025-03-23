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

Optimization techniques used.

### 1. Binomial Tree
Rolling storage, pruning

### 2. Trinomial Tree
Rolling Storage

### 3. Monte Carlo Simulation

Multithreading, Batching, Thread Pool, Memory Optimization, Manual Allocation, Raw Pointers

- **Combined Results:** For 20000 paths and 2000 paths, reduced memory usage by 50%, excution time by 85%.

### 4. Implicit Finite Difference
- 

### 5. Explicit Finite Difference
- 