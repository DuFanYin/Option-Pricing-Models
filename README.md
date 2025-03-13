# Option_Pricing

This this a `c++` practice project implementing the option pricing methods taught in `SMU QF101` by Dr.Z, including:  

- **Binomial Tree**  
- **Trinomial Tree**  
- **Monte Carlo Simulation**  
- **Explicit Finite Difference**  
- **Implicit Finite Difference**  

Algorithems are adapeted from lecture slide;

Additionally, i will explore `multithreading` and `memory optimization` for better performance

Also learn some simple `Linear Algebra`.

## Comparison of Option Pricing Methods

| **Method**                    | **Time Complexity** | **Multithreading Feasibility** | **Memory Pool Optimization Feasibility** | **Computational Cost** | **Suitable Synchronization Method** |
|-------------------------------|--------------------|--------------------------------|----------------------------------------|-----------------------|-------------------------------------|
| **Binomial Tree**              | \(O(N^2)\)         | 🔴 Limited                     | ✅ Yes (rolling storage)               | 🟠 Medium              | Mutex/Lock for thread-safe updates  |
| **Trinomial Tree**             | \(O(N^2)\)         | 🔴 Limited                     | ✅ Yes (rolling storage)               | 🟠 High                | Mutex/Lock for thread-safe updates  |
| **Monte Carlo Simulation**     | \(O(M)\)           | 🟢 Highly Parallelizable       | ✅ Yes (preallocated paths)            | 🔴 Heavy               | Thread Pool / Atomic Operations     |
| **Explicit Finite Difference** | \(O(NM)\)          | 🟢 Parallelizable              | ✅ Yes (sparse grid)                   | 🟠 Moderate-High       | Lock-free or thread-safe structures |
| **Implicit Finite Difference** | \(O(NM)\)          | 🟢 Partially Parallelizable    | ✅ Yes (tridiagonal solver)            | 🟠 Moderate-High       | Synchronization on shared data (mutex) |


## Change Log

The following optimization techniques were applied to each model:

### 1. Binomial Tree
- **Rolling Storage:** only keep two layers, O(N) space complexity
- **Pointer Swap:** swap pointer instead of vector
- **Pruning:** Prune leading and trainling 0 avoid uncessary calculation

### 2. Trinomial Tree
- **Memory Optimization:** Rolling Storage

### 3. Monte Carlo
- 

### 4. Implicit Finite Difference
- 

### 5. Explicit Finite Difference
- 