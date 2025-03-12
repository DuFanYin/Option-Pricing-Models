# Option_Pricing


This project implements five option pricing methods that I learned from **SMU QF101**, including:  

- **Binomial Tree**  
- **Trinomial Tree**  
- **Monte Carlo Simulation**  
- **Explicit Finite Difference**  
- **Implicit Finite Difference**  

Additionally, this project applies **multithreading** and **memory optimization techniques** to enhance performance.  

## Comparison of Option Pricing Methods

| **Method**                    | **Time Complexity** | **Multithreading Feasibility** | **Memory Pool Optimization Feasibility** | **Computational Cost** | **Suitable Synchronization Method** |
|-------------------------------|--------------------|--------------------------------|----------------------------------------|-----------------------|-------------------------------------|
| **Binomial Tree**              | \(O(N^2)\)         | 🔴 Limited                     | ✅ Yes (rolling storage)               | 🟠 Medium              | Mutex/Lock for thread-safe updates  |
| **Trinomial Tree**             | \(O(N^2)\)         | 🔴 Limited                     | ✅ Yes (rolling storage)               | 🟠 High                | Mutex/Lock for thread-safe updates  |
| **Monte Carlo Simulation**     | \(O(M)\)           | 🟢 Highly Parallelizable       | ✅ Yes (preallocated paths)            | 🔴 Heavy               | Thread Pool / Atomic Operations     |
| **Explicit Finite Difference** | \(O(NM)\)          | 🟢 Parallelizable              | ✅ Yes (sparse grid)                   | 🟠 Moderate-High       | Lock-free or thread-safe structures |
| **Implicit Finite Difference** | \(O(NM)\)          | 🟢 Partially Parallelizable    | ✅ Yes (tridiagonal solver)            | 🟠 Moderate-High       | Synchronization on shared data (mutex) |