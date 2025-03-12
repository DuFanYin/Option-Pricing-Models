# Option_Pricing


This project implements five option pricing methods that I learned from **SMU QF101**, including:  

- **CRR Binary Tree**  
- **CRR Ternary Tree**  
- **Monte Carlo Simulation**  
- **Explicit Finite Difference**  
- **Implicit Finite Difference**  

Additionally, this project applies **multithreading** and **memory optimization techniques** to enhance performance.  

## Comparison of Option Pricing Methods

| **Method**                   | **Time Complexity** | **Multithreading Feasibility** | **Memory Pool Optimization Feasibility** | **Computational Cost** |
|------------------------------|--------------------|--------------------------------|--------------------------------|------------------|
| **Binomial Tree**            | \(O(N^2)\)        | 🔴 Limited                     | ✅ Yes (rolling storage)       | 🟠 Medium        |
| **Trinomial Tree**           | \(O(N^2)\)        | 🔴 Limited                     | ✅ Yes (rolling storage)       | 🟠 High         |
| **Monte Carlo Simulation**   | \(O(M)\)          | 🟢 Highly Parallelizable       | ✅ Yes (preallocated paths)    | 🔴 Heavy        |
| **Explicit Finite Difference** | \(O(NM)\)       | 🟢 Parallelizable              | ✅ Yes (sparse grid)          | 🟠 Moderate-High |
| **Implicit Finite Difference** | \(O(NM)\)       | 🟢 Partially Parallelizable    | ✅ Yes (tridiagonal solver)   | 🟠 Moderate-High |