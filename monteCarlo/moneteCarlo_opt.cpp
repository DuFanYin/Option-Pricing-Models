#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <random>
#include <thread>
#include <chrono>
#include <functional>
#include <queue>
#include <condition_variable>
#include <mutex>
#include <atomic>
#include <future>
#include <sys/resource.h> 

// Helper to check mem usage
long getMemoryUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;  // in kilobytes (on Linux/Unix)
}

// Gaussian Elimination with Partial Pivoting
std::vector<double> solve_linear_system(std::vector<std::vector<double>>& A, std::vector<double>& B) {
    int n = A.size();  
    
    std::vector<int> pivot_indices(n);
    for (int i = 0; i < n; ++i) pivot_indices[i] = i;

    for (int i = 0; i < n; ++i) {
        int pivot = i;
        for (int j = i + 1; j < n; ++j) {
            if (abs(A[j][i]) > abs(A[pivot][i])) {
                pivot = j;
            }
        }
        if (A[pivot][i] == 0) throw std::runtime_error("Singular matrix, cannot solve.");
        
        if (pivot != i) {
            std::swap(A[i], A[pivot]);
            std::swap(B[i], B[pivot]);
        }

        double diag = A[i][i];
        for (int j = 0; j < n; ++j) A[i][j] /= diag;
        B[i] /= diag;

        for (int j = i + 1; j < n; ++j) {
            double factor = A[j][i];
            for (int k = i; k < n; ++k) {  
                A[j][k] -= factor * A[i][k];
            }
            B[j] -= factor * B[i];
        }
    }

    std::vector<double> X(n);
    for (int i = n - 1; i >= 0; --i) {
        X[i] = B[i];
        for (int j = i + 1; j < n; ++j) {
            X[i] -= A[i][j] * X[j];
        }
    }

    return X;
}

// Ordinary Least Squares to fit a quadratic function
std::vector<double> ordinary_least_squares(const std::vector<double>& X, const std::vector<double>& Y) {
    if (X.size() != Y.size() || X.empty()) throw std::invalid_argument("Vectors must be non-empty and same size.");

    int n = X.size();
    double sumX = 0, sumX2 = 0, sumX3 = 0, sumX4 = 0;
    double sumY = 0, sumXY = 0, sumX2Y = 0;

    for (int i = 0; i < n; ++i) {
        double x = X[i], y = Y[i];
        double x2 = x * x, x3 = x2 * x, x4 = x3 * x;

        sumX += x;
        sumX2 += x2;
        sumX3 += x3;
        sumX4 += x4;

        sumY += y;
        sumXY += x * y;
        sumX2Y += x2 * y;
    }

    std::vector<std::vector<double>> A = {
        {static_cast<double>(n), sumX, sumX2},
        {sumX, sumX2, sumX3},
        {sumX2, sumX3, sumX4}
    };

    std::vector<double> B = {sumY, sumXY, sumX2Y};

    return solve_linear_system(A, B); 
}

// Pupolate stock path and option price
void generatePaths(double** stockPath, double* optionPath, int startRow, int endRow,
    double S0, double K, double r, double sigma, double dt, int steps) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int j = startRow; j < endRow; ++j) {  
    stockPath[j][0] = S0;  

    for (int i = 1; i <= steps; ++i) {  
    double Z = dist(gen);  
    double dS = (r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z;
    stockPath[j][i] = stockPath[j][i - 1] * std::exp(dS);  
    }
    }

    for (int j = startRow; j < endRow; ++j) {  
    optionPath[j] = std::max(K - stockPath[j][steps], 0.0);
    }
}

// Backward induction to find x and y
void processBackwardInduction(int start, int end, int i, double** stockPath,
    double* optionPath, int* p, double r, double dt,
    std::vector<double>& x, std::vector<double>& y, std::mutex& xy_mutex) {
    
    std::vector<double> local_x, local_y; 

    for (int j = start; j < end; ++j) {
        if (optionPath[j] > 0) {
            local_x.push_back(stockPath[j][i]);
            local_y.push_back(optionPath[j] * std::exp(-r * (p[j] - i) * dt));
        }
    }

    {
        std::lock_guard<std::mutex> lock(xy_mutex);
        x.insert(x.end(), local_x.begin(), local_x.end());
        y.insert(y.end(), local_y.begin(), local_y.end());
    }
}

// ThreadPool class 
class ThreadPool {
    private:
        std::vector<std::thread> workers; 
        std::queue<std::function<void()>> taskQueue;
        std::mutex queueMutex; 
        std::condition_variable condition; 
        bool done; 

    public:
        ThreadPool(int numThreads) : done(false) {
            for (int i = 0; i < numThreads; ++i) {
                workers.push_back(std::thread([this, i] {
                    while (true) {
                        std::function<void()> task; 
                        {
                            std::unique_lock<std::mutex> lock(queueMutex);
    
                            condition.wait(lock, [this] { 
                                return done || !taskQueue.empty(); 
                            });
    
                            if (done && taskQueue.empty()) 
                                return;
    
                            task = std::move(taskQueue.front());
                            taskQueue.pop(); 
                        }
    
                        task();
                    }
                }));
            }
        }
    
        // Submit a task (generate path)
        template <typename F>
        void submit(F&& f) {
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                taskQueue.push(std::forward<F>(f)); 
            }
            condition.notify_one();
        }

        // Submit a task (backward induction)
        template <typename F>
        std::future<void> submit(F&& f, bool with_future) {
            auto task = std::make_shared<std::packaged_task<void()>>(std::forward<F>(f));
            std::future<void> future = task->get_future();

            {
                std::unique_lock<std::mutex> lock(queueMutex);
                taskQueue.push([task]() { (*task)(); }); 
            }

            condition.notify_one();
            return future; 
        }
    
        // Wait for all tasks to finish before shutting down the pool
        void wait() {
            {

                std::unique_lock<std::mutex> lock(queueMutex);
                done = true;
            }
            condition.notify_all(); 
            for (auto& worker : workers) {
                worker.join();
            }
        }
    
        // Shutdown the pool and join all threads
        void shutdown() {
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                done = true; 
            }
            condition.notify_all(); 
    
            for (auto& worker : workers) {
                if (worker.joinable()) {
                    worker.join(); 
                }
            }
        }
    
        ~ThreadPool() {
            shutdown(); 
        }
    };
     
// Monte Carlo simulation for American option pricing
double monteCarlo(int numberOfStockPath, int steps, double S0, double K, double r, double T, double sigma) {
    double dt = T / steps;

    // Memory for stockPath
    double** stockPath = (double**)malloc(numberOfStockPath * sizeof(double*));
    for (int i = 0; i < numberOfStockPath; ++i) {
        stockPath[i] = (double*)malloc((steps + 1) * sizeof(double)); 
    }

    // Memory for optionPath and p
    double* optionPath = (double*)malloc(numberOfStockPath * sizeof(double));
    int* p = (int*)malloc(numberOfStockPath * sizeof(int));

    int numThreads = 8;  
    int batchSize = numberOfStockPath / numThreads;  

    ThreadPool pool(numThreads);  

    std::promise<void> pathGenerationComplete;  
    std::future<void> pathGenerationFuture = pathGenerationComplete.get_future();
    
    for (int i = 0; i < numberOfStockPath; i += batchSize) {
        int startRow = i;
        int endRow = (i == numThreads - 1) ? numberOfStockPath : startRow + batchSize; 

        pool.submit([&, startRow, endRow, i] {
            generatePaths(stockPath, optionPath, startRow, endRow, S0, K, r, sigma, dt, steps);

            if (i + batchSize >= numberOfStockPath) {
                pathGenerationComplete.set_value(); 
            }
        });
    }

    pathGenerationFuture.get();  

    std::mutex xy_mutex;
    double* optionPriceAtStep = (double*)malloc(numberOfStockPath * sizeof(double));

    for (int i = steps - 1; i > 0; --i) {
        std::vector<double> x, y;
        std::vector<std::future<void>> taskFutures;  

        int batch_size = numberOfStockPath / numThreads;

        for (int t = 0; t < numThreads; ++t) {
            int start = t * batch_size;
            int end = (t == numThreads - 1) ? numberOfStockPath : start + batch_size;

            taskFutures.push_back(pool.submit([&, start, end, i] {
                processBackwardInduction(start, end, i, stockPath, optionPath, p, r, dt, x, y, xy_mutex);
            }, true)); 
        }

        // Wait for all task complete
        for (auto& fut : taskFutures) {
            fut.get();
        }
    
        if (x.empty() || y.empty() || x.size() != y.size()) continue;
        std::vector<double> c = ordinary_least_squares(x, y);
        if (c.size() < 3) continue;

        for (int j = 0; j < numberOfStockPath; ++j) {
            double optionPrice = optionPriceAtStep[j];  

            if (optionPrice > 0) {
                double fS = c[2] * stockPath[j][i] * stockPath[j][i] + c[1] * stockPath[j][i] + c[0];
                if (optionPrice > fS) {
                    p[j] = i;  
                    optionPath[j] = optionPrice;
                }
            }
        }
    }

    // Compute Monte Carlo estimate of option price
    double s = 0;
    for (int j = 0; j < numberOfStockPath; ++j) {
        double discount_factor = std::exp(-r * p[j] * dt);
        double payoff = optionPath[j];  
        s += discount_factor * payoff;
    }

    // clean up
    free(optionPriceAtStep);
    for (int i = 0; i < numberOfStockPath; ++i) {
        free(stockPath[i]);  
    }
    free(stockPath); 
    free(optionPath);  
    free(p); 

    return s / numberOfStockPath;
}

int main() {
    double S0 = 50;
    double K = 50;
    double r = 0.1;
    double T = 1;
    double sigma = 0.4;

    int numberOfStockPath = 20000;
    int steps = 2000;

    auto start = std::chrono::high_resolution_clock::now();
    long memoryBefore = getMemoryUsage();  // Measure memory usage before function call

    double optionPrice = monteCarlo(numberOfStockPath, steps, S0, K, r, T, sigma);

    long memoryAfter = getMemoryUsage();  // Measure memory usage after function call
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    long memoryUsed = memoryAfter - memoryBefore;  

    std::cout << "Estimated Option Price: " << optionPrice << std::endl;
    std::cout << "Execution Time (ms): " << duration.count() * 1e6 << std::endl;
    std::cout << "Memory Usage (MB): " << memoryUsed / 1024 << "\n";

    return 0;
}

// clang++ -pthread -o opt moneteCarlo_opt.cpp && ./optm