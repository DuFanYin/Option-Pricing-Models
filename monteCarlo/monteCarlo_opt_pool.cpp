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

// Function to solve a linear system of equations using Gaussian elimination
std::vector<double> solve_linear_system(std::vector<std::vector<double>> A, std::vector<double> B) {
    int n = 3; // 3x3 system
    for (int i = 0; i < n; ++i) {
        // Pivoting (for numerical stability)
        int pivot = i;
        for (int j = i + 1; j < n; ++j) {
            if (abs(A[j][i]) > abs(A[pivot][i])) {
                pivot = j;
            }
        }
        std::swap(A[i], A[pivot]);
        std::swap(B[i], B[pivot]);

        // Make leading coefficient 1
        double diag = A[i][i];
        if (diag == 0) throw std::runtime_error("Singular matrix, cannot solve.");
        for (int j = 0; j < n; ++j) A[i][j] /= diag;
        B[i] /= diag;

        // Eliminate below
        for (int j = i + 1; j < n; ++j) {
            double factor = A[j][i];
            for (int k = 0; k < n; ++k) A[j][k] -= factor * A[i][k];
            B[j] -= factor * B[i];
        }
    }

    // Back substitution
    std::vector<double> X(n);
    for (int i = n - 1; i >= 0; --i) {
        X[i] = B[i];
        for (int j = i + 1; j < n; ++j) {
            X[i] -= A[i][j] * X[j];
        }
    }
    return X;
}

// Ordinary Least Squares (OLS) to fit a quadratic function
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

    // Construct the system Ax = B
    std::vector<std::vector<double>> A = {
        {static_cast<double>(n), sumX, sumX2},
        {sumX, sumX2, sumX3},
        {sumX2, sumX3, sumX4}
    };

    std::vector<double> B = {sumY, sumXY, sumX2Y};

    return solve_linear_system(A, B); // Returns {c, b, a}
}

void generatePaths(std::vector<std::vector<double>>& stockPath, std::vector<std::vector<double>>& optionPath, int startRow, int endRow,
    double S0, double K, double r, double sigma, double dt, int steps) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int j = startRow; j < endRow; ++j) {  
        stockPath[j][0] = S0;  // First column (step 0) is the initial stock price
        optionPath[j][0] =  std::max( K - S0, 0.0);

        for (size_t i = 1; i < steps; ++i) {  // Start from step 1
            double Z = dist(gen);  
            double dS = (r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z;
            stockPath[j][i] = stockPath[j][i - 1] * std::exp(dS);  // Use the previous step price
            optionPath[j][i] = std::max(K - stockPath[j][i], 0.0);
    }
}
}

void processBackwardInduction(int start, int end, int i, const std::vector<std::vector<double>>& stockPath,
    std::vector<std::vector<double>>& optionPath, std::vector<int>& p, double r, double dt,
    std::vector<double>& x, std::vector<double>& y, std::mutex& xy_mutex) {

    std::vector<double> local_x, local_y; // Use local vectors to avoid race conditions

    for (int j = start; j < end; ++j) {
        if (j >= stockPath.size() || j >= optionPath.size() || j >= p.size() || 
            i >= stockPath[j].size() || i >= optionPath[j].size() || 
            p[j] >= optionPath[j].size()) {
            continue; // Skip invalid indices to prevent segmentation faults
        }

        if (optionPath[j][i] > 0) {
            local_x.push_back(stockPath[j][i]);
            local_y.push_back(optionPath[j][p[j]] * std::exp(-r * (p[j] - i) * dt));
        }
    }

    {
        std::lock_guard<std::mutex> lock(xy_mutex);
        x.insert(x.end(), local_x.begin(), local_x.end());
        y.insert(y.end(), local_y.begin(), local_y.end());
    }
}

// ThreadPool class that manages a pool of threads
class ThreadPool {
    private:
        std::vector<std::thread> workers; // Vector of worker threads
        std::queue<std::function<void()>> taskQueue; // Queue holding the tasks
        std::mutex queueMutex; // Mutex to protect shared access to the task queue
        std::condition_variable condition; // Condition variable to synchronize threads
        bool done; // Flag indicating if the pool has been shut down

    public:
        // Constructor to initialize the pool with a specified number of threads
        ThreadPool(int numThreads) : done(false) {
            // Create worker threads and start processing tasks
            for (int i = 0; i < numThreads; ++i) {
                workers.push_back(std::thread([this, i] {
                    while (true) {
                        std::function<void()> task; // Task that the thread will execute
                        {
                            // Lock the mutex to access the task queue
                            std::unique_lock<std::mutex> lock(queueMutex);
    
                            // Wait for tasks to be available or the pool to be shut down
                            condition.wait(lock, [this] { 
                                return done || !taskQueue.empty(); 
                            });
    
                            // If the pool is done and the queue is empty, exit the thread
                            if (done && taskQueue.empty()) 
                                return;
    
                            // Retrieve the task from the front of the queue
                            task = std::move(taskQueue.front());
                            taskQueue.pop(); // Remove the task from the queue
                        }
    
                        // Execute the task
                        task();
                    }
                }));
            }
        }
    
        // Submit a task to the pool
        // The task can be any callable (function, lambda, etc.)
        template <typename F>
        void submit(F&& f) {
            {
                // Lock the mutex before adding the task to the queue
                std::unique_lock<std::mutex> lock(queueMutex);
                taskQueue.push(std::forward<F>(f)); // Add the task to the queue
            }
    
            // Notify one of the threads that there is a new task
            condition.notify_one();
        }

        template <typename F>
        std::future<void> submit(F&& f, bool with_future) {
            auto task = std::make_shared<std::packaged_task<void()>>(std::forward<F>(f));
            std::future<void> future = task->get_future();

            {
                std::unique_lock<std::mutex> lock(queueMutex);
                taskQueue.push([task]() { (*task)(); }); // Wrap the function inside the task
            }

            condition.notify_one();
            return future; // Return future so caller can track completion
        }
    
        // Wait for all tasks to finish before shutting down the pool
        void wait() {
            {
                // Lock the mutex and mark the pool as "done"
                std::unique_lock<std::mutex> lock(queueMutex);
                done = true;
            }
            condition.notify_all(); // Notify all threads to exit
            for (auto& worker : workers) {
                worker.join(); // Join each thread to wait for their completion
            }
        }
    
        // Shutdown the pool and join all threads
        void shutdown() {
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                done = true; // Mark the pool as done
            }
            condition.notify_all(); // Notify all threads to exit
    
            // Join each thread to ensure all threads have finished
            for (auto& worker : workers) {
                if (worker.joinable()) {
                    worker.join(); // Wait for each worker thread to finish
                }
            }
        }
    
        // Destructor that shuts down the thread pool when it is no longer needed
        ~ThreadPool() {
            shutdown(); // Ensure all tasks are finished and threads are joined
        }
    };
     
// Monte Carlo simulation for American option pricing
double monteCarlo(int numberOfstockPath, int steps, double S0, double K, double r, double T, double sigma) {
    double dt = T / steps;

    std::vector<std::vector<double>> stockPath(numberOfstockPath, std::vector<double>(steps + 1, 0));
    std::vector<std::vector<double>> optionPath(numberOfstockPath, std::vector<double>(steps + 1, 0));
    std::vector<int> p(numberOfstockPath, steps);

    int numThreads = 8;  
    int dynamicBatchSize = std::max(500, 1);  // Avoid zero batch size

    ThreadPool pool(numThreads);  

    std::promise<void> pathGenerationComplete;  
    std::future<void> pathGenerationFuture = pathGenerationComplete.get_future();
    
    for (int i = 0; i < numberOfstockPath; i += dynamicBatchSize) {
        int startRow = i;
        int endRow = std::min(i + dynamicBatchSize, numberOfstockPath);

        pool.submit([&, startRow, endRow, i] {
            generatePaths(stockPath, optionPath, startRow, endRow, S0, K, r, sigma, dt, steps);

            if (i + dynamicBatchSize >= numberOfstockPath) {
                pathGenerationComplete.set_value(); 
            }
        });
    }

    pathGenerationFuture.get();  

    std::mutex xy_mutex;
    
    for (int i = steps - 1; i > 0; --i) {
        std::vector<double> x, y;
        std::vector<std::future<void>> futures;  // Collect task futures

        int batch_size = numberOfstockPath / numThreads;

        for (int t = 0; t < numThreads; ++t) {
            int start = t * batch_size;
            int end = (t == numThreads - 1) ? numberOfstockPath : start + batch_size;

            futures.push_back(pool.submit([&, start, end, i] {
                processBackwardInduction(start, end, i, stockPath, optionPath, p, r, dt, x, y, xy_mutex);
            }, true));  // Pass `true` to use the future-returning version
        }

        // Ensure all tasks complete before proceeding
        for (auto& fut : futures) {
            fut.get();
        }
    

        // Perform your post-processing here (ordinary least squares, etc.)
        if (x.empty() || y.empty() || x.size() != y.size()) continue;
        std::vector<double> c = ordinary_least_squares(x, y);
        if (c.size() < 3) continue;

        // Process results and update paths
        for (int j = 0; j < numberOfstockPath; ++j) {
            if (optionPath[j][i] > 0) {
                double fS = c[2] * stockPath[j][i] * stockPath[j][i] + c[1] * stockPath[j][i] + c[0];
                if (optionPath[j][i] > fS) {
                    p[j] = i;
                } else if (p[j] > i) {
                    optionPath[j][i] = 0;
                } else {
                    optionPath[j][i] = optionPath[j][i + 1] * std::exp(-r * dt);
                }
            }
        }
    }

    pool.wait();

    double s = 0;
    for (int j = 0; j < numberOfstockPath; ++j) {
        double discount_factor = std::exp(-r * p[j] * dt);
        double payoff = optionPath[j][p[j]];  
        s += discount_factor * payoff;
    }

    return s / numberOfstockPath;
}

int main() {
    double S0 = 50;
    double K = 50;
    double r = 0.1;
    double T = 1;
    double sigma = 0.4;

    int numberOfstockPath = 10000;
    int steps = 500;

    // Run Monte Carlo simulation
    auto start = std::chrono::high_resolution_clock::now();
    double optionPrice = monteCarlo(numberOfstockPath, steps, S0, K, r, T, sigma);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;


    // Output the estimated option price
    std::cout << "Estimated Option Price: " << optionPrice << std::endl;
    std::cout << "Execution Time (ms): " << duration.count() * 1e6 << std::endl;

    return 0;
}


// clang++ -pthread -o mc monteCarlo_opt_pool.cpp && ./mc