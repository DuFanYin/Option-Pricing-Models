#include <atomic>
#include <functional>
#include <thread>
#include <vector>
#include <condition_variable>
#include <memory>
#include <future>
#include <iostream>

// Lock-free queue implementation using atomic pointers
template <typename T>
class LockFreeQueue {
private:
    struct Node {
        T value;
        std::atomic<Node*> next{nullptr};

        explicit Node(T val) : value(std::move(val)) {}
    };

    std::atomic<Node*> head{nullptr};
    std::atomic<Node*> tail{nullptr};

public:
    LockFreeQueue() {
        head.store(new Node(T()));  // dummy node
        tail.store(head.load());
    }

    ~LockFreeQueue() {
        // Clean up allocated nodes
        while (head.load() != nullptr) {
            Node* node = head.load();
            head.store(node->next.load());
            delete node;
        }
    }

    void push(T value) {
        Node* newNode = new Node(std::move(value));
        Node* oldTail = tail.load();
        
        // Atomically change the tail pointer to the new node
        while (!std::atomic_compare_exchange_weak(&oldTail->next, &oldTail->next, newNode)) {
            oldTail = tail.load();
        }

        tail.store(newNode);
    }

    bool pop(T& result) {
        Node* oldHead = head.load();
        Node* newHead = oldHead->next.load();

        if (newHead == nullptr) {
            return false; // Queue is empty
        }

        result = std::move(newHead->value);
        head.store(newHead);
        delete oldHead;

        return true;
    }

    bool empty() {
        return head.load() == tail.load();
    }
};

class ThreadPool {
private:
    std::vector<std::thread> workers;
    LockFreeQueue<std::function<void()>> taskQueue;
    std::atomic<bool> done{false};
    std::condition_variable condition;
    std::mutex conditionMutex;

public:
    ThreadPool(int numThreads) {
        for (int i = 0; i < numThreads; ++i) {
            workers.push_back(std::thread([this, i] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(conditionMutex);
                        condition.wait(lock, [this] {
                            return done || !taskQueue.empty();
                        });

                        if (done && taskQueue.empty()) {
                            return;
                        }

                        taskQueue.pop(task);
                    }

                    task();
                }
            }));
        }
    }

    template <typename F>
    void submit(F&& f) {
        {
            taskQueue.push(std::forward<F>(f));
        }
        condition.notify_one();
    }

    template <typename F>
    std::future<void> submit(F&& f, bool with_future) {
        auto task = std::make_shared<std::packaged_task<void()>>(std::forward<F>(f));
        std::future<void> future = task->get_future();

        {
            taskQueue.push([task]() { (*task)(); });
        }

        condition.notify_one();
        return future;
    }

    void wait() {
        {
            std::unique_lock<std::mutex> lock(conditionMutex);
            done.store(true);
        }
        condition.notify_all();

        for (auto& worker : workers) {
            worker.join();
        }
    }

    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(conditionMutex);
            done.store(true);
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