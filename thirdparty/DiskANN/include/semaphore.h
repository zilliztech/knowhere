#pragma once
#include <mutex>
#include <condition_variable>

namespace diskann {
class Semaphore {
public:
    Semaphore(long count = 0) : count(count) {}
    void Signal()
    {
        std::unique_lock<std::mutex> unique(mt);
        ++count;
        if (count <= 0) {
            cond.notify_one();
        }
    }
    void Wait()
    {
        std::unique_lock<std::mutex> unique(mt);
        --count;
        if (count < 0) {
            cond.wait(unique);
        }
    }
    bool IsWaitting() {
        std::unique_lock<std::mutex> unique(mt);
        return count < 0;
    }
    
private:
    std::mutex mt;
    std::condition_variable cond;
    long count;
};
}  // namespace diskann