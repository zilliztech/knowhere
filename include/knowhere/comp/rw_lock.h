//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.
#ifndef KNOWHERE_RW_LOCK_H
#define KNOWHERE_RW_LOCK_H
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
/*
FairRWLock is a fair MultiRead-SingleWrite lock
*/
namespace knowhere {
class FairRWLock {
 public:
    FairRWLock() = default;

    void
    LockRead() {
        std::unique_lock<std::mutex> lk(mtx_);
        auto id = id_counter_++;

        read_cv_.wait(lk,
                      [this, id]() { return !writer_ && (write_requests_.empty() || write_requests_.front() > id); });
        ++readers_;
    }

    void
    UnLockRead() {
        std::unique_lock<std::mutex> lk(mtx_);
        if (--readers_ == 0 && !write_requests_.empty()) {
            write_cv_.notify_one();
        }
    }

    void
    LockWrite() {
        std::unique_lock<std::mutex> lk(mtx_);
        auto id = id_counter_++;
        write_requests_.push(id);

        write_cv_.wait(lk, [this, id]() { return !writer_ && readers_ == 0 && write_requests_.front() == id; });
        writer_ = true;
    }

    void
    UnLockWrite() {
        std::unique_lock<std::mutex> lk(mtx_);
        writer_ = false;
        write_requests_.pop();
        if (!write_requests_.empty()) {
            write_cv_.notify_one();
        } else {
            read_cv_.notify_all();
        }
    }

 private:
    uint64_t id_counter_ = 0;
    std::mutex mtx_;
    std::condition_variable read_cv_;
    std::condition_variable write_cv_;
    uint64_t readers_ = 0;
    bool writer_ = false;
    std::queue<uint64_t> write_requests_;
};

class FairReadLockGuard {
 public:
    explicit FairReadLockGuard(FairRWLock& lock) : lock_(lock) {
        lock_.LockRead();
    }

    ~FairReadLockGuard() {
        lock_.UnLockRead();
    }

 private:
    FairRWLock& lock_;
};

class FairWriteLockGuard {
 public:
    explicit FairWriteLockGuard(FairRWLock& lock) : lock_(lock) {
        lock_.LockWrite();
    }

    ~FairWriteLockGuard() {
        lock_.UnLockWrite();
    }

 private:
    FairRWLock& lock_;
};
}  // namespace knowhere
#endif
