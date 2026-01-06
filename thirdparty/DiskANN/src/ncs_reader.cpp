#include "diskann/ncs_reader.h"
#include <stdexcept>
#include <future>

NCSReader::NCSReader(const milvus::NcsDescriptor* descriptor)
    : descriptor_(*descriptor) {
}

NCSReader::~NCSReader() {
    // Clean up any thread-local connectors associated with this reader
    // Note: This only cleans up the current thread's connector.
    // Other threads' connectors will be cleaned up when those threads exit
    // or when the thread_local map is cleared.
    auto it = thread_local_connectors_.find(this);
    if (it != thread_local_connectors_.end()) {
        thread_local_connectors_.erase(it);
    }
}

milvus::NcsConnector* NCSReader::getThreadLocalConnector() {
    auto it = thread_local_connectors_.find(this);
    if (it != thread_local_connectors_.end()) {
        return it->second.get();
    }
    
    // Create a new connector for this thread
    auto connector = std::unique_ptr<milvus::NcsConnector>(
        milvus::NcsConnectorFactory::Instance().createConnector(&descriptor_)
    );
    
    if (!connector) {
        throw std::runtime_error("Failed to create NCS connector for thread");
    }
    
    auto* ptr = connector.get();
    thread_local_connectors_[this] = std::move(connector);
    return ptr;
}

void NCSReader::read(std::vector<ReadReq> &read_reqs) {
    auto* connector = getThreadLocalConnector();
    
    std::vector<uint32_t> keys;
    std::vector<milvus::SpanBytes> buffs;

    keys.reserve(read_reqs.size());
    buffs.reserve(read_reqs.size());

    for (const auto& req : read_reqs) {
        keys.push_back((uint32_t)req.key);
        buffs.push_back(milvus::SpanBytes(req.buf, req.len));
    }

    std::vector<milvus::NcsStatus> results = connector->multiGet(keys, buffs);
    for(auto status : results){
        if(status != milvus::NcsStatus::OK)
            throw std::runtime_error("fail to read from ncs");
    }
}

void NCSReader::get_submitted_req() {
    if (!async_read_future_.has_value() || !async_read_future_->valid()) {
        throw std::runtime_error("Error: Cannot retrieve result. No request was submitted or the result was already retrieved.");
    }
    async_read_future_->get();
}

void NCSReader::submit_req(std::vector<ReadReq> &read_reqs) {
    if (async_read_future_.has_value()) {
        if (async_read_future_->valid()) {
            throw std::runtime_error("Error: A previous request was submitted but not yet retrieved with get_submitted_req().");
        }
    }
    async_read_future_ = std::async(std::launch::async, &NCSReader::read, this, std::ref(read_reqs));
}

