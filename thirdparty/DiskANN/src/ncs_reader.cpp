#include "diskann/ncs_reader.h"
#include <stdexcept>
#include <future>

NCSReader::NCSReader(const milvus::NcsDescriptor* descriptor) {
    ncs_connector = std::unique_ptr<milvus::NcsConnector>(
        milvus::NcsConnectorFactory::Instance().createConnector(descriptor)
    );
}

NCSReader::~NCSReader() = default;

void NCSReader::read(std::vector<ReadReq> &read_reqs) {
    std::vector<uint32_t> keys;
    std::vector<milvus::SpanBytes> buffs;

    keys.reserve(read_reqs.size());
    buffs.reserve(read_reqs.size());

    for (const auto& req : read_reqs) {
        keys.push_back((uint32_t)req.key);
        buffs.push_back(milvus::SpanBytes(req.buf, req.len));
    }

    std::vector<milvus::NcsStatus> results = ncs_connector->multiGet(keys, buffs);
    for(auto status : results){
        if(status != milvus::NcsStatus::OK)
            throw std::runtime_error("fail to read from ncs");
    }
}

void NCSReader::get_submitted_req() {
    if (!async_read_future.has_value() || !async_read_future->valid()) {
        throw std::runtime_error("Error: Cannot retrieve result. No request was submitted or the result was already retrieved.");
    }
    async_read_future->get();
}

void NCSReader::submit_req(std::vector<ReadReq> &read_reqs) {
    if (async_read_future.has_value()) {
        if (async_read_future->valid()) {
            throw std::runtime_error("Error: A previous request was submitted but not yet retrieved with get_submitted_req().");
        }
    }
    async_read_future = std::async(std::launch::async, &NCSReader::read, this, std::ref(read_reqs));
}

