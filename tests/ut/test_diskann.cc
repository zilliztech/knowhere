// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <sys/resource.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <random>
#include <string>
#include <thread>

#include "../DiskANN/include/diskann/defaults.h"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "filemanager/FileManager.h"
#include "filemanager/impl/LocalFileManager.h"
#include "index/diskann/diskann_config.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/knowhere_check.h"
#include "knowhere/context.h"
#include "knowhere/expected.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/utils.h"
#include "knowhere/version.h"
#include "utils.h"
#include "ncs/InMemoryNcs.h"
#include "ncs/InMemNcsConnector.h"
#include "ncs/RedisNcs.h"
#include "ncs/RedisNcsConnector.h"
#include "diskann/ncs_reader.h"
#include "diskann/file_index_reader.h"


#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
error "Missing the <filesystem> header."
#endif
#include <fstream>

namespace {
std::string kDir = fs::current_path().string() + "/diskann_test";
std::string kRawDataPath = kDir + "/raw_data";
std::string kL2IndexDir = kDir + "/l2_index";
std::string kIPIndexDir = kDir + "/ip_index";
std::string kCOSINEIndexDir = kDir + "/cosine_index";
std::string kL2IndexPrefix = kL2IndexDir + "/l2";
std::string kIPIndexPrefix = kIPIndexDir + "/ip";
std::string kCOSINEIndexPrefix = kCOSINEIndexDir + "/cosine";

std::string kEmbListL2IndexDir = kDir + "/emb_list_l2_index";
std::string kEmbListIPIndexDir = kDir + "/emb_list_ip_index";
std::string kEmbListCOSINEIndexDir = kDir + "/emb_list_cosine_index";
std::string kEmbListL2IndexPrefix = kEmbListL2IndexDir + "/max_sim_l2";
std::string kEmbListIPIndexPrefix = kEmbListIPIndexDir + "/max_sim_ip";
std::string kEmbListCOSINEIndexPrefix = kEmbListCOSINEIndexDir + "/max_sim_cosine";
std::string kEmbListOffsetPath = kDir + "/emb_list_offset.bin";

constexpr uint32_t kNumRows = 1000;
constexpr uint32_t kNumQueries = 10;
constexpr uint32_t kDim = 128;
constexpr uint32_t kLargeDim = 256;
constexpr uint32_t kK = 10;
constexpr float kKnnRecall = 0.9;
constexpr float kEmbListKnnRecall = 0.75;
constexpr float AiSAQKnnRecall = 0.01;
constexpr float kL2RangeAp = 0.9;
constexpr float kIpRangeAp = 0.9;
constexpr float kCosineRangeAp = 0.9;
}  // namespace

namespace knowhere {
    template <typename DataType>
    uint64_t GetDiskANNNodeSectorOffsetForTest(knowhere::Index<knowhere::IndexNode>& index, uint64_t node_id);
    template <typename DataType>
    char* GetDiskANNOffsetToNodeForTest(knowhere::Index<knowhere::IndexNode>& index, char* sector_buf, uint64_t node_id);
    template <typename DataType>
    uint64_t GetDiskANNMaxNodeLenForTest(knowhere::Index<knowhere::IndexNode>& index);
    template <typename DataType>
    size_t GetDiskANNReadLenForNodeForTest(knowhere::Index<knowhere::IndexNode>& index);
}


using namespace milvus;

namespace {
/**
 * @brief Get NCS extras config for the given NCS kind.
 * @param ncs_kind The NCS backend type ("in_memory" or "redis")
 * @return json config with appropriate settings for the backend
 */
json getNcsExtras(const std::string& ncs_kind) {
    if (ncs_kind == "redis") {
        return json{{"redis_host", "localhost"}, {"redis_port", 6379}};
    }
    return json::object();
}

/**
 * @brief Initialize NCS singleton for the given NCS kind.
 * Resets the singleton first to ensure clean state.
 * @param ncs_kind The NCS backend type ("in_memory" or "redis")
 */
void initNcsForTest(const std::string& ncs_kind) {
    NcsSingleton::reset();
    if (ncs_kind == "in_memory") {
        NcsSingleton::initNcs(InMemoryNcsFactory::KIND);
    } else {
        NcsSingleton::initNcs(RedisNcsFactory::KIND, getNcsExtras(ncs_kind));
    }
}
} // namespace

// Tests basic NCS connector operations (multiPut, multiGet, multiDelete, bucket management) 
// with error handling for oversized buffers. Runs with both in_memory and redis NCS backends.
TEST_CASE("InMemoryNcsConnector - sanity", "[NcsTest]") {
    auto ncs_kind = GENERATE("in_memory", "redis");
    const uint32_t bucketId = 1;
    
    initNcsForTest(ncs_kind);
    
    Ncs* ncs = NcsSingleton::Instance();
    auto createResult = ncs->createBucket(bucketId);
    REQUIRE(createResult == NcsStatus::OK);
    
    // Create descriptor and connector
    auto ncs_extras = getNcsExtras(ncs_kind);
    
    auto descriptor = NcsDescriptor(ncs_kind, bucketId, ncs_extras);
    auto connector = std::unique_ptr<NcsConnector>(
        NcsConnectorFactory::Instance().createConnector(&descriptor));
    
    REQUIRE(connector != nullptr);

    // Prepare test data with varying sizes
    std::vector<uint32_t> keys = {1, 2, 3};
    std::vector<std::vector<uint8_t>> values = {
        std::vector<uint8_t>(100, 0x11),  // 100 bytes
        std::vector<uint8_t>(200, 0x22),  // 200 bytes
        std::vector<uint8_t>(300, 0x33)   // 300 bytes
    };
    std::vector<SpanBytes> valueSpans;
    for (auto& value : values) {
        valueSpans.emplace_back(value.data(), value.size());
    }

    // Test multiPut
    auto putResults = connector->multiPut(keys, valueSpans);
    REQUIRE(putResults.size() == keys.size());
    for (const auto& status : putResults) {
        REQUIRE(status == NcsStatus::OK);
    }

    // Prepare buffers for reading
    std::vector<std::vector<uint8_t>> readBuffers;
    std::vector<SpanBytes> readSpans;
    for (const auto& value : values) {
        readBuffers.emplace_back(value.size());
        readSpans.emplace_back(readBuffers.back().data(), readBuffers.back().size());
    }

    // Test multiGet
    auto getResults = connector->multiGet(keys, readSpans);
    REQUIRE(getResults.size() == keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        REQUIRE(getResults[i] == NcsStatus::OK);
        REQUIRE(readBuffers[i] == values[i]);
    }

    // Test with buffer too small
    std::vector<uint8_t> smallBuffer(50);
    SpanBytes smallSpan(smallBuffer.data(), smallBuffer.size());
    auto getResult = connector->multiGet({keys[2]}, {smallSpan});
    REQUIRE(getResult.size() == 1);
    REQUIRE(getResult[0] == NcsStatus::ERROR);  // Should fail as buffer is too small

    // Test multiDelete for specific keys
    auto deleteResult = connector->multiDelete(keys);
    REQUIRE(deleteResult.size() == keys.size());
    for (const auto& status : deleteResult) {
        REQUIRE(status == NcsStatus::OK);
    }

    // Verify data is deleted
    auto getResultsAfterDelete = connector->multiGet(keys, readSpans);
    for (const auto& status : getResultsAfterDelete) {
        REQUIRE(status == NcsStatus::ERROR);
    }

    // Test bucket deletion
    auto bucketDeleteResult = ncs->deleteBucket(bucketId);
    REQUIRE(bucketDeleteResult == NcsStatus::OK);
}

// Tests NCS reader for both synchronous and asynchronous read operations after data has been
// put into NCS storage. Verifies data integrity through both read methods.
// Runs with both in_memory and redis NCS backends.
TEST_CASE("NcsReader", "[NcsTest]") {
    auto ncs_kind = GENERATE("in_memory", "redis");
    const uint32_t bucketId = 1;
    
    initNcsForTest(ncs_kind);
    
    Ncs* ncs = NcsSingleton::Instance();
    auto createResult = ncs->createBucket(bucketId);
    REQUIRE(createResult == NcsStatus::OK);
    
    auto ncs_extras = getNcsExtras(ncs_kind);
    
    auto descriptor = NcsDescriptor(ncs_kind, bucketId, ncs_extras);
    auto connector = std::unique_ptr<NcsConnector>(
        NcsConnectorFactory::Instance().createConnector(&descriptor));
    
    REQUIRE(connector != nullptr);

    std::vector<uint32_t> keys = {1, 2, 3};
    std::vector<size_t> value_sizes = {100, 200, 300};
    std::vector<uint8_t> value_patterns = {0x11, 0x22, 0x33};
    std::vector<std::vector<uint8_t>> values;
    for (size_t i = 0; i < keys.size(); ++i) {
        values.emplace_back(value_sizes[i], value_patterns[i]);
    }

    std::vector<SpanBytes> valueSpans;
    for (auto& value : values) {
        valueSpans.emplace_back(value.data(), value.size());
    }

    auto putResults = connector->multiPut(keys, valueSpans);
    REQUIRE(putResults.size() == keys.size());
    for (const auto& status : putResults) {
        REQUIRE(status == NcsStatus::OK);
    }

    auto ncs_reader = std::make_unique<NCSReader>(&descriptor);

    std::vector<ReadReq> read_reqs;
    auto buffer_size = *std::max_element(value_sizes.begin(), value_sizes.end());
    std::vector<std::vector<char>> buffers(keys.size(), std::vector<char>(buffer_size));

    for(uint i = 0; i < keys.size() ; i++){
        read_reqs.push_back({(uint32_t)keys[i], values[i].size(), buffers[i].data()});
    }

    ncs_reader->read(read_reqs);
    int i=0;
    for(auto read_req : read_reqs){
        for(int j=0; j < read_req.len ; j++) 
            REQUIRE(static_cast<uint8_t*>(read_req.buf)[j] == static_cast<uint8_t*>(valueSpans[i].data())[j] );
        i++;
    }

    std::vector<ReadReq> async_read_reqs;
    std::vector<std::vector<char>> async_buffers(keys.size(), std::vector<char>(buffer_size));


    for(uint i = 0; i < keys.size() ; i++){
        async_read_reqs.push_back({(uint32_t)keys[i], values[i].size(), async_buffers[i].data()});
    }

    ncs_reader->submit_req(async_read_reqs);
    ncs_reader->get_submitted_req();
    i=0;
    for(auto read_req : async_read_reqs){
        for(int j=0; j < read_req.len ; j++) 
            REQUIRE(static_cast<uint8_t*>(read_req.buf)[j] == static_cast<uint8_t*>(valueSpans[i].data())[j] );
        i++;
    }

}

// Tests NCSReader thread_local connector model with concurrent access from multiple threads.
// Each thread should transparently get its own connector instance via thread_local storage.
// This validates the concurrency model where a single NCSReader can be safely shared across threads.
// Runs with both in_memory and redis NCS backends.
TEST_CASE("NCSReader - concurrent thread_local access", "[NcsTest]") {
    auto ncs_kind = GENERATE("in_memory", "redis");
    const uint64_t bucketId = 9999;
    const size_t numThreads = 16;
    const size_t opsPerThread = 10000;
    const size_t numKeys = 100;
    const size_t valueSize = 4096;
    const size_t maxBatchSize = 5;
    
    initNcsForTest(ncs_kind);
    
    Ncs* ncs = NcsSingleton::Instance();
    auto createResult = ncs->createBucket(bucketId);
    REQUIRE(createResult == NcsStatus::OK);
    
    // Create descriptor for NCSReader
    auto ncs_extras = getNcsExtras(ncs_kind);
    NcsDescriptor descriptor(ncs_kind, bucketId, ncs_extras);
    
    // Pre-populate data using a connector
    std::unique_ptr<NcsConnector> setup_connector(
        NcsConnectorFactory::Instance().createConnector(&descriptor));
    REQUIRE(setup_connector != nullptr);
    
    std::vector<uint32_t> allKeys;
    std::vector<std::vector<uint8_t>> allValues;
    std::vector<SpanBytes> allSpans;
    
    for (size_t i = 0; i < numKeys; ++i) {
        allKeys.push_back(static_cast<uint32_t>(i));
        allValues.emplace_back(valueSize, static_cast<uint8_t>(i % 256));
    }
    for (auto& v : allValues) {
        allSpans.emplace_back(v.data(), v.size());
    }
    
    auto putResults = setup_connector->multiPut(allKeys, allSpans);
    for (const auto& r : putResults) {
        REQUIRE(r == NcsStatus::OK);
    }
    setup_connector.reset();
    
    // Create a single NCSReader instance to be shared across threads
    NCSReader ncs_reader(&descriptor);
    
    // Track success/failure across threads
    std::atomic<size_t> successfulOps{0};
    std::atomic<size_t> failedOps{0};
    std::vector<std::thread> threads;
    
    // Barrier to synchronize all threads before their first read
    std::mutex barrierMutex;
    std::condition_variable barrierCv;
    size_t readyCount = 0;
    bool startFlag = false;
    
    // Launch multiple threads that all use the same NCSReader instance
    // Each thread will get its own connector via thread_local storage
    for (size_t t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 rng(static_cast<unsigned>(t));
            std::uniform_int_distribution<size_t> keyDist(0, numKeys - 1);
            std::uniform_int_distribution<size_t> batchDist(1, maxBatchSize);
            
            // Pre-allocate buffers for max batch size before starting iterations
            std::vector<std::vector<uint8_t>> buffers;
            buffers.reserve(maxBatchSize);
            for (size_t b = 0; b < maxBatchSize; ++b) {
                buffers.emplace_back(valueSize);
            }
            
            // Signal ready and wait for all threads to be ready
            {
                std::unique_lock<std::mutex> lock(barrierMutex);
                ++readyCount;
                if (readyCount == numThreads) {
                    // Last thread to arrive - signal everyone to start
                    startFlag = true;
                    barrierCv.notify_all();
                } else {
                    // Wait for the start signal
                    barrierCv.wait(lock, [&] { return startFlag; });
                }
            }
            
            for (size_t op = 0; op < opsPerThread; ++op) {
                size_t batchSize = batchDist(rng);
                std::vector<ReadReq> read_reqs;
                read_reqs.reserve(batchSize);
                
                for (size_t b = 0; b < batchSize; ++b) {
                    size_t keyIdx = keyDist(rng);
                    read_reqs.push_back({
                        static_cast<uint32_t>(keyIdx), 
                        valueSize, 
                        buffers[b].data()
                    });
                }
                
                try {
                    ncs_reader.read(read_reqs);
                    
                    // Verify data integrity
                    bool allOk = true;
                    for (size_t i = 0; i < read_reqs.size(); ++i) {
                        uint8_t expectedPattern = static_cast<uint8_t>(read_reqs[i].key % 256);
                        if (buffers[i][0] != expectedPattern) {
                            allOk = false;
                            break;
                        }
                    }
                    
                    if (allOk) {
                        successfulOps.fetch_add(batchSize);
                    } else {
                        failedOps.fetch_add(batchSize);
                    }
                } catch (const std::exception& e) {
                    failedOps.fetch_add(batchSize);
                }
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
    
    REQUIRE(failedOps.load() == 0);
    REQUIRE(successfulOps.load() > 0);
    
    // Cleanup
    ncs->deleteBucket(bucketId);
}

// Tests DiskANN index building with NCS upload and verifies data consistency between
// FileIndexReader and NCS connector reads. Ensures uploaded data can be retrieved via connector.
// Runs with both in_memory and redis NCS backends.
TEST_CASE("Test NcsUpload Using Connector", "[NcsTest]") {
    auto ncs_kind = GENERATE("in_memory", "redis");
    const uint32_t bucketId = 1;
    
    initNcsForTest(ncs_kind);
    
    Ncs* ncs = NcsSingleton::Instance();
    auto createResult = ncs->createBucket(bucketId);
    REQUIRE(createResult == NcsStatus::OK);

    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kIPIndexDir));
    int rows_num = 10;
    auto version = GenTestVersionList();
    
    std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);

    auto base_ds = GenDataSet(rows_num, kDim, 30);
    auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
    WriteRawDataToDisk<float>(kRawDataPath, base_ptr, rows_num, kDim);

    auto ncs_extras = getNcsExtras(ncs_kind);

    knowhere::Json json;
    json["data_path"] = kRawDataPath;
    json["index_prefix"] = kL2IndexPrefix;
    json["dim"] = kDim;
    json["metric_type"] = "L2";
    json["k"] = 100;
    json["data_path"] = kRawDataPath;
    json["max_degree"] = 24;
    json["search_list_size"] = 64;
    json["pq_code_budget_gb"] = sizeof(float) * kDim * rows_num * 0.125 / (1024 * 1024 * 1024);
    json["build_dram_budget_gb"] = 32.0;
    json["search_cache_budget_gb"] = sizeof(float) * kDim * rows_num * 0.05 / (1024 * 1024 * 1024);
    json["beamwidth"] = 8;
    json["min_k"] = 10;
    json["max_k"] = 8000;
    json["ncs_enable"] = true;
    json["ncs_descriptor"] = NcsDescriptor(ncs_kind, bucketId, ncs_extras);
    
    knowhere::DataSetPtr ds_ptr = nullptr;
    auto binarySet = knowhere::BinarySet();
    auto diskann = knowhere::IndexFactory::Instance().Create<knowhere::fp32>("DISKANN", version, diskann_index_pack).value();
    diskann.Build(ds_ptr, json);
    diskann.Serialize(binarySet);

    knowhere::Status ncs_upload_res = diskann.NcsUpload(json);
    REQUIRE(knowhere::Status::success == ncs_upload_res);
    
    auto descriptor = std::make_unique<NcsDescriptor>(NcsDescriptor(ncs_kind, bucketId, ncs_extras));
    auto connector = std::unique_ptr<NcsConnector>(
        NcsConnectorFactory::Instance().createConnector(descriptor.get()));

    std::vector<uint32_t> keys;
    for (int i = 0; i < rows_num; ++i) {
        keys.push_back(i);
    }
    std::vector<SpanBytes> buffs;
    
    std::unique_ptr<IndexReader> reader = std::make_unique<FileIndexReader>(
        kL2IndexPrefix+"_disk_data.index", 
        [&diskann](uint64_t node_id) { 
            return knowhere::GetDiskANNNodeSectorOffsetForTest<knowhere::fp32>(diskann, node_id); 
        },
        [&diskann](char* sector_buf, uint64_t node_id) {
            return knowhere::GetDiskANNOffsetToNodeForTest<knowhere::fp32>(diskann, sector_buf, node_id);
        },
        knowhere::GetDiskANNReadLenForNodeForTest<knowhere::fp32>(diskann)
    );
    REQUIRE(reader != nullptr);

    // Use actual node size instead of sector-aligned size since FileIndexReader handles alignment
    uint64_t max_node_len = knowhere::GetDiskANNMaxNodeLenForTest<knowhere::fp32>(diskann);
    std::vector<std::vector<char>> file_reader_buffers(keys.size(), std::vector<char>(max_node_len));

    int i =0;
    std::vector<ReadReq> reqs;
    for(size_t key : keys){
        reqs.emplace_back(key, max_node_len, (void*)file_reader_buffers[i].data());
        i++;
    }
    reader->read(reqs);

    std::vector<std::vector<uint8_t>> ncs_reader_buffers;
    std::vector<SpanBytes> readSpans;
    for (size_t key : keys) {
        ncs_reader_buffers.emplace_back(max_node_len);
        readSpans.emplace_back(ncs_reader_buffers.back().data(), max_node_len);
    }

    auto getResults = connector->multiGet(keys, readSpans);
    REQUIRE(getResults.size() == keys.size());
    bool allOK = std::all_of(getResults.begin(), getResults.end(),
        [](milvus::NcsStatus s) {
            return s == milvus::NcsStatus::OK;
        }
    );
    REQUIRE(allOK);

    for(int i = 0 ; i < keys.size() ; i++){
        for(size_t j = 0 ; j < max_node_len ; j+=1){
            REQUIRE(static_cast<uint8_t*>(ncs_reader_buffers[i].data())[j] == static_cast<uint8_t*>(reqs[i].buf)[j]);
        }
        REQUIRE(static_cast<uint8_t*>(ncs_reader_buffers[i].data())[100] != static_cast<uint8_t*>(reqs[i].buf)[0]);

    }

    fs::remove_all(kDir);
    fs::remove(kDir);
}

// Tests that FileIndexReader and NCSReader return identical data after DiskANN NCS upload.
// Compares data read from disk index file vs data read from NCS storage to ensure consistency.
// Runs with both in_memory and redis NCS backends.
TEST_CASE("FileReader Compare NcsReader", "[NcsTest]") {
    auto ncs_kind = GENERATE("in_memory", "redis");
    const uint32_t bucketId = 2;
    
    initNcsForTest(ncs_kind);
    
    Ncs* ncs = NcsSingleton::Instance();
    auto createResult = ncs->createBucket(bucketId);
    REQUIRE(createResult == NcsStatus::OK);

    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kIPIndexDir));
    int rows_num = 10;
    auto version = GenTestVersionList();
    
    std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);

    auto base_ds = GenDataSet(rows_num, kDim, 30);
    auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
    WriteRawDataToDisk<float>(kRawDataPath, base_ptr, rows_num, kDim);

    auto ncs_extras = getNcsExtras(ncs_kind);

    knowhere::Json json;
    json["data_path"] = kRawDataPath;
    json["index_prefix"] = kL2IndexPrefix;
    json["dim"] = kDim;
    json["metric_type"] = "L2";
    json["k"] = 100;
    json["data_path"] = kRawDataPath;
    json["max_degree"] = 24;
    json["search_list_size"] = 64;
    json["pq_code_budget_gb"] = sizeof(float) * kDim * rows_num * 0.125 / (1024 * 1024 * 1024);
    json["build_dram_budget_gb"] = 32.0;
    json["search_cache_budget_gb"] = sizeof(float) * kDim * rows_num * 0.05 / (1024 * 1024 * 1024);
    json["beamwidth"] = 8;
    json["min_k"] = 10;
    json["max_k"] = 8000;
    json["ncs_enable"] = true;
    json["ncs_descriptor"] = NcsDescriptor(ncs_kind, bucketId, ncs_extras);
    
    knowhere::DataSetPtr ds_ptr = nullptr;
    auto binarySet = knowhere::BinarySet();
    auto diskann = knowhere::IndexFactory::Instance().Create<knowhere::fp32>("DISKANN", version, diskann_index_pack).value();
    diskann.Build(ds_ptr, json);
    diskann.Serialize(binarySet);

    knowhere::Status ncs_upload_res = diskann.NcsUpload(json);
    REQUIRE(knowhere::Status::success == ncs_upload_res);
    
    auto descriptor = std::make_unique<NcsDescriptor>(NcsDescriptor(ncs_kind, bucketId, ncs_extras));
    
    std::unique_ptr<IndexReader> file_reader = std::make_unique<FileIndexReader>(
        kL2IndexPrefix+"_disk_data.index", 
        [&diskann](uint64_t node_id) { 
            return knowhere::GetDiskANNNodeSectorOffsetForTest<knowhere::fp32>(diskann, node_id); 
        },
        [&diskann](char* sector_buf, uint64_t node_id) {
            return knowhere::GetDiskANNOffsetToNodeForTest<knowhere::fp32>(diskann, sector_buf, node_id);
        },
        knowhere::GetDiskANNReadLenForNodeForTest<knowhere::fp32>(diskann)

    );
    REQUIRE(file_reader != nullptr);

    std::unique_ptr<NCSReader> ncs_reader = std::make_unique<NCSReader>(descriptor.get());
    REQUIRE(ncs_reader != nullptr);

    std::vector<uint32_t> keys;
    for (int i = 0; i < rows_num; ++i) {
        keys.push_back(i);
    }

    // Use actual node size instead of sector-aligned size since FileIndexReader handles alignment
    uint64_t max_node_len = knowhere::GetDiskANNMaxNodeLenForTest<knowhere::fp32>(diskann);

    std::vector<std::vector<char>> file_buffers(keys.size(), std::vector<char>(max_node_len));
    std::vector<ReadReq> file_reqs;
    file_reqs.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        file_reqs.emplace_back(keys[i], max_node_len, file_buffers[i].data());
    }

    std::vector<std::vector<char>> ncs_buffers(keys.size(), std::vector<char>(max_node_len));
    std::vector<ReadReq> ncs_reqs;
    ncs_reqs.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        ncs_reqs.emplace_back(keys[i], max_node_len, ncs_buffers[i].data());
    }

    file_reader->read(file_reqs);
    ncs_reader->read(ncs_reqs);

    for(size_t i = 0; i < keys.size(); ++i) {
        uint8_t* file_data = static_cast<uint8_t*>(file_reqs[i].buf);
        uint8_t* ncs_data = static_cast<uint8_t*>(ncs_reqs[i].buf);
        for(size_t j = 0; j < max_node_len; ++j) {
            REQUIRE(file_data[j] == ncs_data[j]);
        }
    }
    
    fs::remove_all(kDir);
    fs::remove(kDir);
}


///////  

// Tests DiskANN parameter validation and dynamic budget calculation for various PQ ratio
// configurations. Verifies that budget values are properly calculated and constraints are enforced.
TEST_CASE("Valid diskann build params test", "[diskann]") {
    int rows_num = 1000000;
    auto version = GenTestVersionList();

    auto ratio = GENERATE(as<float>{}, 0.01, 0.1, 0.125);

    float pq_code_budget_gb = sizeof(float) * kDim * rows_num * 0.125 / (1024 * 1024 * 1024);
    float search_cache_budget_gb = sizeof(float) * kDim * rows_num * 0.05 / (1024 * 1024 * 1024);

    auto test_gen = [&]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = "L2";
        json["k"] = 100;
        json["index_prefix"] = kL2IndexPrefix;
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 24;
        json["search_list_size"] = 64;
        json["vec_field_size_gb"] = 1.0;
        json["pq_code_budget_gb_ratio"] = ratio;
        json["pq_code_budget_gb"] = pq_code_budget_gb;
        json["build_dram_budget_gb"] = 32.0;
        json["search_cache_budget_gb_ratio"] = ratio;
        json["search_cache_budget_gb"] = search_cache_budget_gb;
        json["beamwidth"] = 8;
        json["min_k"] = 10;
        json["max_k"] = 8000;
        return json;
    };

    SECTION("Dynamic param check") {
        knowhere::Json test_json = test_gen();

        auto cfg = knowhere::IndexStaticFaced<float>::CreateConfig(knowhere::IndexEnum::INDEX_DISKANN, version);
        knowhere::Json json_(test_json);
        std::string msg;
        auto res = knowhere::Config::FormatAndCheck(*cfg, json_, &msg);
        REQUIRE(res == knowhere::Status::success);
        res = knowhere::Config::Load(*cfg, json_, knowhere::PARAM_TYPE::TRAIN, &msg);
        REQUIRE(res == knowhere::Status::success);

        knowhere::DiskANNConfig diskCfg = static_cast<const knowhere::DiskANNConfig&>(*cfg);
        REQUIRE(diskCfg.pq_code_budget_gb == std::max(pq_code_budget_gb, 1.0f * ratio));
        REQUIRE(diskCfg.search_cache_budget_gb == std::max(search_cache_budget_gb, 1.0f * ratio));
    }
}

// Tests error handling for invalid DiskANN build and search parameters including invalid metrics,
// missing data files, and out-of-range parameter values. Verifies appropriate error codes are returned.
TEST_CASE("Invalid diskann params test", "[diskann]") {
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kIPIndexDir));
    int rows_num = 10;
    auto version = GenTestVersionList();
    auto test_gen = [rows_num]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = "L2";
        json["k"] = 100;
        json["index_prefix"] = kL2IndexPrefix;
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 24;
        json["search_list_size"] = 64;
        json["pq_code_budget_gb"] = sizeof(float) * kDim * rows_num * 0.125 / (1024 * 1024 * 1024);
        json["build_dram_budget_gb"] = 32.0;
        json["search_cache_budget_gb"] = sizeof(float) * kDim * rows_num * 0.05 / (1024 * 1024 * 1024);
        json["beamwidth"] = 8;
        json["min_k"] = 10;
        json["max_k"] = 8000;
        return json;
    };
    std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);
    auto base_ds = GenDataSet(rows_num, kDim, 30);
    auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
    WriteRawDataToDisk<float>(kRawDataPath, base_ptr, rows_num, kDim);
    // build process
    SECTION("Invalid build params test") {
        knowhere::DataSetPtr ds_ptr = nullptr;
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>("DISKANN", version, diskann_index_pack).value();
        knowhere::Json test_json;
        knowhere::Status test_stat;
        // invalid metric type
        test_json = test_gen();
        test_json["metric_type"] = knowhere::metric::JACCARD;
        test_stat = diskann.Build(ds_ptr, test_json);
        REQUIRE(test_stat == knowhere::Status::invalid_metric_type);
        // raw data path not exist
        test_json = test_gen();
        test_json["data_path"] = kL2IndexPrefix + ".temp";
        test_stat = diskann.Build(ds_ptr, test_json);
        REQUIRE(test_stat == knowhere::Status::disk_file_error);
    }

    SECTION("Invalid search params test") {
        knowhere::DataSetPtr ds_ptr = nullptr;
        auto binarySet = knowhere::BinarySet();
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>("DISKANN", version, diskann_index_pack).value();
        diskann.Build(ds_ptr, test_gen());
        diskann.Serialize(binarySet);
        diskann.Deserialize(binarySet, test_gen());

        knowhere::Json test_json;
        auto query_ds = GenDataSet(kNumQueries, kDim, 42);

#ifndef KNOWHERE_WITH_CARDINAL
        // search list size < topk
        {
            test_json = test_gen();
            test_json["search_list_size"] = 1;
            auto res = diskann.Search(query_ds, test_json, nullptr);
            REQUIRE_FALSE(res.has_value());
            REQUIRE(res.error() == knowhere::Status::out_of_range_in_json);
        }
#endif
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}

template <typename DataType>
inline void
base_search() {
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kIPIndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kCOSINEIndexDir));

    auto metric_str = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP, knowhere::metric::COSINE);
    auto version = GenTestVersionList();

    std::unordered_map<knowhere::MetricType, std::string> metric_dir_map = {
        {knowhere::metric::L2, kL2IndexPrefix},
        {knowhere::metric::IP, kIPIndexPrefix},
        {knowhere::metric::COSINE, kCOSINEIndexPrefix},
    };
    std::unordered_map<knowhere::MetricType, float> metric_range_ap_map = {
        {knowhere::metric::L2, kL2RangeAp},
        {knowhere::metric::IP, kIpRangeAp},
        {knowhere::metric::COSINE, kCosineRangeAp},
    };

    auto base_gen = [&metric_str]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = metric_str;
        json["k"] = kK;
        if (metric_str == knowhere::metric::L2) {
            json["radius"] = CFG_FLOAT::value_type(200000);
            json["range_filter"] = CFG_FLOAT::value_type(0);
        } else if (metric_str == knowhere::metric::IP) {
            json["radius"] = CFG_FLOAT::value_type(350000);
            json["range_filter"] = std::numeric_limits<CFG_FLOAT::value_type>::max();
        } else {
            json["radius"] = 0.75f;
            json["range_filter"] = 1.0f;
        }
        return json;
    };

    auto build_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 56;
        json["search_list_size"] = 128;
        json["pq_code_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["build_dram_budget_gb"] = 32.0;
        return json;
    };

    auto deserialize_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        return json;
    };

    auto knn_search_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["search_list_size"] = 36;
        json["beamwidth"] = 8;
        return json;
    };

    auto range_search_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["beamwidth"] = 8;
        return json;
    };

    auto fp32_query_ds = GenDataSet(kNumQueries, kDim, 42);
    knowhere::DataSetPtr knn_gt_ptr = nullptr;
    knowhere::DataSetPtr range_search_gt_ptr = nullptr;
    auto fp32_base_ds = GenDataSet(kNumRows, kDim, 30);

    auto base_ds = knowhere::ConvertToDataTypeIfNeeded<DataType>(fp32_base_ds);
    auto query_ds = knowhere::ConvertToDataTypeIfNeeded<DataType>(fp32_query_ds);

    {
        auto base_ptr = static_cast<const DataType*>(base_ds->GetTensor());
        WriteRawDataToDisk<DataType>(kRawDataPath, base_ptr, kNumRows, kDim);

        // generate the gt of knn search and range search
        auto base_json = base_gen();
        auto result_knn = knowhere::BruteForce::Search<DataType>(base_ds, query_ds, base_json, nullptr);
        knn_gt_ptr = result_knn.value();
        auto result_range = knowhere::BruteForce::RangeSearch<DataType>(base_ds, query_ds, base_json, nullptr);
        range_search_gt_ptr = result_range.value();
    }

    SECTION("Test search and range search") {
        std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
        auto diskann_index_pack = knowhere::Pack(file_manager);
        knowhere::Json deserialize_json = knowhere::Json::parse(deserialize_gen().dump());
        knowhere::BinarySet binset;

        auto build_json = build_gen().dump();
        knowhere::Json json = knowhere::Json::parse(build_json);
        // build process
        {
            knowhere::DataSetPtr ds_ptr = nullptr;
            auto diskann =
                knowhere::IndexFactory::Instance().Create<DataType>("DISKANN", version, diskann_index_pack).value();
            diskann.Build(ds_ptr, json);
            diskann.Serialize(binset);
        }
        {
            // knn search
            auto diskann =
                knowhere::IndexFactory::Instance().Create<DataType>("DISKANN", version, diskann_index_pack).value();
            diskann.Deserialize(binset, deserialize_json);
            REQUIRE(diskann.HasRawData(metric_str) ==
                    knowhere::IndexStaticFaced<DataType>::HasRawData("DISKANN", version, json));

            auto knn_search_json = knn_search_gen().dump();
            knowhere::Json knn_json = knowhere::Json::parse(knn_search_json);
            auto res = diskann.Search(query_ds, knn_json, nullptr);
            REQUIRE(res.has_value());
            auto knn_recall = GetKNNRecall(*knn_gt_ptr, *res.value());
            CAPTURE(knn_json.dump());
            REQUIRE(knn_recall > kKnnRecall);
            // knn search without cache file
            {
                std::string cached_nodes_file_path =
                    std::string(build_gen()["index_prefix"]) + std::string("_cached_nodes.bin");
                if (fs::exists(cached_nodes_file_path)) {
                    fs::remove(cached_nodes_file_path);
                }
                auto diskann_tmp =
                    knowhere::IndexFactory::Instance().Create<DataType>("DISKANN", version, diskann_index_pack).value();
                diskann_tmp.Deserialize(binset, deserialize_json);
                auto knn_search_json = knn_search_gen().dump();
                knowhere::Json knn_json = knowhere::Json::parse(knn_search_json);
                auto res = diskann_tmp.Search(query_ds, knn_json, nullptr);
                REQUIRE(res.has_value());
                REQUIRE(GetKNNRecall(*knn_gt_ptr, *res.value()) >= kKnnRecall);
            }
            // knn search with bitset
            std::vector<std::function<std::vector<uint8_t>(size_t, size_t)>> gen_bitset_funcs = {
                GenerateBitsetWithFirstTbitsSet, GenerateBitsetWithRandomTbitsSet};
            const auto bitset_percentages = {0.4f, 0.98f};
            const auto bitset_thresholds = {-1.0f, 0.9f};
            for (const float threshold : bitset_thresholds) {
                knn_json["filter_threshold"] = threshold;
                for (const float percentage : bitset_percentages) {
                    for (const auto& gen_func : gen_bitset_funcs) {
                        auto bitset_data = gen_func(kNumRows, percentage * kNumRows);
                        knowhere::BitsetView bitset(bitset_data.data(), kNumRows);
                        auto results = diskann.Search(query_ds, knn_json, bitset);
                        auto gt = knowhere::BruteForce::Search<DataType>(base_ds, query_ds, knn_json, bitset);
                        float recall = GetKNNRecall(*gt.value(), *results.value());
                        if (percentage == 0.98f) {
                            REQUIRE(recall >= 0.9f);
                        } else {
                            REQUIRE(recall >= kKnnRecall);
                        }
                    }
                }
            }

            // range search process
            auto range_search_json = range_search_gen().dump();
            knowhere::Json range_json = knowhere::Json::parse(range_search_json);
            auto range_search_res = diskann.RangeSearch(query_ds, range_json, nullptr);
            REQUIRE(range_search_res.has_value());
            auto ap = GetRangeSearchRecall(*range_search_gt_ptr, *range_search_res.value());
            float standard_ap = metric_range_ap_map[metric_str];
            REQUIRE(ap > standard_ap);
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}

// Tests standard DiskANN KNN search, range search, search with bitsets, and cache handling.
// Verifies search accuracy against brute-force ground truth for multiple metric types (L2, IP, COSINE).
// Tests include cache file presence/absence and various bitset filtering scenarios.
TEST_CASE("Test DiskANNIndexNode.", "[diskann]") {
    base_search<knowhere::fp32>();
}

template <typename DataType>
inline void
emb_list_search() {
    auto version = GenTestEmbListVersionList();

    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kEmbListL2IndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kEmbListIPIndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kEmbListCOSINEIndexDir));

    auto metric_str = GENERATE(as<std::string>{}, knowhere::metric::MAX_SIM_COSINE, knowhere::metric::MAX_SIM_IP,
                               knowhere::metric::MAX_SIM_L2);

    std::unordered_map<knowhere::MetricType, std::string> metric_dir_map = {
        {knowhere::metric::MAX_SIM_L2, kEmbListL2IndexPrefix},
        {knowhere::metric::MAX_SIM_IP, kEmbListIPIndexPrefix},
        {knowhere::metric::MAX_SIM_COSINE, kEmbListCOSINEIndexPrefix},
    };

    auto base_gen = [&metric_str]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = metric_str;
        json["k"] = kK;
        return json;
    };

    auto build_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["data_path"] = kRawDataPath;
        json["emb_list_offset_file_path"] = kEmbListOffsetPath;
        json["max_degree"] = 56;
        json["search_list_size"] = 128;
        json["pq_code_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["build_dram_budget_gb"] = 32.0;
        return json;
    };

    auto deserialize_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        return json;
    };

    auto knn_search_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["search_list_size"] = 36;
        json["beamwidth"] = 8;
        json["retrieval_ann_ratio"] = 3.0f;
        return json;
    };

    int each_el_len = 10;
    int num_el = int((kNumRows + each_el_len - 1) / each_el_len);
    auto fp32_query_ds = GenQueryEmbListDataSet(kNumQueries, kDim, 42);
    knowhere::DataSetPtr knn_gt_ptr = nullptr;
    auto fp32_base_ds = GenEmbListDataSet(kNumRows, kDim, 42, each_el_len);
    auto emb_list_offset = GenEmbListOffset(kNumRows, each_el_len);

    auto base_ds = knowhere::ConvertToDataTypeIfNeeded<DataType>(fp32_base_ds);
    auto query_ds = knowhere::ConvertToDataTypeIfNeeded<DataType>(fp32_query_ds);

    {
        auto base_ptr = static_cast<const DataType*>(base_ds->GetTensor());
        WriteRawDataToDisk<DataType>(kRawDataPath, base_ptr, kNumRows, kDim);
        WriteEmbListOffsetToDisk(kEmbListOffsetPath, emb_list_offset.data(), emb_list_offset.size());
        // generate the gt of knn search
        auto base_json = base_gen();
        auto result_knn = knowhere::BruteForce::Search<DataType>(base_ds, query_ds, base_json, nullptr);
        knn_gt_ptr = result_knn.value();
    }

    SECTION("Test EmbList knn search") {
        std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
        auto diskann_index_pack = knowhere::Pack(file_manager);
        knowhere::Json deserialize_json = knowhere::Json::parse(deserialize_gen().dump());
        knowhere::BinarySet binset;

        auto build_json = build_gen().dump();
        knowhere::Json json = knowhere::Json::parse(build_json);
        // build process
        {
            auto diskann =
                knowhere::IndexFactory::Instance().Create<DataType>("DISKANN", version, diskann_index_pack).value();
            diskann.Build(nullptr, json);
            diskann.Serialize(binset);
        }
        {
            // knn search
            auto diskann =
                knowhere::IndexFactory::Instance().Create<DataType>("DISKANN", version, diskann_index_pack).value();
            diskann.Deserialize(binset, deserialize_json);
            REQUIRE(diskann.HasRawData(metric_str) ==
                    knowhere::IndexStaticFaced<DataType>::HasRawData("DISKANN", version, json));

            auto knn_search_json = knn_search_gen().dump();
            knowhere::Json knn_json = knowhere::Json::parse(knn_search_json);
            auto res = diskann.Search(query_ds, knn_json, nullptr);
            REQUIRE(res.has_value());
            auto knn_recall = GetKNNRecall(*knn_gt_ptr, *res.value());
            CAPTURE(knn_json.dump());
            REQUIRE(knn_recall > kEmbListKnnRecall);
            // knn search without cache file
            {
                std::string cached_nodes_file_path =
                    std::string(build_gen()["index_prefix"]) + std::string("_cached_nodes.bin");
                if (fs::exists(cached_nodes_file_path)) {
                    fs::remove(cached_nodes_file_path);
                }
                auto diskann_tmp =
                    knowhere::IndexFactory::Instance().Create<DataType>("DISKANN", version, diskann_index_pack).value();
                diskann_tmp.Deserialize(binset, deserialize_json);
                auto knn_search_json = knn_search_gen().dump();
                knowhere::Json knn_json = knowhere::Json::parse(knn_search_json);
                auto res = diskann_tmp.Search(query_ds, knn_json, nullptr);
                REQUIRE(res.has_value());
                REQUIRE(GetKNNRecall(*knn_gt_ptr, *res.value()) >= kEmbListKnnRecall);
            }
            // knn search with bitset
            std::vector<std::function<std::vector<uint8_t>(size_t, float, size_t)>> gen_bitset_funcs = {
                GenerateBitsetByPartition};
            const auto bitset_percentages = {0.1f, 0.5f, 0.9f, 0.98f};
            for (const float percentage : bitset_percentages) {
                for (const auto& gen_func : gen_bitset_funcs) {
                    auto bitset_data = gen_func(num_el, 1 - percentage, 1);
                    knowhere::BitsetView bitset(bitset_data.data(), num_el);
                    auto results = diskann.Search(query_ds, knn_json, bitset);
                    auto gt = knowhere::BruteForce::Search<DataType>(base_ds, query_ds, knn_json, bitset);
                    float recall = GetKNNRecall(*gt.value(), *results.value());
                    REQUIRE(recall >= kEmbListKnnRecall);
                }
            }
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}

// Tests DiskANN KNN search with embedding list format data using special metric types (MAX_SIM variants).
// Handles variable-length embedding lists and verifies search accuracy with bitset filtering.
TEST_CASE("Test DISKANN for EmbList", "[diskann]") {
    emb_list_search<knowhere::fp32>();
}

// This test case only check L2
// Tests DiskANN GetVectorByIds operation for retrieving vectors by ID. Tests with various cache
// sizes and batch retrieval sizes. Verifies data integrity matches original dataset.
// Runs with both standard (128) and large (256) dimensions.
TEST_CASE("Test DiskANN GetVectorByIds", "[diskann]") {
    auto version = GenTestVersionList();
    for (const uint32_t dim : {kDim, kLargeDim}) {
        fs::remove_all(kDir);
        fs::remove(kDir);
        REQUIRE_NOTHROW(fs::create_directories(kL2IndexDir));

        auto base_gen = [=] {
            knowhere::Json json;
            json["dim"] = dim;
            json["metric_type"] = knowhere::metric::L2;
            json["k"] = kK;
            return json;
        };

        auto build_gen = [=]() {
            knowhere::Json json = base_gen();
            json["index_prefix"] = kL2IndexPrefix;
            json["data_path"] = kRawDataPath;
            json["max_degree"] = 5;
            json["search_list_size"] = kK;
            json["pq_code_budget_gb"] = sizeof(float) * dim * kNumRows * 0.03125 / (1024 * 1024 * 1024);
            json["build_dram_budget_gb"] = 32.0;
            return json;
        };

        auto query_ds = GenDataSet(kNumQueries, dim, 42);
        auto base_ds = GenDataSet(kNumRows, dim, 30);
        auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
        WriteRawDataToDisk<float>(kRawDataPath, base_ptr, kNumRows, dim);

        std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
        auto diskann_index_pack = knowhere::Pack(file_manager);

        knowhere::DataSetPtr ds_ptr = nullptr;
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>("DISKANN", version, diskann_index_pack).value();
        auto build_json = build_gen().dump();
        knowhere::Json json = knowhere::Json::parse(build_json);
        diskann.Build(ds_ptr, json);
        knowhere::BinarySet binset;
        diskann.Serialize(binset);
        {
            std::vector<double> cache_sizes = {0, 1.0f * sizeof(float) * dim * kNumRows * 0.125 / (1024 * 1024 * 1024)};
            for (const auto cache_size : cache_sizes) {
                auto deserialize_gen = [&base_gen, cache = cache_size]() {
                    knowhere::Json json = base_gen();
                    json["index_prefix"] = kL2IndexPrefix;
                    json["search_cache_budget_gb"] = cache;
                    return json;
                };
                knowhere::Json deserialize_json = knowhere::Json::parse(deserialize_gen().dump());
                auto index = knowhere::IndexFactory::Instance()
                                 .Create<knowhere::fp32>("DISKANN", version, diskann_index_pack)
                                 .value();
                auto ret = index.Deserialize(binset, deserialize_json);
                REQUIRE(ret == knowhere::Status::success);

                REQUIRE(diskann.HasRawData(knowhere::metric::L2) ==
                        knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData("DISKANN", version, json));

                if (!diskann.HasRawData(knowhere::metric::L2)) {
                    continue;
                }

                std::vector<double> ids_sizes = {1, kNumRows * 0.2, kNumRows * 0.7, kNumRows};
                for (const auto ids_size : ids_sizes) {
                    LOG_KNOWHERE_DEBUG_ << "Testing dim = " << dim << ", cache_size = " << cache_size
                                        << ", ids_size = " << ids_size;
                    auto ids_ds = GenIdsDataSet(ids_size, ids_size);
                    auto results = index.GetVectorByIds(ids_ds);
                    REQUIRE(results.has_value());
                    auto xb = (float*)base_ds->GetTensor();
                    auto data = (float*)results.value()->GetTensor();
                    for (size_t i = 0; i < ids_size; ++i) {
                        auto id = ids_ds->GetIds()[i];
                        for (size_t j = 0; j < dim; ++j) {
                            INFO("Checking vector at i " << i << ", j " << j << ", dim " << dim << ", id " << id << ", data: " << data[i * dim + j] << ", xb: " << xb[id * dim + j]);
                            REQUIRE(data[i * dim + j] == xb[id * dim + j]);
                        }
                    }
                }
            }
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}

// Tests AiSAQ search with dynamic PQ read page cache sizes, measuring search performance at various
// cache levels. Helps understand cache impact on query latency for different metric types.
TEST_CASE("Test_AiSAQ_dynamic_cache", "[diskann]") {
    std::string index_type = "AISAQ";
    constexpr uint32_t kNumRowsTest = 10000;
    constexpr uint32_t kNumQueriesTest = 10;
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kIPIndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kCOSINEIndexDir));
    auto metric_str = GENERATE(as<std::string>{}, knowhere::metric::COSINE, knowhere::metric::IP, knowhere::metric::L2);
    auto version = GenTestVersionList();

    std::unordered_map<knowhere::MetricType, std::string> metric_dir_map = {
        {knowhere::metric::L2, kL2IndexPrefix},
        {knowhere::metric::IP, kIPIndexPrefix},
        {knowhere::metric::COSINE, kCOSINEIndexPrefix},
    };
    std::unordered_map<knowhere::MetricType, float> metric_range_ap_map = {
        {knowhere::metric::L2, kL2RangeAp},
        {knowhere::metric::IP, kIpRangeAp},
        {knowhere::metric::COSINE, kCosineRangeAp},
    };

    auto base_gen = [&metric_str]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = metric_str;
        json["k"] = kK;
        if (metric_str == knowhere::metric::L2) {
            json["radius"] = CFG_FLOAT::value_type(200000);
            json["range_filter"] = CFG_FLOAT::value_type(0);
        } else if (metric_str == knowhere::metric::IP) {
            json["radius"] = CFG_FLOAT::value_type(350000);
            json["range_filter"] = std::numeric_limits<CFG_FLOAT::value_type>::max();
        } else {
            json["radius"] = 0.75f;
            json["range_filter"] = 1.0f;
        }
        return json;
    };

    auto build_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 64;
        json["search_list_size"] = 128;
        json["pq_code_budget_gb"] = sizeof(float) * kDim * kNumRowsTest * 0.125 * 0.5 / (1024 * 1024 * 1024);
        json["search_cache_budget_gb"] = 0;
        json["build_dram_budget_gb"] = 16.0;
        json["rearrange"] = true;
        json["inline_pq"] = 0;
        json["pq_cache_size"] = 0;
        json["num_entry_points"] = 100;
        return json;
    };

    auto deserialize_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["search_cache_budget_gb"] = 0;
        json["pq_cache_size"] = 0;
        return json;
    };

    auto knn_search_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["search_list_size"] = 36;
        json["beamwidth"] = 4;
        json["vectors_beamwidth"] = 4;
        return json;
    };

    std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);
    auto fp32_base_ds = GenDataSet(kNumRowsTest, kLargeDim, 30);
    auto fp32_query_ds = GenDataSet(kNumQueriesTest, kLargeDim, 42);

    knowhere::BinarySet binset;
    auto build_json = build_gen().dump();
    knowhere::Json json = knowhere::Json::parse(build_json);
    knowhere::Json deserialize_json = knowhere::Json::parse(deserialize_gen().dump());
    auto base_ds = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp32>(fp32_base_ds);
    auto query_ds = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp32>(fp32_query_ds);
    knowhere::DataSetPtr knn_gt_ptr = nullptr;
    knowhere::DataSetPtr range_search_gt_ptr = nullptr;

    {
        auto base_ptr = static_cast<const knowhere::fp32*>(base_ds->GetTensor());
        WriteRawDataToDisk<knowhere::fp32>(kRawDataPath, base_ptr, kNumRowsTest, kDim);

        // generate the gt of knn search and range search
        auto base_json = base_gen();

        auto result_knn = knowhere::BruteForce::Search<knowhere::fp32>(base_ds, query_ds, base_json, nullptr);
        knn_gt_ptr = result_knn.value();
        auto result_range = knowhere::BruteForce::RangeSearch<knowhere::fp32>(base_ds, query_ds, base_json, nullptr);
        range_search_gt_ptr = result_range.value();
    }
    // build process
    {
        knowhere::DataSetPtr ds_ptr = nullptr;
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version, diskann_index_pack).value();

        diskann.Build(ds_ptr, json);
        diskann.Serialize(binset);
    }
    {
        auto knn_search_json = knn_search_gen().dump();
        knowhere::Json knn_json = knowhere::Json::parse(knn_search_json);
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version, diskann_index_pack).value();
        diskann.Deserialize(binset, deserialize_json);

        knn_json["pq_read_page_cache_size"] = 0;
        auto start = std::chrono::high_resolution_clock::now();
        auto results = diskann.Search(query_ds, knn_json, nullptr);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "*********************** First Run time: " << duration.count() << " ms"
                  << " read_page_cache_size: 0 KiB" << std::endl;
        uint32_t read_page_cache_size_limit = /*diskann::defaults::MAX_PQ_READ_PAGE_CACHE_SIZE*/ 204800;
        for (uint32_t read_page_cache_size = 0; read_page_cache_size < read_page_cache_size_limit;
             read_page_cache_size += 4096) {
            knn_json["pq_read_page_cache_size"] = read_page_cache_size;
            auto start = std::chrono::high_resolution_clock::now();
            diskann.Search(query_ds, knn_json, nullptr);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "*********************** Run time: " << duration.count() << " ms"
                      << " read_page_cache_size: " << read_page_cache_size / 1024 << " KiB" << std::endl;
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}
// This test case only check L2
// Tests AiSAQ GetVectorByIds operation for retrieving vectors by ID. Identical test logic to
// DiskANN GetVectorByIds but using AiSAQ index type. Verifies data integrity with various cache configurations.
// Runs with both standard (128) and large (256) dimensions.
TEST_CASE("Test_AiSAQ_GetVectorByIds", "[diskann]") {
    auto version = GenTestVersionList();
    for (const uint32_t dim : {kDim, kLargeDim}) {
        fs::remove_all(kDir);
        fs::remove(kDir);
        REQUIRE_NOTHROW(fs::create_directories(kL2IndexDir));

        auto base_gen = [&] {
            knowhere::Json json;
            json[knowhere::meta::RETRIEVE_FRIENDLY] = true;
            json["dim"] = dim;
            json["metric_type"] = knowhere::metric::L2;
            json["k"] = kK;
            return json;
        };

        auto build_gen = [&]() {
            knowhere::Json json = base_gen();
            json["index_prefix"] = kL2IndexPrefix;
            json["data_path"] = kRawDataPath;
            json["max_degree"] = 5;
            json["search_list_size"] = kK;
            json["pq_code_budget_gb"] = sizeof(float) * dim * kNumRows * 0.03125 / (1024 * 1024 * 1024);
            json["build_dram_budget_gb"] = 32.0;
            json["rearrange"] = false;
            return json;
        };

        auto query_ds = GenDataSet(kNumQueries, dim, 42);
        auto base_ds = GenDataSet(kNumRows, dim, 30);
        auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
        WriteRawDataToDisk<float>(kRawDataPath, base_ptr, kNumRows, dim);

        std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
        auto diskann_index_pack = knowhere::Pack(file_manager);

        knowhere::DataSetPtr ds_ptr = nullptr;
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>("AISAQ", version, diskann_index_pack).value();
        auto build_json = build_gen().dump();
        knowhere::Json json = knowhere::Json::parse(build_json);
        diskann.Build(ds_ptr, json);
        knowhere::BinarySet binset;
        diskann.Serialize(binset);
        {
            std::vector<double> cache_sizes = {0};
            for (const auto cache_size : cache_sizes) {
                auto deserialize_gen = [&base_gen, cache = cache_size]() {
                    knowhere::Json json = base_gen();
                    json["index_prefix"] = kL2IndexPrefix;
                    json["search_cache_budget_gb"] = cache;
                    json["pq_cache_size"] = 1024;
                    json["pq_read_page_cache_size"] = 0;
                    return json;
                };
                knowhere::Json deserialize_json = knowhere::Json::parse(deserialize_gen().dump());
                auto index = knowhere::IndexFactory::Instance()
                                 .Create<knowhere::fp32>("AISAQ", version, diskann_index_pack)
                                 .value();
                auto ret = index.Deserialize(binset, deserialize_json);
                REQUIRE(ret == knowhere::Status::success);

                REQUIRE(diskann.HasRawData(knowhere::metric::L2) ==
                        knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData("AISAQ", version, json));

                std::vector<double> ids_sizes = {1, kNumRows * 0.2, kNumRows * 0.7, kNumRows};
                for (const auto ids_size : ids_sizes) {
                    LOG_KNOWHERE_DEBUG_ << "Testing dim = " << dim << ", cache_size = " << cache_size
                                        << ", ids_size = " << ids_size;
                    auto ids_ds = GenIdsDataSet(ids_size, ids_size);
                    auto results = index.GetVectorByIds(ids_ds);
                    REQUIRE(results.has_value());
                    auto xb = (float*)base_ds->GetTensor();
                    auto data = (float*)results.value()->GetTensor();
                    for (size_t i = 0; i < ids_size; ++i) {
                        auto id = ids_ds->GetIds()[i];
                        for (size_t j = 0; j < dim; ++j) {
                            REQUIRE(data[i * dim + j] == xb[id * dim + j]);
                        }
                    }
                }
            }
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}

template <typename DataType>
inline void
base_AiSAQ_param_test() {
    std::string index_type = "AISAQ";
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kIPIndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kCOSINEIndexDir));

    auto metric_str = knowhere::metric::L2;
    auto version = GenTestVersionList();

    std::unordered_map<knowhere::MetricType, std::string> metric_dir_map = {
        {knowhere::metric::L2, kL2IndexPrefix},
        {knowhere::metric::IP, kIPIndexPrefix},
        {knowhere::metric::COSINE, kCOSINEIndexPrefix},
    };
    std::unordered_map<knowhere::MetricType, float> metric_range_ap_map = {
        {knowhere::metric::L2, kL2RangeAp},
        {knowhere::metric::IP, kIpRangeAp},
        {knowhere::metric::COSINE, kCosineRangeAp},
    };

    auto base_gen = [&metric_str]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = metric_str;
        json["k"] = kK;
        if (metric_str == knowhere::metric::L2) {
            json["radius"] = CFG_FLOAT::value_type(200000);
            json["range_filter"] = CFG_FLOAT::value_type(0);
        } else if (metric_str == knowhere::metric::IP) {
            json["radius"] = CFG_FLOAT::value_type(350000);
            json["range_filter"] = std::numeric_limits<CFG_FLOAT::value_type>::max();
        } else {
            json["radius"] = 0.75f;
            json["range_filter"] = 1.0f;
        }
        return json;
    };

    auto build_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 64;
        json["search_list_size"] = 128;
        json["pq_code_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 * 0.5 / (1024 * 1024 * 1024);
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["build_dram_budget_gb"] = 16.0;
        return json;
    };

    auto deserialize_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        return json;
    };

    auto knn_search_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["search_list_size"] = 36;
        json["beamwidth"] = 4;
        json["vectors_beamwidth"] = 4;
        return json;
    };

    auto fp32_query_ds = GenDataSet(kNumQueries, kDim, 42);
    knowhere::DataSetPtr knn_gt_ptr = nullptr;
    knowhere::DataSetPtr range_search_gt_ptr = nullptr;
    auto fp32_base_ds = GenDataSet(kNumRows, kDim, 30);

    auto base_ds = knowhere::ConvertToDataTypeIfNeeded<DataType>(fp32_base_ds);
    auto query_ds = knowhere::ConvertToDataTypeIfNeeded<DataType>(fp32_query_ds);

    {
        auto base_ptr = static_cast<const DataType*>(base_ds->GetTensor());
        WriteRawDataToDisk<DataType>(kRawDataPath, base_ptr, kNumRows, kDim);

        // generate the gt of knn search and range search
        auto base_json = base_gen();

        auto result_knn = knowhere::BruteForce::Search<DataType>(base_ds, query_ds, base_json, nullptr);
        knn_gt_ptr = result_knn.value();
        auto result_range = knowhere::BruteForce::RangeSearch<DataType>(base_ds, query_ds, base_json, nullptr);
        range_search_gt_ptr = result_range.value();
    }

    SECTION("Test search and range search") {
        std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
        auto diskann_index_pack = knowhere::Pack(file_manager);
        knowhere::Json deserialize_json = knowhere::Json::parse(deserialize_gen().dump());
        knowhere::BinarySet binset;
        auto build_json = build_gen().dump();
        knowhere::Json json = knowhere::Json::parse(build_json);
        LOG_KNOWHERE_DEBUG_ << "========== build_json =================";
        LOG_KNOWHERE_DEBUG_ << json.dump();
        LOG_KNOWHERE_DEBUG_ << "===========================";
        LOG_KNOWHERE_DEBUG_ << "========== deserialize_json =================";
        LOG_KNOWHERE_DEBUG_ << deserialize_json.dump();
        LOG_KNOWHERE_DEBUG_ << "===========================";
        knowhere::Status test_stat;
        knowhere::Json test_json;
        // build process
        {
            knowhere::DataSetPtr ds_ptr = nullptr;
            auto diskann =
                knowhere::IndexFactory::Instance().Create<DataType>(index_type, version, diskann_index_pack).value();
            LOG_KNOWHERE_INFO_ << "Test invalid metric_type parameter";
            test_json = build_gen();
            test_json["metric_type"] = knowhere::metric::JACCARD;
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::invalid_metric_type);
            // raw data path not exist
            LOG_KNOWHERE_INFO_ << "Test data path not exist";
            test_json = build_gen();
            test_json["data_path"] = kL2IndexPrefix + ".temp";
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::disk_file_error);
            LOG_KNOWHERE_INFO_ << "Test pq_cache_size parameter value more than maximum";
            test_json = build_gen();
            test_json["pq_cache_size"] = diskann::defaults::MAX_PQ_CACHE_SIZE + 1;
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::out_of_range_in_json);
            LOG_KNOWHERE_INFO_ << "Test pq_cache_size parameter value less than minimum";
            test_json = build_gen();
            test_json["pq_cache_size"] = -1;
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::out_of_range_in_json);
            LOG_KNOWHERE_INFO_ << "Test num_entry_points parameter value more than maximum";
            test_json = build_gen();
            test_json["num_entry_points"] = diskann::defaults::MAX_NUM_ENTRY_POINTS + 1;
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::out_of_range_in_json);
            LOG_KNOWHERE_INFO_ << "Test num_entry_points parameter value less than minimum";
            test_json = build_gen();
            test_json["num_entry_points"] = -1;
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::out_of_range_in_json);
            LOG_KNOWHERE_INFO_ << "Test inline_pq parameter value less than minimum";
            test_json = build_gen();
            test_json["inline_pq"] = -2;
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::out_of_range_in_json);
            LOG_KNOWHERE_INFO_ << "Test disk_pq_dims parameter value less than minimum";
            test_json = build_gen();
            test_json["disk_pq_dims"] = -2;
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::aisaq_error);
            LOG_KNOWHERE_INFO_ << "Test disk_pq_dims parameter value more than maximum";
            test_json = build_gen();
            test_json["disk_pq_dims"] = kDim + 2;
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::aisaq_error);
            LOG_KNOWHERE_INFO_ << "Test inline_pq parameter value more than max_degree";
            test_json = build_gen();
            test_json["inline_pq"] = 68;
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::aisaq_error);
            LOG_KNOWHERE_INFO_ << "Test max_degree parameter value more than maximum";
            test_json = build_gen();
            test_json["max_degree"] = diskann::defaults::MAX_AISAQ_MAX_DEGREE + 1;
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::aisaq_error);

            diskann.Build(ds_ptr, json);
            diskann.Serialize(binset);
        }
        {
            // knn search
            auto diskann =
                knowhere::IndexFactory::Instance().Create<DataType>(index_type, version, diskann_index_pack).value();
            diskann.Deserialize(binset, deserialize_json);

            REQUIRE(diskann.HasRawData(metric_str) ==
                    knowhere::IndexStaticFaced<DataType>::HasRawData(index_type, version, json));

            LOG_KNOWHERE_INFO_ << "Test vectors_beamwidth parameter value less than minimum";
            test_json = knn_search_gen();
            test_json["vectors_beamwidth"] = 0;
            auto search_stat = diskann.Search(query_ds, test_json, nullptr);
            REQUIRE(search_stat.error() == knowhere::Status::out_of_range_in_json);
            LOG_KNOWHERE_INFO_ << "Test pq_read_page_cache_size parameter value less than minimum";
            test_json = knn_search_gen();
            test_json["pq_read_page_cache_size"] = -1;
            search_stat = diskann.Search(query_ds, test_json, nullptr);
            REQUIRE(search_stat.error() == knowhere::Status::out_of_range_in_json);
            LOG_KNOWHERE_INFO_ << "Test pq_read_page_cache_size parameter value more than maximum";
            test_json = knn_search_gen();
            test_json["pq_read_page_cache_size"] = diskann::defaults::MAX_PQ_READ_PAGE_CACHE_SIZE + 1024;
            search_stat = diskann.Search(query_ds, test_json, nullptr);
            REQUIRE(search_stat.error() == knowhere::Status::out_of_range_in_json);
            LOG_KNOWHERE_INFO_ << "Test beamwidth parameter value less than minimum";
            test_json = knn_search_gen();
            test_json["beamwidth"] = 0;
            search_stat = diskann.Search(query_ds, test_json, nullptr);
            REQUIRE(search_stat.error() == knowhere::Status::out_of_range_in_json);
            LOG_KNOWHERE_INFO_ << "Test beamwidth parameter value more than maximum";
            test_json = knn_search_gen();
            test_json["beamwidth"] = diskann::defaults::MAX_AISAQ_BEAMWIDTH + 1;
            search_stat = diskann.Search(query_ds, test_json, nullptr);
            REQUIRE(search_stat.error() == knowhere::Status::aisaq_error);
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}

// Tests AiSAQ parameter validation for build and search operations including cache sizes, entry points,
// inline_pq, beamwidth settings, and disk_pq_dims. Verifies appropriate error codes for invalid parameters.
TEST_CASE("Test_AiSAQ_Params", "[diskann]") {
    base_AiSAQ_param_test<knowhere::fp32>();
}

template <typename DataType>
inline void
base_AiSAQ_search() {
    std::string index_type = "AISAQ";
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kIPIndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kCOSINEIndexDir));

    auto metric_str = GENERATE(as<std::string>{}, knowhere::metric::COSINE, knowhere::metric::IP, knowhere::metric::L2);
    auto version = GenTestVersionList();

    std::unordered_map<knowhere::MetricType, std::string> metric_dir_map = {
        {knowhere::metric::L2, kL2IndexPrefix},
        {knowhere::metric::IP, kIPIndexPrefix},
        {knowhere::metric::COSINE, kCOSINEIndexPrefix},
    };
    std::unordered_map<knowhere::MetricType, float> metric_range_ap_map = {
        {knowhere::metric::L2, kL2RangeAp},
        {knowhere::metric::IP, kIpRangeAp},
        {knowhere::metric::COSINE, kCosineRangeAp},
    };

    auto base_gen = [&metric_str]() {
        knowhere::Json json;
        json["dim"] = kLargeDim;
        json["metric_type"] = metric_str;
        json["k"] = kK;
        if (metric_str == knowhere::metric::L2) {
            json["radius"] = CFG_FLOAT::value_type(200000);
            json["range_filter"] = CFG_FLOAT::value_type(0);
        } else if (metric_str == knowhere::metric::IP) {
            json["radius"] = CFG_FLOAT::value_type(350000);
            json["range_filter"] = std::numeric_limits<CFG_FLOAT::value_type>::max();
        } else {
            json["radius"] = 0.75f;
            json["range_filter"] = 1.0f;
        }
        return json;
    };

    auto build_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 64;
        json["search_list_size"] = 100;
        json["pq_code_budget_gb"] = sizeof(float) * kLargeDim * kNumRows * 0.5 / (1024 * 1024 * 1024);
        json["search_cache_budget_gb"] = 0;
        json["disk_pq_dims"] = kLargeDim;
        json["build_dram_budget_gb"] = 16.0;
        return json;
    };

    auto deserialize_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["search_cache_budget_gb"] = 0;
        return json;
    };

    auto knn_search_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["beamwidth"] = 4;
        json["vectors_beamwidth"] = 4;
        return json;
    };

    auto fp32_query_ds = GenDataSet(kNumQueries, kLargeDim, 42);
    knowhere::DataSetPtr knn_gt_ptr = nullptr;
    knowhere::DataSetPtr range_search_gt_ptr = nullptr;
    auto fp32_base_ds = GenDataSet(kNumRows, kLargeDim, 30);

    auto base_ds = knowhere::ConvertToDataTypeIfNeeded<DataType>(fp32_base_ds);
    auto query_ds = knowhere::ConvertToDataTypeIfNeeded<DataType>(fp32_query_ds);

    {
        auto base_ptr = static_cast<const DataType*>(base_ds->GetTensor());
        WriteRawDataToDisk<DataType>(kRawDataPath, base_ptr, kNumRows, kLargeDim);

        // generate the gt of knn search and range search
        auto base_json = base_gen();

        auto result_knn = knowhere::BruteForce::Search<DataType>(base_ds, query_ds, base_json, nullptr);
        knn_gt_ptr = result_knn.value();
        auto result_range = knowhere::BruteForce::RangeSearch<DataType>(base_ds, query_ds, base_json, nullptr);
        range_search_gt_ptr = result_range.value();
    }
    const auto rearrange_list = {true, false};
    const auto inline_pq_list = {0, -1, 10};
    const auto pq_cache_size_list = {0, 4096};
    const auto num_entry_points_list = {0, 100};
    const auto bitset_percentages = {0.05f, 0.20f, 0.40f, 0.60f, 0.80f, 0.9f};
    const auto bitset_thresholds = {0.9f};
    const auto l_search_dict = {100};
    const auto pq_read_page_cache_size_list = {0, 1048576};
    SECTION("Test search and range search") {
        for (const bool rearrange : rearrange_list) {
            for (const int inline_pq : inline_pq_list) {
                for (const int pq_cache_size : pq_cache_size_list) {
                    for (const int num_entry_points : num_entry_points_list) {
                        std::shared_ptr<milvus::FileManager> file_manager =
                            std::make_shared<milvus::LocalFileManager>();
                        auto diskann_index_pack = knowhere::Pack(file_manager);
                        knowhere::Json deserialize_json = knowhere::Json::parse(deserialize_gen().dump());
                        knowhere::BinarySet binset;
                        auto build_json = build_gen().dump();
                        knowhere::Json json = knowhere::Json::parse(build_json);
                        json["rearrange"] = rearrange;
                        json["inline_pq"] = inline_pq;
                        json["pq_cache_size"] = pq_cache_size;
                        json["num_entry_points"] = num_entry_points;
                        LOG_KNOWHERE_DEBUG_ << "========== build_json =================";
                        LOG_KNOWHERE_DEBUG_ << json.dump();
                        LOG_KNOWHERE_DEBUG_ << "===========================";
                        LOG_KNOWHERE_DEBUG_ << "========== deserialize_json =================";
                        LOG_KNOWHERE_DEBUG_ << deserialize_json.dump();
                        LOG_KNOWHERE_DEBUG_ << "===========================";

                        // build process
                        knowhere::DataSetPtr ds_ptr = nullptr;
                        auto diskann = knowhere::IndexFactory::Instance()
                                           .Create<DataType>(index_type, version, diskann_index_pack)
                                           .value();

                        diskann.Build(ds_ptr, json);
                        diskann.Serialize(binset);
                        // knn search;
                        diskann.Deserialize(binset, deserialize_json);

                        REQUIRE(diskann.HasRawData(metric_str) ==
                                knowhere::IndexStaticFaced<DataType>::HasRawData(index_type, version, json));

                        auto knn_search_json = knn_search_gen().dump();
                        knowhere::Json knn_json = knowhere::Json::parse(knn_search_json);

                        // knn search with bitset
                        std::vector<std::function<std::vector<uint8_t>(size_t, size_t)>> gen_bitset_funcs = {
                            GenerateBitsetWithRandomTbitsSet};
                        for (const int pq_read_page_cache_size : pq_read_page_cache_size_list) {
                            for (const int l_search : l_search_dict) {
                                knn_json["search_list_size"] = l_search;
                                knn_json["pq_read_page_cache_size"] = pq_read_page_cache_size;
                                std::cout << " Test parameters: "
                                          << "metric type: " << metric_str << " rearrange: " << rearrange
                                          << " node cache size: " << deserialize_json["search_cache_budget_gb"] << " Gb"
                                          << " inline_pq: " << inline_pq << " pq_cache_size: " << pq_cache_size
                                          << " pq_read_page_cache_size: " << pq_read_page_cache_size
                                          << " num_entry_points: " << num_entry_points << " vectors_beamwidth: 4"
                                          << " search_list_size: " << l_search << std::endl;
                                for (const float threshold : bitset_thresholds) {
                                    knn_json["filter_threshold"] = threshold;
                                    for (const float percentage : bitset_percentages) {
                                        for (const auto& gen_func : gen_bitset_funcs) {
                                            auto bitset_data = gen_func(kNumRows, percentage * kNumRows);
                                            knowhere::BitsetView bitset(bitset_data.data(), kNumRows);
                                            auto results = diskann.Search(query_ds, knn_json, bitset);
                                            auto gt = knowhere::BruteForce::Search<DataType>(base_ds, query_ds,
                                                                                             knn_json, bitset);
                                            float recall = GetKNNRecall(*gt.value(), *results.value());
                                            std::cout << " Test the map is " << percentage * 100
                                                      << "% full -  Recall :" << recall * 100
                                                      << "%; threshold: " << threshold << " l_search: " << l_search
                                                      << std::endl;
                                            if (percentage == 0.98f) {
                                                REQUIRE(recall >= 0.9f);
                                            } else {
                                                REQUIRE(recall >= AiSAQKnnRecall);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}

// Comprehensive test of AiSAQ with various configuration parameter combinations (rearrange, inline_pq,
// cache sizes, entry points) and multiple metric types (L2, IP, COSINE). Tests include bitset filtering
// and validates search recall against brute-force ground truth.
TEST_CASE("Test_AiSAQ_IndexNode", "[diskann]") {
    base_AiSAQ_search<knowhere::fp32>();
}

// Tests search cancellation mechanism with OpContext for both DISKANN and AISAQ index types.
// Verifies proper handling of cancellation requests, pre-cancelled contexts, and successful searches
// with non-cancelled contexts.
TEST_CASE("Test DiskANN Search Cancellation", "[diskann][cancellation]") {
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));

    auto version = GenTestVersionList();
    auto index_type = GENERATE(as<std::string>{}, "DISKANN", "AISAQ");

    auto build_gen = [&index_type]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = knowhere::metric::L2;
        json["k"] = kK;
        json["index_prefix"] = kL2IndexPrefix;
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 56;
        json["search_list_size"] = 128;
        json["pq_code_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["build_dram_budget_gb"] = 32.0;
        if (index_type == "AISAQ") {
            json["rearrange"] = true;
            json["inline_pq"] = 0;
            json["pq_cache_size"] = 0;
            json["num_entry_points"] = 100;
        }
        return json;
    };

    auto deserialize_gen = [&index_type]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = knowhere::metric::L2;
        json["k"] = kK;
        json["index_prefix"] = kL2IndexPrefix;
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        if (index_type == "AISAQ") {
            json["pq_cache_size"] = 0;
        }
        return json;
    };

    auto search_gen = [&index_type]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = knowhere::metric::L2;
        json["k"] = kK;
        json["index_prefix"] = kL2IndexPrefix;
        json["search_list_size"] = 36;
        json["beamwidth"] = 8;
        if (index_type == "AISAQ") {
            json["vectors_beamwidth"] = 4;
        }
        return json;
    };

    // Prepare data and build index
    auto base_ds = GenDataSet(kNumRows, kDim, 30);
    auto query_ds = GenDataSet(kNumQueries, kDim, 42);
    auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
    WriteRawDataToDisk<float>(kRawDataPath, base_ptr, kNumRows, kDim);

    std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);
    knowhere::BinarySet binset;

    // Build index
    {
        knowhere::DataSetPtr ds_ptr = nullptr;
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version, diskann_index_pack).value();
        diskann.Build(ds_ptr, build_gen());
        diskann.Serialize(binset);
    }

    SECTION("Test Search with cancellation from another thread") {
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version, diskann_index_pack).value();
        diskann.Deserialize(binset, deserialize_gen());

        folly::CancellationSource cs;
        milvus::OpContext op_context(cs.getToken());

        std::atomic<bool> search_started{false};
        std::atomic<bool> search_finished{false};

        std::thread search_thread([&]() {
            search_started = true;
            auto results = diskann.Search(query_ds, search_gen(), nullptr, &op_context);
            search_finished = true;
            (void)results;
        });

        while (!search_started) {
            std::this_thread::yield();
        }

        cs.requestCancellation();
        search_thread.join();

        REQUIRE(search_finished);
    }

    SECTION("Test Search with pre-cancelled context should return error") {
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version, diskann_index_pack).value();
        diskann.Deserialize(binset, deserialize_gen());

        folly::CancellationSource cs;
        milvus::OpContext op_context(cs.getToken());

        cs.requestCancellation();
        bool is_cancelled = op_context.cancellation_token.isCancellationRequested();
        REQUIRE(is_cancelled);

        auto results = diskann.Search(query_ds, search_gen(), nullptr, &op_context);
        REQUIRE(!results.has_value());
        // DISKANN returns diskann_inner_error, AISAQ returns aisaq_error
        auto expected_error =
            (index_type == "AISAQ") ? knowhere::Status::aisaq_error : knowhere::Status::diskann_inner_error;
        REQUIRE((results.error() == expected_error || results.error() == knowhere::Status::cardinal_inner_error));
    }

    SECTION("Test Search without cancellation should succeed") {
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version, diskann_index_pack).value();
        diskann.Deserialize(binset, deserialize_gen());

        // Search without OpContext should succeed
        auto results = diskann.Search(query_ds, search_gen(), nullptr);
        REQUIRE(results.has_value());

        // Search with non-cancelled OpContext should also succeed∏
        folly::CancellationSource cs;
        milvus::OpContext op_context(cs.getToken());
        auto results2 = diskann.Search(query_ds, search_gen(), nullptr, &op_context);
        REQUIRE(results2.has_value());
    }

    fs::remove_all(kDir);
    fs::remove(kDir);
}

// Tests DiskANN index build with NCS upload and verifies data retrieval through various methods.
// Tests both NcsConnector (direct connector reads) and NCSReader (synchronous/asynchronous reads).
// Validates data integrity and presence in NCS storage. Runs with both in_memory and redis NCS backends.
TEST_CASE("DiskANN NcsUpload", "[diskann][ncs]") {
    auto ncs_kind = GENERATE("in_memory", "redis");
    auto test_mode = GENERATE("connector", "reader");
    
    initNcsForTest(ncs_kind);
    
    // Configure test parameters based on mode
    const uint32_t bucketId = (test_mode == std::string("connector")) ? 3260486057 : 1001;
    const uint32_t testNumRows = (test_mode == std::string("connector")) ? 2000 : 1000;
    const uint32_t testDim = (test_mode == std::string("connector")) ? 8 : 128;
    const std::string testDir = (test_mode == std::string("connector")) ? 
        fs::current_path().string() + "/diskann_ncs_test" :
        fs::current_path().string() + "/diskann_ncs_integration_test";
    const std::string indexPrefixSuffix = (test_mode == std::string("connector")) ? "/_disk" : "/test_index";
    
    std::string kNcsTestDir = testDir;
    std::string kNcsRawDataPath = kNcsTestDir + "/raw_data";
    std::string kNcsIndexDir = kNcsTestDir + "/index" + ((test_mode == std::string("connector")) ? "_files" : "");
    std::string kNcsIndexPrefix = kNcsIndexDir + indexPrefixSuffix;
    
    // Clean up and create directories
    fs::remove_all(kNcsTestDir);
    fs::remove(kNcsTestDir);
    REQUIRE_NOTHROW(fs::create_directories(kNcsIndexDir));
    
    Ncs* ncs = NcsSingleton::Instance();
    auto createResult = ncs->createBucket(bucketId);
    REQUIRE(createResult == NcsStatus::OK);
    
    // Generate test data
    auto version = GenTestVersionList();
    auto base_ds = GenDataSet(testNumRows, testDim, (test_mode == std::string("connector")) ? 30 : 42);
    auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
    WriteRawDataToDisk<float>(kNcsRawDataPath, base_ptr, testNumRows, testDim);
    
    auto ncs_extras = getNcsExtras(ncs_kind);
    
    // Build configuration
    knowhere::Json build_config;
    build_config["dim"] = testDim;
    build_config["metric_type"] = "L2";
    build_config["index_prefix"] = kNcsIndexPrefix;
    build_config["data_path"] = kNcsRawDataPath;
    build_config["ncs_enable"] = true;
    build_config["ncs_descriptor"] = NcsDescriptor(ncs_kind, bucketId, ncs_extras);
    
    // Mode-specific configuration
    if (test_mode == std::string("connector")) {
        build_config["max_degree"] = 56;
        build_config["search_list_size"] = 100;
        build_config["pq_code_budget_gb"] = 7.000000096013537e-06;
        build_config["pq_code_budget_gb_ratio"] = 0.125;
        build_config["search_cache_budget_gb"] = 6.000000212225132e-06;
        build_config["search_cache_budget_gb_ratio"] = 0.10000000149011612;
        build_config["build_dram_budget_gb"] = 503.04913330078125;
        build_config["num_build_thread"] = 80;
    } else {
        build_config["max_degree"] = 48;
        build_config["search_list_size"] = 128;
        build_config["pq_code_budget_gb"] = sizeof(float) * testDim * testNumRows * 0.125 / (1024 * 1024 * 1024);
        build_config["search_cache_budget_gb"] = sizeof(float) * testDim * testNumRows * 0.125 / (1024 * 1024 * 1024);
        build_config["build_dram_budget_gb"] = 16.0;
    }
    
    // Build the DiskANN index
    std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);
    
    knowhere::DataSetPtr ds_ptr = nullptr;
    auto diskann = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>("DISKANN", version, diskann_index_pack)
                       .value();
    
    auto build_status = diskann.Build(ds_ptr, build_config);
    REQUIRE(build_status == knowhere::Status::success);
    
    // Verify index files are created
    std::string disk_index_metadata_file = kNcsIndexPrefix + "_disk_metadata.index";
    std::string disk_index_data_file = kNcsIndexPrefix + "_disk_data.index";
    REQUIRE(fs::exists(disk_index_metadata_file));
    REQUIRE(fs::exists(disk_index_data_file));
    
    // Perform NCS upload
    auto ncs_upload_status = diskann.NcsUpload(build_config);
    REQUIRE(ncs_upload_status == knowhere::Status::success);

    uint64_t max_node_len = knowhere::GetDiskANNMaxNodeLenForTest<knowhere::fp32>(diskann);
    
    // Mode-specific verification
    auto descriptor = NcsDescriptor(ncs_kind, bucketId, ncs_extras);
    
    if (test_mode == std::string("connector")) {
        // Verify data through connector
        auto connector = std::unique_ptr<NcsConnector>(
            NcsConnectorFactory::Instance().createConnector(&descriptor));
        REQUIRE(connector != nullptr);
        
        std::vector<uint32_t> test_keys = {0, 1, 2};
        std::vector<SpanBytes> read_spans;
        
        std::vector<std::vector<char>> read_buffers(test_keys.size(), std::vector<char>(max_node_len));
        read_spans.reserve(test_keys.size());
        for (size_t i = 0; i < test_keys.size(); ++i) {
            read_spans.emplace_back(read_buffers[i].data(), max_node_len);
        }
        
        auto get_results = connector->multiGet(test_keys, read_spans);
        REQUIRE(get_results.size() == test_keys.size());
        
        bool some_success = false;
        for (const auto& status : get_results) {
            if (status == NcsStatus::OK) {
                some_success = true;
                break;
            }
        }
        REQUIRE(some_success);
    } else {
        // Test with NCSReader - verify reader can perform sync and async reads
        NCSReader ncs_reader(&descriptor);
        
        // Test synchronous read
        
        int num_test_reads = 5;
        std::vector<std::vector<char>> buffers(num_test_reads, std::vector<char>(max_node_len));
        std::vector<ReadReq> read_reqs;
        read_reqs.reserve(num_test_reads);

        for (int i = 0; i < num_test_reads; ++i) {
            read_reqs.push_back({static_cast<uint32_t>(i), max_node_len, buffers[i].data()});
        }
        
        REQUIRE_NOTHROW(ncs_reader.read(read_reqs));
        
        // Verify buffers contain some data
        bool has_data = false;
        for (const auto& req : read_reqs) {
            const char* buf = static_cast<const char*>(req.buf);
            for (size_t i = 0; i < req.len; ++i) {
                if (buf[i] != 0) {
                    has_data = true;
                    break;
                }
            }
            if (has_data) break;
        }
        REQUIRE(has_data);
        
        // Test asynchronous read
        std::vector<std::vector<char>> async_buffers(num_test_reads, std::vector<char>(max_node_len));
        std::vector<ReadReq> async_read_reqs;
        async_read_reqs.reserve(num_test_reads);
        for (int i = 0; i < num_test_reads; ++i) {
            async_read_reqs.push_back({static_cast<uint32_t>(i), max_node_len, async_buffers[i].data()});
        }
        
        REQUIRE_NOTHROW(ncs_reader.submit_req(async_read_reqs));
        REQUIRE_NOTHROW(ncs_reader.get_submitted_req());
    }
    
    // Clean up
    auto deleteResult = ncs->deleteBucket(bucketId);
    REQUIRE(deleteResult == NcsStatus::OK);
    
    fs::remove_all(kNcsTestDir);
    fs::remove(kNcsTestDir);
}
