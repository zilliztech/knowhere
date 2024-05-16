#include <fstream>
#include <iomanip>
#include <random>

#include "cuda.h"
#include "cudaProfiler.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/comp/local_file_manager.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/version.h"

knowhere::DataSetPtr
GenDataSet(int rows, int dim) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<> distrib(-1.0, 1.0);
    float* ts = new float[rows * dim];
    for (int i = 0; i < rows * dim; ++i) {
        ts[i] = (float)distrib(rng);
    }
    auto ds = knowhere::GenDataSet(rows, dim, ts);
    ds->SetIsOwner(false);
    return ds;
}

void
Dump(knowhere::BinarySetPtr binset, const std::string& file_name) {
    auto binary_set = *binset;
    auto binary_map = binset->binary_map_;
    std::ofstream outfile;
    outfile.open(file_name, std::ios::out | std::ios::trunc);
    if (outfile.good()) {
        for (auto it = binary_map.begin(); it != binary_map.end(); ++it) {
            // serialization: name_length(size_t); name(char[]); binset_size(size_t); binset(uint8[]);
            auto name = it->first;
            uint64_t name_len = name.size();
            outfile << name_len;
            outfile << name;
            auto value = it->second;
            outfile << value->size;
            outfile.write(reinterpret_cast<char*>(value->data.get()), value->size);
        }
        // end with 0
        outfile << 0;
        outfile.flush();
    }
}
void
Load(knowhere::BinarySetPtr binset, const std::string& file_name) {
    std::ifstream infile;
    infile.open(file_name, std::ios::in);
    if (infile.good()) {
        uint64_t name_len;
        while (true) {
            // deserialization: name_length(size_t); name(char[]); binset_size(size_t); binset(uint8[]);
            infile >> name_len;
            if (name_len == 0)
                break;

            auto _name = new char[name_len];
            infile.read(_name, name_len);
            std::string name(_name, name_len);

            int64_t size;
            infile >> size;
            if (size > 0) {
                auto data = new uint8_t[size];
                std::shared_ptr<uint8_t[]> data_ptr(data);
                infile.read(reinterpret_cast<char*>(data_ptr.get()), size);
                binset->Append(name, data_ptr, size);
            }
        }
    }
}

int
main() {
    knowhere::KnowhereConfig::SetRaftMemPool(1024, 1024);
    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    knowhere::Json json;
    json[knowhere::meta::DIM] = 768;
    json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    json[knowhere::meta::TOPK] = 100;

    json[knowhere::indexparam::INTERMEDIATE_GRAPH_DEGREE] = 256;
    json[knowhere::indexparam::GRAPH_DEGREE] = 128;
    json[knowhere::indexparam::ITOPK_SIZE] = 512;
    json[knowhere::indexparam::SEARCH_WIDTH] = 32;
    json[knowhere::indexparam::BUILD_ALGO] = "NN_DESCENT";
    json[knowhere::indexparam::CACHE_DATASET_ON_DEVICE] = false;
    json[knowhere::indexparam::NN_DESCENT_NITER] = 100;

    std::cout << json.dump() << std::endl;

    // auto xb_ds = GenDataSet(1000000, 768);
    knowhere::Index<knowhere::IndexNode> idx1 =
        knowhere::IndexFactory::Instance()
            .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_RAFT_CAGRA, version)
            .value();

    // idx1.Build(*xb_ds, json);
    std::shared_ptr<knowhere::BinarySet> bset = std::make_shared<knowhere::BinarySet>();
    Load(bset, "./bset");
    idx1.Deserialize(*bset);
    // idx1.Serialize(*bset);
    // Dump(bset, "./bset");

    auto xq_ds = GenDataSet(10, 768);
    for (int i = 0; i < 10000; ++i) auto ret = idx1.Search(*xq_ds, json, nullptr);

    return 0;
}
