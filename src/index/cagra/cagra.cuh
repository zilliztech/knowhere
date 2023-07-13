#include "cagra_config.h"
#include "common/raft/raft_utils.h"
#include "common/raft_metric.h"
#include "knowhere/dataset.h"
#include "knowhere/index_node.h"
#include "knowhere/log.h"
#include "raft/neighbors/cagra.cuh"
#include "raft/neighbors/cagra_serialize.cuh"

namespace knowhere {

namespace detail {
struct device_setter {
    device_setter(int new_device)
        : prev_device_{[]() {
              auto result = int{};
              RAFT_CUDA_TRY(cudaGetDevice(&result));
              return result;
          }()} {
        RAFT_CUDA_TRY(cudaSetDevice(new_device));
    }

    ~device_setter() {
        RAFT_CUDA_TRY_NO_THROW(cudaSetDevice(prev_device_));
    }

 private:
    int prev_device_;
};
}  // namespace detail

class CagraIndexNode : public IndexNode {
    using idx_type = std::int64_t;
    using cagra_index = raft::neighbors::experimental::cagra::index<float, idx_type>;

 public:
    CagraIndexNode(const Object& object) : devs_{}, gpu_index_{} {
    }

    Status
    Train(const DataSet& dataset, const Config& cfg) override {
        auto cagra_cfg = static_cast<const knowhere::CagraConfig&>(cfg);
        if (gpu_index_) {
            LOG_KNOWHERE_WARNING_ << "index is already trained";
            return Status::index_already_trained;
        }
        if (cagra_cfg.gpu_ids.value().size() != 1) {
            LOG_KNOWHERE_WARNING_ << "Cagra implementation is single-GPU only" << std::endl;
            return Status::raft_inner_error;
        }
        auto metric = Str2RaftMetricType(cagra_cfg.metric_type.value());
        if (!metric.has_value()) {
            LOG_KNOWHERE_WARNING_ << "please check metric value: " << cagra_cfg.metric_type.value();
            return metric.error();
        }
        if (metric.value() != raft::distance::DistanceType::L2Expanded) {
            LOG_KNOWHERE_WARNING_ << "only support L2Expanded metric type";
            return Status::invalid_metric_type;
        }
        if (cagra_cfg.intermediate_graph_degree.value() < cagra_cfg.graph_degree.value()) {
            LOG_KNOWHERE_WARNING_ << "Intermediate graph degree must be bigger than graph degree" << std::endl;
            return Status::raft_inner_error;
        }
        try {
            devs_.insert(devs_.begin(), cagra_cfg.gpu_ids.value().begin(), cagra_cfg.gpu_ids.value().end());
            auto scoped_device = raft_utils::device_setter{*cagra_cfg.gpu_ids.value().begin()};
            raft_utils::init_gpu_resources();

            auto build_params = raft::neighbors::experimental::cagra::index_params{};
            build_params.intermediate_graph_degree = cagra_cfg.intermediate_graph_degree.value();
            build_params.graph_degree = cagra_cfg.graph_degree.value();
            build_params.metric = metric.value();
            auto& res = raft_utils::get_raft_resources();
            auto rows = dataset.GetRows();
            auto dim = dataset.GetDim();
            auto* data = reinterpret_cast<float const*>(dataset.GetTensor());
            auto data_gpu = raft::make_device_matrix<float, idx_type>(res, rows, dim);
            RAFT_CUDA_TRY(cudaMemcpyAsync(data_gpu.data_handle(), data, data_gpu.size() * sizeof(float),
                                          cudaMemcpyDefault, res.get_stream().value()));
            gpu_index_ = raft::neighbors::experimental::cagra::build<float, idx_type>(
                res, build_params, raft::make_const_mdspan(data_gpu.view()));
            this->dim_ = dim;
            this->counts_ = rows;
            res.sync_stream();
        } catch (std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "RAFT inner error, " << e.what();
            return Status::raft_inner_error;
        }
        return Status::success;
    }

    Status
    Add(const DataSet& dataset, const Config& cfg) override {
        return Status::success;
    }

    expected<DataSetPtr>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        auto cagra_cfg = static_cast<const CagraConfig&>(cfg);
        auto rows = dataset.GetRows();
        auto dim = dataset.GetDim();
        if (cagra_cfg.k.value() >= cagra_cfg.itopk_size.value()) {
            LOG_KNOWHERE_WARNING_ << "topk should be smaller than itopk_size parameter" << std::endl;
            return Status::raft_inner_error;
        }
        auto* data = reinterpret_cast<float const*>(dataset.GetTensor());
        auto output_size = rows * cagra_cfg.k.value();
        auto ids = std::unique_ptr<idx_type[]>(new idx_type[output_size]);
        auto dis = std::unique_ptr<float[]>(new float[output_size]);
        try {
            auto scoped_device = raft_utils::device_setter{devs_[0]};
            auto& res = raft_utils::get_raft_resources();

            auto data_gpu = raft::make_device_matrix<float, idx_type>(res, rows, dim);
            raft::copy(data_gpu.data_handle(), data, data_gpu.size(), res.get_stream());

            auto search_params = raft::neighbors::experimental::cagra::search_params{};
            search_params.max_queries = cagra_cfg.max_queries.value();
            search_params.itopk_size = cagra_cfg.itopk_size.value();

            auto ids_dev = raft::make_device_matrix<idx_type, idx_type>(res, rows, cagra_cfg.k.value());
            auto dis_dev = raft::make_device_matrix<float, idx_type>(res, rows, cagra_cfg.k.value());
            raft::neighbors::experimental::cagra::search(res, search_params, *gpu_index_,
                                                         raft::make_const_mdspan(data_gpu.view()), ids_dev.view(),
                                                         dis_dev.view());

            raft::copy(ids.get(), ids_dev.data_handle(), output_size, res.get_stream());
            raft::copy(dis.get(), dis_dev.data_handle(), output_size, res.get_stream());
            res.sync_stream();

        } catch (std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "RAFT inner error, " << e.what();
            return Status::raft_inner_error;
        }
        return GenResultDataSet(rows, cagra_cfg.k.value(), ids.release(), dis.release());
    }

    expected<DataSetPtr>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        return Status::not_implemented;
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSet& dataset) const override {
        return Status::not_implemented;
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        return false;
    }

    expected<DataSetPtr>
    GetIndexMeta(const Config& cfg) const override {
        return Status::not_implemented;
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (!gpu_index_.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Can not serialize empty RaftCagraIndex.";
            return Status::empty_index;
        }
        std::stringbuf buf;
        std::ostream os(&buf);
        os.write((char*)(&this->dim_), sizeof(this->dim_));
        os.write((char*)(&this->counts_), sizeof(this->counts_));
        os.write((char*)(&this->devs_[0]), sizeof(this->devs_[0]));

        auto scoped_device = raft_utils::device_setter{devs_[0]};
        auto& res = raft_utils::get_raft_resources();

        raft::neighbors::experimental::cagra::serialize<float, idx_type>(res, os, *gpu_index_);

        os.flush();
        std::shared_ptr<uint8_t[]> index_binary(new (std::nothrow) uint8_t[buf.str().size()]);

        memcpy(index_binary.get(), buf.str().c_str(), buf.str().size());
        binset.Append(this->Type(), index_binary, buf.str().size());
        return Status::success;
    }

    Status
    Deserialize(const BinarySet& binset, const Config& config) override {
        std::stringbuf buf;
        auto binary = binset.GetByName(this->Type());
        if (binary == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Invalid binary set.";
            return Status::invalid_binary_set;
        }
        buf.sputn((char*)binary->data.get(), binary->size);
        std::istream is(&buf);

        is.read((char*)(&this->dim_), sizeof(this->dim_));
        is.read((char*)(&this->counts_), sizeof(this->counts_));
        this->devs_.resize(1);
        is.read((char*)(&this->devs_[0]), sizeof(this->devs_[0]));
        auto scoped_device = raft_utils::device_setter{devs_[0]};

        raft_utils::init_gpu_resources();
        auto& res = raft_utils::get_raft_resources();

        auto index_ = raft::neighbors::experimental::cagra::deserialize<float, idx_type>(res, is);
        is.sync();
        gpu_index_ = cagra_index(std::move(index_));

        return Status::success;
    }

    Status
    DeserializeFromFile(const std::string& filename, const Config& config) override {
        LOG_KNOWHERE_ERROR_ << "CAGRA doesn't support Deserialization from file.";
        return Status::not_implemented;
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<CagraConfig>();
    }

    int64_t
    Dim() const override {
        return dim_;
    }

    int64_t
    Size() const override {
        return 0;
    }

    int64_t
    Count() const override {
        return counts_;
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_RAFT_CAGRA;
    }

 private:
    std::vector<int32_t> devs_;
    int64_t dim_ = 0;
    int64_t counts_ = 0;
    std::optional<cagra_index> gpu_index_;
};

}  // namespace knowhere
