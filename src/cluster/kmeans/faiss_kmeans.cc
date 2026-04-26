// Copyright (c) KIOXIA Corporation. All rights reserved.
// Licensed under the MIT license.

#include <faiss/IndexFlat.h>

#include "cluster/kmeans/kmeans_config.h"
#include "common/metric.h"
#include "faiss/Clustering.h"
#include "knowhere/cluster/cluster_factory.h"
#include "knowhere/cluster/cluster_node.h"
#include "knowhere/comp/task.h"
#include "knowhere/thread_pool.h"

namespace knowhere {

struct VariantProgressiveDimIndexFactory : public faiss::ProgressiveDimIndexFactory {
    VariantProgressiveDimIndexFactory(const std::string& metric) {
        this->metric = metric;
    }
    /// ownership transferred to caller
    faiss::Index*
    operator()(int dim) override {
        if (metric == metric::L2) {
            return new faiss::IndexFlatL2(dim);
        } else {
            return new faiss::IndexFlatIP(dim);
        }
    }

    ~VariantProgressiveDimIndexFactory() override {
    }

 private:
    std::string metric;
};

template <typename DataType>
class FaissKmeansClusterNode : public ClusterNode {
 public:
    FaissKmeansClusterNode(const Object& object) {
        search_pool = ThreadPool::GetGlobalSearchThreadPool();
        build_pool = ThreadPool::GetGlobalBuildThreadPool();
    }

    // kmeans train, return id_mapping
    // (rows, uint32_t* id_mapping)
    expected<DataSetPtr>
    Train(const DataSet& dataset, const Config& cfg) override;

    // cluster assign, return id_mapping
    // (rows, uint32_t* id_mapping)
    expected<DataSetPtr>
    Assign(const DataSet& dataset) override {
        size_t score;
        return AssignInternal(dataset, false, score);
    }

    // return centroids, must be called after trained
    // (rows, dim, centroid_vector_list)
    expected<DataSetPtr>
    GetCentroids() const override;

    std::unique_ptr<Config>
    CreateConfig() const override;

    std::string
    Type() const override;

    ~FaissKmeansClusterNode() override {
    }

    inline bool
    CheckMetric(const std::string& metric) {
        if (metric != metric::L2 && metric != metric::IP) {
            LOG_KNOWHERE_ERROR_ << "FaissKmeans currently only supports L2 and IP metric type" << std::endl;
            return false;
        } else {
            return true;
        }
    }

 private:
    expected<DataSetPtr>
    AssignInternal(const DataSet& dataset, bool balance, size_t& score);
    std::shared_ptr<faiss::Clustering> clustering;
    std::unique_ptr<faiss::ProgressiveDimIndexFactory> index_factory;
    std::shared_ptr<ThreadPool> search_pool;
    std::shared_ptr<ThreadPool> build_pool;
    std::vector<size_t> points_per_cluster;
};

template <typename DataType>
expected<DataSetPtr>
FaissKmeansClusterNode<DataType>::Train(const DataSet& dataset, const Config& cfg) {
    auto kmeans_conf = static_cast<const KmeansConfig&>(cfg);
    const auto dim = dataset.GetDim();
    const auto num_rows = dataset.GetRows();
    if (!CheckMetric(kmeans_conf.metric_type.value())) {
        LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << kmeans_conf.metric_type.value();
        return expected<DataSetPtr>::Err(Status::invalid_metric_type, "Invalid metric type");
    }
    if (!kmeans_conf.num_clusters.has_value()) {
        LOG_KNOWHERE_ERROR_ << "kmeans num_clusters is empty." << std::endl;
        return expected<DataSetPtr>::Err(Status::invalid_param_in_json, "kmeans num_clusters is empty");
    }
    index_factory = std::make_unique<VariantProgressiveDimIndexFactory>(kmeans_conf.metric_type.value());
    const auto num_clusters = kmeans_conf.num_clusters.value();
    std::shared_ptr<faiss::Clustering> previous_clustering = clustering;
    if (previous_clustering &&
        (previous_clustering->k != (size_t)num_clusters || previous_clustering->d != (size_t)dim)) {
        LOG_KNOWHERE_ERROR_ << "train called again with different params: " << num_clusters << " , " << dim
                            << std::endl;
        return expected<DataSetPtr>::Err(Status::invalid_param_in_json, "rain called again with different params");
    }
    std::vector<size_t> total_points_per_cluster = points_per_cluster;
    size_t best_score = (size_t)-1, current_score = 0;
    std::vector<size_t> best_point_per_cluster;
    expected<DataSetPtr> best_res;
    std::shared_ptr<faiss::Clustering> best_clustering;
    // for first train batch try several kmeans to select the one which brings max balance
    int num_iter = previous_clustering ? 1 : kmeans_conf.num_iter.value();
    for (int i = 0; i < num_iter; i++) {
        points_per_cluster.resize(num_clusters);
        clustering = std::make_shared<faiss::Clustering>(dim, num_clusters);
        if (previous_clustering) {
            clustering->centroids = previous_clustering->centroids;
        }
        // clustering->progressive_dim_steps = kmeans_conf.dim_steps.value();
        clustering->seed = -1;
        // clustering->apply_pca = kmeans_conf.with_pca.value();
        clustering->max_points_per_centroid = (num_rows / num_clusters) * 11 / 10;
        clustering->verbose = true;
        clustering->niter = kmeans_conf.num_iter.value();

        LOG_KNOWHERE_INFO_ << "start cluster training with " << num_rows << " vectors, num clusters: " << num_clusters
                           << " metric type: " << kmeans_conf.metric_type.value();
        auto train_data = static_cast<const DataType*>(dataset.GetTensor());
        {
            {
                int num_threads = kmeans_conf.num_build_thread.has_value() ? kmeans_conf.num_build_thread.value() : 0;
                ThreadPool::ScopedBuildOmpSetter setter(num_threads);
                clustering->train(num_rows, train_data, *(*index_factory)(dataset.GetDim()));
            }
            auto res = AssignInternal(dataset, false, current_score);
            if (!res.has_value()) {
                return res;
            }
            if (current_score < best_score) {
                best_score = current_score;
                best_clustering = clustering;
                best_point_per_cluster = points_per_cluster;
                best_res = res;
            }
        }
    }
    clustering = best_clustering;
    LOG_KNOWHERE_INFO_ << "training done. best_score: " << best_score;
    float a = 1.0;
    if (previous_clustering) {
        // update streaming training
        // c_t+1 = [(c_t * n_t * a) + (x_t * m_t)] / [n_t + m_t]
        // n_t+t = n_t * a + m_t
        for (size_t i = 0; i < clustering->k; i++) {
            if (best_point_per_cluster[i]) {
                for (size_t j = 0; j < (size_t)dim; j++) {
                    size_t index = i * dim + j;
                    previous_clustering->centroids[index] =
                        (previous_clustering->centroids[index] * total_points_per_cluster[i] * a +
                         clustering->centroids[index] * best_point_per_cluster[i]) /
                        (best_point_per_cluster[i] + total_points_per_cluster[i]);
                }
                total_points_per_cluster[i] += best_point_per_cluster[i];
            }
        }
        clustering = previous_clustering;
        points_per_cluster = total_points_per_cluster;
    }
    return best_res;
}

template <typename DataType>
expected<DataSetPtr>
FaissKmeansClusterNode<DataType>::AssignInternal(const DataSet& dataset, bool balance, size_t& score) {
    if (!clustering) {
        LOG_KNOWHERE_ERROR_ << "Kmeans not prepared";
        return expected<DataSetPtr>::Err(Status::empty_index, "kmeans not loaded");
    }
    const auto num_rows = dataset.GetRows();
    const float* queries = static_cast<const float*>(dataset.GetTensor());
    size_t probes = balance ? clustering->k : 1;
    std::vector<DataType> dis(num_rows * probes);
    std::vector<faiss::idx_t> idx(num_rows * probes);
    size_t minimum_row = 3;
    size_t avg_points_per_cluster = (num_rows / clustering->k);
    size_t max_points_per_cluster = avg_points_per_cluster * 2;
    size_t min_points_per_cluster = avg_points_per_cluster / 100 + 1;
    size_t min_points_for_transfer = min_points_per_cluster * 10;
    size_t num_points_assigned_not_optimally = 0;
    std::vector<size_t> location_diff(num_rows);
    memset(points_per_cluster.data(), 0, clustering->k * sizeof(size_t));
    memset(location_diff.data(), 0, num_rows * sizeof(size_t));
    auto p_id = std::make_unique<uint32_t[]>(num_rows);
    std::unique_ptr<faiss::Index> index((*index_factory)(dataset.GetDim()));
    if (!index->is_trained) {
        index->train(clustering->k, clustering->centroids.data());
    }

    index->add(clustering->k, clustering->centroids.data());

    try {
        std::vector<folly::Future<folly::Unit>> futs;
        futs.reserve(num_rows);
        // 1 thread per element
        ThreadPool::ScopedSearchOmpSetter setter(1);
        for (int64_t i = 0; i < num_rows; ++i) {
            futs.emplace_back(search_pool->push([&, current = i] {
                index->search(1, &queries[current * dataset.GetDim()], probes, dis.data() + current * probes,
                              idx.data() + current * probes);
            }));
        }
        WaitAllSuccess(futs);
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
        return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
    }
    if (balance) {
        for (int i = 0; i < num_rows; i++) {
            for (size_t j = 0; j < clustering->k; j++) {
                faiss::idx_t next_best = idx[i * clustering->k + j];
                if (points_per_cluster[next_best] < max_points_per_cluster) {
                    p_id.get()[i] = next_best;
                    points_per_cluster[next_best]++;
                    if (j > 0) {
                        location_diff[i] = j;
                        num_points_assigned_not_optimally++;
                    }
                    break;
                }
            }
        }
        std::sort(location_diff.begin(), location_diff.end(), std::greater<size_t>());
    } else {
        for (int i = 0; i < num_rows; i++) {
            p_id.get()[i] = idx[i];
            points_per_cluster[idx[i]]++;
        }
    }
    size_t min_points = num_rows, max_points = 0;

    for (size_t i = 0; i < clustering->k; i++) {
        if (points_per_cluster[i] < min_points) {
            min_points = points_per_cluster[i];
        }
        if (points_per_cluster[i] > max_points) {
            max_points = points_per_cluster[i];
        }
        while (balance && points_per_cluster[i] < min_points_per_cluster) {
            size_t best_row = clustering->k + 1;
            int selected_point = -1;
            for (int j = 0; j < num_rows; j++) {
                bool found = false;
                if (p_id.get()[j] != (uint32_t)i) {
                    faiss::idx_t* row_selections = idx.data() + j * clustering->k;
                    for (size_t l = 1; l < clustering->k; l++) {
                        if (row_selections[l] == (faiss::idx_t)i &&
                            points_per_cluster[row_selections[0]] > min_points_for_transfer) {
                            if (l < best_row) {
                                best_row = l;
                                selected_point = j;
                                if (l <= minimum_row) {
                                    found = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (found) {
                        break;
                    }
                }
            }
            p_id.get()[selected_point] = i;
            points_per_cluster[i]++;
            points_per_cluster[selected_point]--;
        }
    }
    LOG_KNOWHERE_INFO_ << "End assign: min points: " << min_points << " max points: " << max_points
                       << " num reassigned: " << num_points_assigned_not_optimally
                       << " max shift: " << location_diff[0];
    score = max_points - min_points;
    auto ret_ds = std::make_shared<DataSet>();
    ret_ds->SetRows(num_rows);
    ret_ds->SetTensor(std::move(p_id));
    ret_ds->SetIsOwner(true);
    return ret_ds;
}

// return centroids, must be called after trained
// (rows, dim, centroid_vector_list)
template <typename DataType>
expected<DataSetPtr>
FaissKmeansClusterNode<DataType>::GetCentroids() const {
    if (!clustering) {
        LOG_KNOWHERE_ERROR_ << "Kmeans not prepared";
        return expected<DataSetPtr>::Err(Status::empty_index, "kmeans not loaded");
    }
    auto ret_ds = std::make_shared<DataSet>();
    auto centroids = std::make_unique<DataType[]>(clustering->k * clustering->d);
    memcpy(centroids.get(), clustering->centroids.data(), sizeof(DataType) * clustering->k * clustering->d);
    ret_ds->SetRows(clustering->k);
    ret_ds->SetDim(clustering->d);
    ret_ds->SetTensor(std::move(centroids));
    ret_ds->SetIsOwner(true);
    return ret_ds;
}

template <typename DataType>
std::unique_ptr<Config>
FaissKmeansClusterNode<DataType>::CreateConfig() const {
    return std::make_unique<KmeansConfig>();
}

template <typename DataType>
std::string
FaissKmeansClusterNode<DataType>::Type() const {
    return knowhere::ClusterEnum::CLUSTER_KMEANS;
}

KNOWHERE_CLUSTER_SIMPLE_REGISTER_GLOBAL(KMEANS, FaissKmeansClusterNode, fp32);

}  // namespace knowhere
