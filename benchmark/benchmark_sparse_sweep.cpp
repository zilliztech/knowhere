// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0.
// Reuse the ordinary benchmark's validated input readers and score checks.
#define main ordinary_sparse_benchmark_main
#include "benchmark_sparse.cpp"
#undef main

#include <atomic>
#include <mutex>
#include <unordered_map>

namespace {
void
SaveJson(const std::filesystem::path& path, const Json& value) {
    auto tmp = path.string() + ".tmp";
    std::ofstream stream(tmp);
    stream << value.dump(2) << '\n';
    stream.close();
    Check(stream.good(), "Cannot write " + tmp);
    std::filesystem::rename(tmp, path);
}

double
HitRecall(const Truth& hits, const Truth& truth, size_t nq, int k) {
    size_t shared = 0, valid = 0;
    for (size_t q = 0; q < nq; ++q) {
        for (int j = 0; j < k; ++j) {
            auto id = truth.ids[q * k + j];
            if (id < 0)
                continue;
            ++valid;
            shared += std::find(hits.ids.begin() + q * k, hits.ids.begin() + (q + 1) * k, id) !=
                      hits.ids.begin() + (q + 1) * k;
        }
    }
    return valid ? double(shared) / valid : 1;
}

void
SaveHits(const std::filesystem::path& path, const Truth& hits, size_t nq, int k) {
    std::ofstream f(path);
    f << "query_id,rank,document_id,score\n" << std::setprecision(9);
    for (size_t q = 0; q < nq; ++q)
        for (int j = 0; j < k; ++j)
            f << q << ',' << j + 1 << ',' << hits.ids[q * k + j] << ',' << hits.scores[q * k + j] << '\n';
    f.close();
    Check(f.good(), "Cannot save hits");
}

// An independent exhaustive inverted traversal, with no pruning. FP64 scores
// are ranked before FP32 storage, including zero-score docs, ties by row ID.
// It models FP16 document values for IP, integer TF/original FP32 lengths for BM25;
// executor FP32/fused arithmetic can still differ from this mathematical oracle.
void
Oracle(const Json& cfg, const SparseData& base, const Json& search, const std::filesystem::path& out) {
    struct Posting {
        uint32_t doc;
        float value;
    };
    std::unordered_map<uint32_t, size_t> mapping;
    std::vector<uint64_t> counts;
    for (const auto& row : base.rows)
        for (size_t j = 0; j < row.size(); ++j) {
            auto [it, fresh] = mapping.try_emplace(row[j].id, counts.size());
            if (fresh)
                counts.push_back(0);
            ++counts[it->second];
        }
    std::vector<uint64_t> offsets(counts.size() + 1, 0);
    std::partial_sum(counts.begin(), counts.end(), offsets.begin() + 1);
    std::vector<Posting> postings(offsets.back());
    auto cursors = offsets;
    const bool bm = cfg.at("metric") == "BM25";
    std::vector<float> lengths(base.rows.size(), 0);
    for (size_t d = 0; d < base.rows.size(); ++d)
        for (size_t j = 0; j < base.rows[d].size(); ++j) {
            const auto v = base.rows[d][j];
            if (bm)
                lengths[d] += v.val;
            postings[cursors[mapping.at(v.id)]++] = {uint32_t(d), bm ? v.val : float(knowhere::fp16(v.val))};
        }
    const double k1 = bm ? search.at("bm25_k1").get<double>() : 0;
    const double b = bm ? search.at("bm25_b").get<double>() : 0;
    const double avgdl = bm ? search.at("bm25_avgdl").get<double>() : 1;
    Json records = Json::array();
    for (const auto& item : cfg.at("queries")) {
        auto started = Clock::now();
        auto queries = LoadCsr(item.at("path"), nullptr);
        const size_t nq = queries.rows.size(), width = std::min<size_t>(100, base.rows.size());
        Truth truth;
        truth.ids.resize(nq * 100, -1);
        truth.scores.resize(nq * 100, 0);
        std::atomic<size_t> next{0};
        std::vector<std::thread> workers;
        std::exception_ptr error;
        std::mutex error_mutex;
        for (int worker = 0; worker < cfg.at("threads").get<int>(); ++worker)
            workers.emplace_back([&] {
                try {
                    std::vector<double> scores(base.rows.size());
                    std::vector<uint32_t> ranked(base.rows.size());
                    for (size_t q = next.fetch_add(1); q < nq; q = next.fetch_add(1)) {
                        std::fill(scores.begin(), scores.end(), 0);
                        const auto& row = queries.rows[q];
                        for (size_t j = 0; j < row.size(); ++j) {
                            const auto term = row[j];
                            auto it = mapping.find(term.id);
                            if (it == mapping.end())
                                continue;
                            for (auto p = offsets[it->second]; p < offsets[it->second + 1]; ++p) {
                                const auto entry = postings[p];
                                const double value = entry.value;
                                scores[entry.doc] += bm ? double(term.val) * value * (k1 + 1) /
                                                              (value + k1 * (1 - b + b * lengths[entry.doc] / avgdl))
                                                        : double(term.val) * value;
                            }
                        }
                        std::iota(ranked.begin(), ranked.end(), 0);
                        std::partial_sort(ranked.begin(), ranked.begin() + width, ranked.end(), [&](auto x, auto y) {
                            return scores[x] != scores[y] ? scores[x] > scores[y] : x < y;
                        });
                        for (size_t j = 0; j < width; ++j) {
                            truth.ids[q * 100 + j] = ranked[j];
                            truth.scores[q * 100 + j] = float(scores[ranked[j]]);
                        }
                    }
                } catch (...) {
                    std::lock_guard lock(error_mutex);
                    error = std::current_exception();
                }
            });
        for (auto& worker : workers) worker.join();
        if (error)
            std::rethrow_exception(error);
        const auto path = out / (item.at("label").get<std::string>() + ".represented.k100.gt");
        std::ofstream f(path, std::ios::binary);
        const uint32_t n = nq, k = 100;
        f.write(reinterpret_cast<const char*>(&n), 4);
        f.write(reinterpret_cast<const char*>(&k), 4);
        for (auto id : truth.ids) {
            uint32_t v = id < 0 ? UINT32_MAX : uint32_t(id);
            f.write(reinterpret_cast<const char*>(&v), 4);
        }
        f.write(reinterpret_cast<const char*>(truth.scores.data()), truth.scores.size() * 4);
        f.close();
        Check(f.good(), "Cannot save oracle");
        std::vector<uint64_t> selected(nq);
        std::iota(selected.begin(), selected.end(), 0);
        Json record = {{"query", item.at("label")},   {"queries", nq},         {"corpus_rows", base.rows.size()},
                       {"seconds", Seconds(started)}, {"path", path.string()}, {"scope", "all queries / full corpus"}};
        for (int topk : {10, 100}) {
            auto cached = LoadTruth(item.at("truth").at(std::to_string(topk)), selected, nq, base.rows.size(), topk);
            auto represented = LoadTruth(path, selected, nq, base.rows.size(), topk);
            size_t valid = 0, zero = 0;
            for (size_t j = 0; j < cached.ids.size(); ++j) {
                valid += cached.ids[j] >= 0;
                zero += cached.ids[j] >= 0 && cached.scores[j] == 0;
            }
            record["k" + std::to_string(topk)] = {
                {"represented_vs_original_recall", HitRecall(represented, cached, nq, topk)},
                {"valid_truth", valid},
                {"zero_truth", zero},
                {"positive_truth_fraction", valid ? double(valid - zero) / valid : 1}};
        }
        records.push_back(record);
        SaveJson(out / "oracle.json", records);
        std::cout << "ORACLE " << record.dump() << std::endl;
    }
}

int
Sweep(const Json& cfg, const SparseData& base, Json search, const std::filesystem::path& out) {
    const std::string codec = cfg.at("codec");
    const auto version = knowhere::Version::GetMaximumVersion().VersionNumber();
    Json build = search;
    build["dim"] = base.dim;
    build["inverted_index_algo"] = "SINDI";
    build["quant_type"] = codec;
    build["sindi_window_size"] = 4096;
    build["sindi_h2"] = false;
    auto create = [&] {
        auto value = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
            knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, version);
        Check(value.has_value(), "Cannot create index");
        return std::move(value.value());
    };
    auto off = create();
    auto start = Clock::now();
    Check(off.Build(base.Dataset(), build) == knowhere::Status::success, "Build failed");
    const double build_seconds = Seconds(start);
    const std::string payload = cfg.at("index_path");
    Check(!std::filesystem::exists(payload), "Task index already exists");
    Json storage;
    {
        knowhere::BinarySet bs;
        Check(off.Serialize(bs) == knowhere::Status::success, "Serialize failed");
        auto binary = bs.GetByName(off.Type());
        Check(bool(binary), "No serialized index");
        std::ofstream f(payload, std::ios::binary);
        f.write(reinterpret_cast<const char*>(binary->data.get()), binary->size);
        f.close();
        Check(f.good(), "Cannot save index");
        storage["serialized_bytes"] = binary->size;
        if (cfg.at("metric") == "BM25") {
            auto type = knowhere::sparse::inverted::peek_quant_type_from_index_data(binary->data.get(), binary->size);
            storage["resolved_codec"] =
                type == knowhere::sparse::inverted::InvertedIndexQuantType::BM25_U8 ? "u8" : "u16";
        } else
            storage["resolved_codec"] = "fp16";
    }
    off = create();
    start = Clock::now();
    Check(off.DeserializeFromFile(payload, build) == knowhere::Status::success, "Off mmap failed");
    const double off_load = Seconds(start);
    auto on = create();
    build["sindi_h2"] = true;
    start = Clock::now();
    Check(on.DeserializeFromFile(payload, build) == knowhere::Status::success, "On mmap failed");
    Json resources = {{"build_seconds", build_seconds},
                      {"off_load_seconds", off_load},
                      {"on_load_seconds", Seconds(start)},
                      {"off_bytes", off.Size()},
                      {"on_bytes", on.Size()},
                      {"summary_bytes", on.Size() - off.Size()},
                      {"storage", storage},
                      {"index_policy", "both modes mmap-load the same fresh payload; both resident during timings"}};
    SaveJson(out / "resources.json", resources);
    std::vector<float> ratios = {1.05f, 1.f, .95f, .9f, .8f};
    Json all = Json::array();
    for (const auto& item : cfg.at("queries")) {
        auto queries = LoadCsr(item.at("path"), nullptr);
        queries.dim = std::max(queries.dim, base.dim);
        const size_t nq = queries.rows.size();
        auto query_dataset = queries.Dataset();
        std::vector<uint64_t> selected(nq);
        std::iota(selected.begin(), selected.end(), 0);
        for (int k : {10, 100}) {
            const auto query_label = item.at("label").get<std::string>();
            const auto case_label = query_label + ".k" + std::to_string(k);
            auto truth = LoadTruth(item.at("truth").at(std::to_string(k)), selected, nq, base.rows.size(), k);
            auto represented = LoadTruth(item.at("represented_truth"), selected, nq, base.rows.size(), k);
            Truth baseline[2];
            Json local = Json::array();
            auto run = [&](std::string stage, int session, int block, int mode, int ri, int repeats) {
                search["k"] = k;
                search["dim_max_score_ratio"] = ratios[ri];
                auto& index = mode ? on : off;
                start = Clock::now();
                auto warmup = index.Search(query_dataset, search, nullptr);
                Check(warmup.has_value(), "Warmup failed");
                const double warmup_seconds = Seconds(start);
                warmup.value().reset();
                Truth hits;
                Json batches = Json::array();
                double seconds = 0;
                for (int rep = 0; rep < repeats; ++rep) {
                    start = Clock::now();
                    auto result = index.Search(query_dataset, search, nullptr);
                    const double elapsed = Seconds(start);
                    Check(result.has_value(), "Search failed");
                    auto captured = Capture(result.value(), nq * k);
                    const double recall = Recall(result.value(), truth, nq, k, base.rows.size());
                    if (rep)
                        Check(hits.ids == captured.ids && hits.scores == captured.scores, "Repeat output mismatch");
                    else
                        hits = std::move(captured);
                    seconds += elapsed;
                    batches.push_back({{"seconds", elapsed}, {"qps", nq / elapsed}, {"recall", recall}});
                }
                CheckHitScores(hits, base, queries, k, search);
                if (ri == 0)
                    baseline[mode] = hits;
                auto label = case_label + "." + stage + ".s" + std::to_string(session) + ".b" + std::to_string(block) +
                             ".h" + std::to_string(mode) + ".r" + std::to_string(ri);
                SaveHits(out / (label + ".hits.csv"), hits, nq, k);
                size_t valid = 0, miss_zero = 0, miss_repr = 0, miss_shared = 0;
                for (size_t q = 0; q < nq; ++q)
                    for (int j = 0; j < k; ++j) {
                        auto pos = q * k + j;
                        auto id = truth.ids[pos];
                        if (id < 0)
                            continue;
                        ++valid;
                        if (std::find(hits.ids.begin() + q * k, hits.ids.begin() + (q + 1) * k, id) !=
                            hits.ids.begin() + (q + 1) * k)
                            continue;
                        if (truth.scores[pos] == 0)
                            ++miss_zero;
                        else if (std::find(represented.ids.begin() + q * k, represented.ids.begin() + (q + 1) * k,
                                           id) == represented.ids.begin() + (q + 1) * k)
                            ++miss_repr;
                        else
                            ++miss_shared;
                    }
                struct rusage usage {};
                getrusage(RUSAGE_SELF, &usage);
                Json record = {{"label", label},
                               {"query", query_label},
                               {"category", item.at("category")},
                               {"k", k},
                               {"stage", stage},
                               {"session", session},
                               {"block", block},
                               {"h2", bool(mode)},
                               {"ratio_index", ri},
                               {"ratio", ratios[ri]},
                               {"queries", nq},
                               {"base_rows", base.rows.size()},
                               {"warmup_batches", 1},
                               {"warmup_seconds", warmup_seconds},
                               {"repeats", repeats},
                               {"measurements", batches},
                               {"pooled_qps", nq * repeats / seconds},
                               {"recall", HitRecall(hits, truth, nq, k)},
                               {"represented_recall", HitRecall(hits, represented, nq, k)},
                               {"baseline_output_recall", HitRecall(hits, baseline[mode], nq, k)},
                               {"independent_hit_scores_passed", true},
                               {"repeat_identity_passed", true},
                               {"valid_truth", valid},
                               {"missed_zero_truth", miss_zero},
                               {"missed_positive_displaced_in_represented_truth", miss_repr},
                               {"missed_positive_shared_with_represented_truth", miss_shared},
                               {"peak_rss_kib", usage.ru_maxrss}};
                SaveJson(out / (label + ".json"), record);
                all.push_back(record);
                local.push_back(record);
                SaveJson(out / "runs.json", all);
                std::cout << "RUN " << label << " qps=" << record["pooled_qps"] << " recall=" << record["recall"]
                          << std::endl;
                return record;
            };
            for (int ri = 0; ri < 5; ++ri)
                for (int mode = 0; mode < 2; ++mode) run("screen", 0, 0, mode, ri, 1);
            Check(baseline[0].ids == baseline[1].ids && baseline[0].scores == baseline[1].scores,
                  "Conservative off/on differs");
            std::set<int> selected_ratios;
            Json shortlist = Json::array();
            for (int mode = 0; mode < 2; ++mode) {
                std::set<int> chosen;
                for (const auto& a : local)
                    if (a.at("h2") == bool(mode)) {
                        bool dominated = false;
                        for (const auto& b : local)
                            if (b.at("h2") == bool(mode) && b.at("recall") >= a.at("recall") &&
                                b.at("pooled_qps") >= a.at("pooled_qps") &&
                                (b.at("recall") > a.at("recall") || b.at("pooled_qps") > a.at("pooled_qps")))
                                dominated = true;
                        if (!dominated)
                            chosen.insert(a.at("ratio_index").get<int>());
                    }
                for (double target : {.999, .995, .99}) {
                    double best = -1;
                    int ri = -1;
                    for (const auto& a : local)
                        if (a.at("h2") == bool(mode) && a.at("recall").get<double>() >= target &&
                            a.at("pooled_qps").get<double>() > best) {
                            best = a.at("pooled_qps");
                            ri = a.at("ratio_index");
                        }
                    if (ri >= 0)
                        chosen.insert(ri);
                }
                for (int ri : chosen) {
                    selected_ratios.insert(ri);
                    shortlist.push_back({{"h2", bool(mode)}, {"ratio_index", ri}});
                }
            }
            SaveJson(out / (case_label + ".shortlist.json"), shortlist);
            SaveJson(out / (case_label + ".complete.json"), {{"queries", nq},
                                                             {"selected_ratio_indices", selected_ratios},
                                                             {"conservative_off_on_identical", true},
                                                             {"stage", "complete"}});
        }
    }
    std::filesystem::remove(payload);
    SaveJson(out / "complete.json", {{"stage", "complete"}, {"runs", all.size()}, {"resources", resources}});
    return 0;
}
}  // namespace

int
main(int argc, char** argv) {
    try {
        Check(argc == 3, "Usage: benchmark_sparse_sweep CONFIG.json OUTPUT_DIR");
        const auto cfg = ReadJson(argv[1]);
        const std::filesystem::path out = argv[2];
        Check(!std::filesystem::exists(out), "Output directory already exists");
        std::filesystem::create_directories(out);
        const int threads = cfg.at("threads");
        cpu_set_t cpus;
        CPU_ZERO(&cpus);
        Check(sched_getaffinity(0, sizeof(cpus), &cpus) == 0, "Cannot read affinity");
        Json affinity = Json::array();
        for (int i = 0; i < CPU_SETSIZE; ++i)
            if (CPU_ISSET(i, &cpus))
                affinity.push_back(i);
        Check(affinity.size() == size_t(threads), "Thread count must match full CPU affinity");
        SaveJson(out / "environment.json", {{"config", cfg},
                                            {"affinity", affinity},
                                            {"compiler", __VERSION__},
                                            {"revision", KNOWHERE_SPARSE_BENCHMARK_REVISION},
                                            {"queries", "all; no offsets or limits"}});
        knowhere::KnowhereConfig::SetBuildThreadPoolSize(threads);
        knowhere::KnowhereConfig::SetSearchThreadPoolSize(threads);
        auto base = LoadCsr(cfg.at("base"), nullptr);
        Json search = {
            {"metric_type", cfg.at("metric")}, {"k", 100}, {"drop_ratio_search", 0.0}, {"dim_max_score_ratio", 1.05}};
        const auto metadata = ReadJson(cfg.at("path").get<std::string>() + "/gt/metadata.json");
        Check(metadata.at("metric") == (cfg.at("metric") == "BM25" ? "bm25" : "ip"), "Metric mismatch");
        if (cfg.at("metric") == "BM25") {
            for (auto key : {"k1", "b", "avgdl"}) search[std::string("bm25_") + key] = metadata.at("bm25").at(key);
            for (const auto& row : base.rows)
                for (size_t j = 0; j < row.size(); ++j)
                    Check(row[j].val == std::floor(row[j].val) && row[j].val <= 65535, "Invalid TF");
        }
        if (cfg.at("phase") == "oracle") {
            Oracle(cfg, base, search, out);
            return 0;
        }
        for (const auto& q : cfg.at("queries")) {
            MappedFile f(q.at("path"));
            base.dim = std::max(base.dim, f.Read<uint64_t>(8));
        }
        return Sweep(cfg, base, search, out);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
