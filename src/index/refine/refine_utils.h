#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "faiss/Index.h"
#include "faiss/impl/ScalarQuantizer.h"
#include "knowhere/expected.h"
#include "knowhere/operands.h"

namespace knowhere {

// Returns a baseline faiss::ScalarQuantizer::QuantizerType. The integer
// values for every qtype this function returns match the fork's enum,
// so static_cast at the boundary of a fork IndexScalarQuantizer ctor is
// lossless. The fork enum is retired at the knowhere layer; fork ctors
// are the only remaining consumers.
expected<faiss::ScalarQuantizer::QuantizerType>
get_sq_quantizer_type(const std::string& sq_type);

expected<bool>
is_flat_refine(const std::optional<std::string>& refine_type);

bool
has_lossless_quant(const expected<faiss::ScalarQuantizer::QuantizerType>& quant_type, DataFormatEnum dataFormat);

bool
has_lossless_refine_index(const std::optional<bool>& refine, const std::optional<std::string>& refine_type,
                          DataFormatEnum dataFormat);

expected<std::unique_ptr<faiss::Index>>
pick_refine_index(const DataFormatEnum data_format, const std::optional<std::string>& refine_type,
                  std::unique_ptr<faiss::Index>&& base_index,
                  // These two could be borrowed from base_index. But it seems that
                  //   for HNSW these things are borrowed from base_index.storage.
                  //   So, let's provide these externally
                  const size_t base_d, const faiss::MetricType base_metric_type);

}  // namespace knowhere
