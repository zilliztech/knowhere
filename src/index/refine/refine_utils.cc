#include "index/refine/refine_utils.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>

#include "faiss/cppcontrib/knowhere/IndexRefine.h"
#include "faiss/cppcontrib/knowhere/IndexScalarQuantizer.h"
#include "fmt/format.h"
#include "knowhere/log.h"
#include "knowhere/tolower.h"

namespace knowhere {

// a supporting function
expected<faiss::cppcontrib::knowhere::ScalarQuantizer::QuantizerType>
get_sq_quantizer_type(const std::string& sq_type) {
    std::map<std::string, faiss::cppcontrib::knowhere::ScalarQuantizer::QuantizerType> sq_types = {
        {"sq4u", faiss::cppcontrib::knowhere::ScalarQuantizer::QT_4bit_uniform},
        {"sq6", faiss::cppcontrib::knowhere::ScalarQuantizer::QT_6bit},
        {"sq8", faiss::cppcontrib::knowhere::ScalarQuantizer::QT_8bit},
        {"fp16", faiss::cppcontrib::knowhere::ScalarQuantizer::QT_fp16},
        {"bf16", faiss::cppcontrib::knowhere::ScalarQuantizer::QT_bf16},
        {"int8", faiss::cppcontrib::knowhere::ScalarQuantizer::QT_8bit_direct_signed}};

    // todo: tolower
    auto sq_type_tolower = str_to_lower(sq_type);
    auto itr = sq_types.find(sq_type_tolower);
    if (itr == sq_types.cend()) {
        return expected<faiss::cppcontrib::knowhere::ScalarQuantizer::QuantizerType>::Err(
            Status::invalid_args, fmt::format("invalid scalar quantizer type ({})", sq_type_tolower));
    }

    return itr->second;
}

expected<bool>
is_flat_refine(const std::optional<std::string>& refine_type) {
    // grab a type of a refine index
    if (!refine_type.has_value()) {
        return true;
    };

    // todo: tolower
    std::string refine_type_tolower = str_to_lower(refine_type.value());
    if (refine_type_tolower == "fp32" || refine_type_tolower == "flat") {
        return true;
    };

    // parse
    auto refine_sq_type = get_sq_quantizer_type(refine_type_tolower);
    if (!refine_sq_type.has_value()) {
        LOG_KNOWHERE_ERROR_ << "Invalid refine type: " << refine_type.value();
        return expected<bool>::Err(Status::invalid_args, fmt::format("invalid refine type ({})", refine_type.value()));
    }

    return false;
}

bool
has_lossless_quant(const expected<faiss::cppcontrib::knowhere::ScalarQuantizer::QuantizerType>& quant_type,
                   DataFormatEnum dataFormat) {
    if (!quant_type.has_value()) {
        return false;
    }

    auto quant = quant_type.value();
    switch (dataFormat) {
        case DataFormatEnum::fp32:
            return false;
        case DataFormatEnum::fp16:
            return quant == faiss::cppcontrib::knowhere::ScalarQuantizer::QuantizerType::QT_fp16;
        case DataFormatEnum::bf16:
            return quant == faiss::cppcontrib::knowhere::ScalarQuantizer::QuantizerType::QT_bf16;
        case DataFormatEnum::int8:
            return quant == faiss::cppcontrib::knowhere::ScalarQuantizer::QuantizerType::QT_8bit_direct_signed;
        default:
            return false;
    }
}

bool
has_lossless_refine_index(const std::optional<bool>& refine, const std::optional<std::string>& refine_type,
                          DataFormatEnum dataFormat) {
    bool has_refine = refine.value_or(false) && refine_type.has_value();
    if (has_refine) {
        expected<bool> flat_refine = is_flat_refine(refine_type);
        if (flat_refine.has_value() && flat_refine.value()) {
            return true;
        }

        auto sq_refine_type = get_sq_quantizer_type(refine_type.value());
        return has_lossless_quant(sq_refine_type, dataFormat);
    }
    return false;
}

// pick a refine index
expected<std::unique_ptr<faiss::cppcontrib::knowhere::Index>>
pick_refine_index(const DataFormatEnum data_format, const std::optional<std::string>& refine_type,
                  std::unique_ptr<faiss::cppcontrib::knowhere::Index>&& base_index, const size_t base_d,
                  const faiss::MetricType base_metric_type) {
    // grab a type of a refine index
    expected<bool> is_fp32_flat = is_flat_refine(refine_type);
    if (!is_fp32_flat.has_value()) {
        return expected<std::unique_ptr<faiss::cppcontrib::knowhere::Index>>::Err(Status::invalid_args, "");
    }

    const bool is_fp32_flat_v = is_fp32_flat.value();

    // check input data_format
    if (data_format == DataFormatEnum::fp16) {
        // make sure that we're using fp16 refine
        auto refine_sq_type = get_sq_quantizer_type(refine_type.value());
        if (!(refine_sq_type.has_value() &&
              (refine_sq_type.value() != faiss::cppcontrib::knowhere::ScalarQuantizer::QT_bf16 && !is_fp32_flat_v))) {
            LOG_KNOWHERE_ERROR_ << "fp16 input data does not accept bf16 or fp32 as a refine index.";
            return expected<std::unique_ptr<faiss::cppcontrib::knowhere::Index>>::Err(
                Status::invalid_args, "fp16 input data does not accept bf16 or fp32 as a refine index.");
        }
    }

    if (data_format == DataFormatEnum::bf16) {
        // make sure that we're using bf16 refine
        auto refine_sq_type = get_sq_quantizer_type(refine_type.value());
        if (!(refine_sq_type.has_value() &&
              (refine_sq_type.value() != faiss::cppcontrib::knowhere::ScalarQuantizer::QT_fp16 && !is_fp32_flat_v))) {
            LOG_KNOWHERE_ERROR_ << "bf16 input data does not accept fp16 or fp32 as a refine index.";
            return expected<std::unique_ptr<faiss::cppcontrib::knowhere::Index>>::Err(
                Status::invalid_args, "bf16 input data does not accept fp16 or fp32 as a refine index.");
        }
    }

    // build
    std::unique_ptr<faiss::cppcontrib::knowhere::Index> local_index = std::move(base_index);

    // either build flat or sq
    if (is_fp32_flat_v) {
        // build IndexFlat as a refine
        auto refine_index = std::make_unique<faiss::cppcontrib::knowhere::IndexRefineFlat>(local_index.get());

        // let refine_index to own everything
        refine_index->own_fields = true;
        local_index.release();

        // reassign
        return refine_index;
    } else {
        // being IndexScalarQuantizer as a refine
        auto refine_sq_type = get_sq_quantizer_type(refine_type.value());

        // a redundant check
        if (!refine_sq_type.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Invalid refine type: " << refine_type.value();
            return expected<std::unique_ptr<faiss::cppcontrib::knowhere::Index>>::Err(
                Status::invalid_args, fmt::format("invalid refine type ({})", refine_type.value()));
        }

        // create an sq
        auto sq_refine = std::make_unique<faiss::cppcontrib::knowhere::IndexScalarQuantizer>(
            base_d, refine_sq_type.value(), base_metric_type);

        auto refine_index =
            std::make_unique<faiss::cppcontrib::knowhere::IndexRefine>(local_index.get(), sq_refine.get());

        // let refine_index to own everything
        refine_index->own_refine_index = true;
        refine_index->own_fields = true;
        local_index.release();
        sq_refine.release();

        // reassign
        return refine_index;
    }
}

}  // namespace knowhere
