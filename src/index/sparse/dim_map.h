#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "index/sparse/inverted_index_format.h"
#include "knowhere/expected.h"
#include "knowhere/log.h"
#include "ptrhash/ptrhash.hpp"

namespace knowhere::sparse::inverted {

enum class DimMapMphfStorage {
    SeparateSection,
    LegacyTrailer,
};

class SealedDimMap {
 public:
    SealedDimMap() = default;
    SealedDimMap(const SealedDimMap&) = delete;
    SealedDimMap&
    operator=(const SealedDimMap&) = delete;

    SealedDimMap(SealedDimMap&& other) noexcept
        : owned_dim_map_reverse_(std::move(other.owned_dim_map_reverse_)),
          dim_map_reverse_(other.dim_map_reverse_),
          legacy_dim_map_(std::move(other.legacy_dim_map_)),
          dim_map_mphf_(std::move(other.dim_map_mphf_)),
          dim_map_mphf_view_(other.dim_map_mphf_view_),
          loaded_mphf_data_(other.loaded_mphf_data_) {
        refresh_views_after_move();
        other.reset_non_owning_views();
    }

    SealedDimMap&
    operator=(SealedDimMap&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        owned_dim_map_reverse_ = std::move(other.owned_dim_map_reverse_);
        dim_map_reverse_ = other.dim_map_reverse_;
        legacy_dim_map_ = std::move(other.legacy_dim_map_);
        dim_map_mphf_ = std::move(other.dim_map_mphf_);
        dim_map_mphf_view_ = other.dim_map_mphf_view_;
        loaded_mphf_data_ = other.loaded_mphf_data_;
        refresh_views_after_move();
        other.reset_non_owning_views();
        return *this;
    }

    void
    build_from_external_dims(const std::unordered_set<uint32_t>& dims) {
        std::vector<uint32_t> unique_dims;
        unique_dims.reserve(dims.size());
        unique_dims.insert(unique_dims.end(), dims.begin(), dims.end());

        clear_reverse();
        clear_mphf();
        clear_legacy_dim_map();

        if (unique_dims.empty()) {
            return;
        }

        dim_map_mphf_ = ptrhash::PtrHash::build(unique_dims);
        dim_map_mphf_view_ = dim_map_mphf_.view();
        owned_dim_map_reverse_.assign(dim_map_mphf_view_.n(), 0);
        set_owned_reverse_view();

        std::vector<uint8_t> seen(dim_map_mphf_view_.n(), 0);
        for (uint32_t dim : unique_dims) {
            const auto inner_dim = dim_map_mphf_view_.index(static_cast<uint64_t>(dim));
            if (inner_dim >= reverse_size() || seen[inner_dim] != 0) {
                throw std::runtime_error("MPHF generated an invalid sparse dim map");
            }
            owned_dim_map_reverse_[inner_dim] = dim;
            seen[inner_dim] = 1;
        }
    }

    [[nodiscard]] std::optional<uint32_t>
    lookup(uint32_t dim) const {
        if (!has_reverse()) {
            return std::nullopt;
        }

        if (dim_map_mphf_view_.n() == 0) {
            auto dim_it = legacy_dim_map_.find(dim);
            if (dim_it == legacy_dim_map_.cend()) {
                return std::nullopt;
            }
            return dim_it->second;
        }

        const auto candidate = dim_map_mphf_view_.index(static_cast<uint64_t>(dim));
        if (candidate >= reverse_size() || reverse_at(candidate) != dim) {
            return std::nullopt;
        }
        return static_cast<uint32_t>(candidate);
    }

    [[nodiscard]] uint32_t
    size() const {
        return static_cast<uint32_t>(reverse_size());
    }

    [[nodiscard]] size_t
    byte_size() const {
        return reverse_size_bytes() + mphf_size() +
               legacy_dim_map_.size() * sizeof(typename decltype(legacy_dim_map_)::value_type);
    }

    [[nodiscard]] size_t
    reverse_size_bytes() const {
        return reverse_size() * sizeof(uint32_t);
    }

    [[nodiscard]] size_t
    reverse_section_size(DimMapMphfStorage storage) const {
        return reverse_size_bytes() + (storage == DimMapMphfStorage::LegacyTrailer ? mphf_size() : 0);
    }

    [[nodiscard]] bool
    has_mphf_section(DimMapMphfStorage storage) const {
        return storage == DimMapMphfStorage::SeparateSection && mphf_size() != 0;
    }

    [[nodiscard]] size_t
    mphf_section_size(DimMapMphfStorage storage) const {
        return has_mphf_section(storage) ? mphf_size() : 0;
    }

    [[nodiscard]] size_t
    mphf_serialized_size() const {
        return mphf_size();
    }

    std::vector<uint32_t>
    materialize_reverse() const {
        return {dim_map_reverse_.begin(), dim_map_reverse_.end()};
    }

    void
    write_reverse_section(MemoryIOWriter& writer, DimMapMphfStorage storage) const {
        if (has_reverse()) {
            writer.write(dim_map_reverse_.data(), sizeof(uint32_t), reverse_size());
        }
        if (storage == DimMapMphfStorage::LegacyTrailer) {
            write_mphf(writer);
        }
    }

    void
    write_mphf_section(MemoryIOWriter& writer, DimMapMphfStorage storage) const {
        if (has_mphf_section(storage)) {
            write_mphf(writer);
        }
    }

    Status
    load_sections(MemoryIOReader& reader, const std::vector<InvertedIndexSectionHeader>& section_headers,
                  uint32_t expected_dims, DimMapMphfStorage storage) {
        const auto* reverse_section = find_section_header(section_headers, InvertedIndexSectionType::DIM_MAP_REVERSE);
        if (reverse_section == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Sparse inverted index missing DIM_MAP_REVERSE section";
            return Status::invalid_serialized_index_type;
        }

        const auto allow_trailer = storage == DimMapMphfStorage::LegacyTrailer;
        if (auto status = load_reverse_array(reader, *reverse_section, expected_dims, allow_trailer);
            status != Status::success) {
            return status;
        }
        if (expected_dims == 0) {
            return Status::success;
        }

        const auto* mphf_section = find_section_header(section_headers, InvertedIndexSectionType::DIM_MAP_MPHF);
        if (storage == DimMapMphfStorage::LegacyTrailer) {
            if (mphf_section != nullptr) {
                LOG_KNOWHERE_ERROR_ << "Sparse inverted index legacy dim map trailer mode does not support "
                                       "DIM_MAP_MPHF section";
                return Status::invalid_serialized_index_type;
            }

            const auto reverse_bytes = static_cast<uint64_t>(expected_dims) * sizeof(uint32_t);
            const auto legacy_mphf_size = static_cast<size_t>(reverse_section->size - reverse_bytes);
            if (legacy_mphf_size == 0) {
                return load_legacy_dim_map(expected_dims);
            }

            const auto* mphf_data = reader.data() + reader.tellg();
            reader.advance(legacy_mphf_size);
            return load_mphf(mphf_data, legacy_mphf_size, expected_dims);
        }

        if (mphf_section == nullptr) {
            return load_legacy_dim_map(expected_dims);
        }
        if (mphf_section->size == 0) {
            LOG_KNOWHERE_ERROR_ << "Sparse inverted index DIM_MAP_MPHF section is empty";
            return Status::invalid_serialized_index_type;
        }

        reader.seekg(mphf_section->offset);
        const auto* mphf_data = reader.data() + reader.tellg();
        reader.advance(mphf_section->size);
        return load_mphf(mphf_data, static_cast<size_t>(mphf_section->size), expected_dims);
    }

 private:
    Status
    load_reverse_array(MemoryIOReader& reader, const InvertedIndexSectionHeader& section_header, uint32_t expected_dims,
                       bool allow_extra_bytes) {
        const auto reverse_bytes = static_cast<uint64_t>(expected_dims) * sizeof(uint32_t);
        if (section_header.size < reverse_bytes) {
            LOG_KNOWHERE_ERROR_ << "Sparse inverted index DIM_MAP_REVERSE section is truncated, section_size="
                                << section_header.size << ", expected_reverse_bytes=" << reverse_bytes;
            return Status::invalid_serialized_index_type;
        }
        if (!allow_extra_bytes && section_header.size != reverse_bytes) {
            LOG_KNOWHERE_ERROR_
                << "Sparse inverted index DIM_MAP_REVERSE section has unexpected extra bytes, section_size="
                << section_header.size << ", expected_reverse_bytes=" << reverse_bytes;
            return Status::invalid_serialized_index_type;
        }

        reader.seekg(section_header.offset);
        clear_mphf();
        clear_legacy_dim_map();
        clear_reverse();
        dim_map_reverse_ =
            std::span<const uint32_t>(reinterpret_cast<const uint32_t*>(reader.data() + reader.tellg()), expected_dims);
        reader.advance(static_cast<size_t>(reverse_bytes));
        return Status::success;
    }

    Status
    load_mphf(const uint8_t* data, size_t data_size, uint32_t expected_dims) {
        try {
            auto view = ptrhash::PtrHashView::from_bytes(data, data_size);
            if (view.serialized_size() != data_size || view.n() != expected_dims) {
                LOG_KNOWHERE_ERROR_ << "Sparse inverted index MPHF dim map size mismatch, mphf_n=" << view.n()
                                    << ", nr_inner_dims=" << expected_dims
                                    << ", serialized_size=" << view.serialized_size() << ", section_size=" << data_size;
                return Status::invalid_serialized_index_type;
            }
            if (reverse_size() != expected_dims) {
                LOG_KNOWHERE_ERROR_
                    << "Sparse inverted index MPHF dim map loaded before matching DIM_MAP_REVERSE, reverse_size="
                    << reverse_size() << ", nr_inner_dims=" << expected_dims;
                return Status::invalid_serialized_index_type;
            }
            for (uint32_t inner_dim = 0; inner_dim < expected_dims; ++inner_dim) {
                const uint32_t dim = reverse_at(inner_dim);
                if (view.index(static_cast<uint64_t>(dim)) != inner_dim) {
                    LOG_KNOWHERE_ERROR_ << "Sparse inverted index MPHF dim map does not match DIM_MAP_REVERSE";
                    return Status::invalid_serialized_index_type;
                }
            }

            dim_map_mphf_ = ptrhash::PtrHash();
            dim_map_mphf_view_ = view;
            loaded_mphf_data_ = data;
            clear_legacy_dim_map();
            return Status::success;
        } catch (const std::exception& e) {
            LOG_KNOWHERE_ERROR_ << "Failed to load sparse inverted index MPHF dim map: " << e.what();
            return Status::invalid_serialized_index_type;
        }
    }

    Status
    load_legacy_dim_map(uint32_t expected_dims) {
        legacy_dim_map_.clear();
        legacy_dim_map_.reserve(expected_dims);
        for (uint32_t inner_dim = 0; inner_dim < expected_dims; ++inner_dim) {
            legacy_dim_map_[reverse_at(inner_dim)] = inner_dim;
        }
        return Status::success;
    }

    void
    clear_mphf() {
        dim_map_mphf_ = ptrhash::PtrHash();
        dim_map_mphf_view_ = ptrhash::PtrHashView();
        loaded_mphf_data_ = nullptr;
    }

    void
    clear_legacy_dim_map() {
        legacy_dim_map_.clear();
    }

    void
    write_mphf(MemoryIOWriter& writer) const {
        const auto size = mphf_size();
        if (size != 0) {
            writer.write(mphf_data(), size);
        }
    }

    [[nodiscard]] const uint8_t*
    mphf_data() const {
        const auto& owned_bytes = dim_map_mphf_.serialize();
        if (!owned_bytes.empty()) {
            return owned_bytes.data();
        }
        return loaded_mphf_data_;
    }

    [[nodiscard]] size_t
    mphf_size() const {
        return dim_map_mphf_view_.serialized_size();
    }

    [[nodiscard]] bool
    has_reverse() const {
        return !dim_map_reverse_.empty();
    }

    [[nodiscard]] uint32_t
    reverse_at(size_t inner_dim) const {
        return dim_map_reverse_[inner_dim];
    }

    [[nodiscard]] size_t
    reverse_size() const {
        return dim_map_reverse_.size();
    }

    void
    clear_reverse() {
        owned_dim_map_reverse_.clear();
        dim_map_reverse_ = std::span<const uint32_t>();
    }

    void
    set_owned_reverse_view() {
        dim_map_reverse_ = std::span<const uint32_t>(owned_dim_map_reverse_.data(), owned_dim_map_reverse_.size());
    }

    void
    refresh_views_after_move() {
        if (!owned_dim_map_reverse_.empty()) {
            set_owned_reverse_view();
        }

        if (loaded_mphf_data_ == nullptr && !dim_map_mphf_.serialize().empty()) {
            dim_map_mphf_view_ = dim_map_mphf_.view();
        }
    }

    void
    reset_non_owning_views() {
        dim_map_reverse_ = std::span<const uint32_t>();
        legacy_dim_map_.clear();
        dim_map_mphf_view_ = ptrhash::PtrHashView();
        loaded_mphf_data_ = nullptr;
    }

    std::vector<uint32_t> owned_dim_map_reverse_;
    std::span<const uint32_t> dim_map_reverse_;
    std::unordered_map<uint32_t, uint32_t> legacy_dim_map_;
    ptrhash::PtrHash dim_map_mphf_;
    ptrhash::PtrHashView dim_map_mphf_view_;
    const uint8_t* loaded_mphf_data_{nullptr};
};

class GrowableDimMap {
 public:
    uint32_t
    append_legacy_entry(uint32_t dim) {
        if (auto inner_dim = lookup(dim); inner_dim.has_value()) {
            return inner_dim.value();
        }

        const auto inner_dim = static_cast<uint32_t>(dim_map_.size());
        dim_map_[dim] = inner_dim;
        return inner_dim;
    }

    [[nodiscard]] std::optional<uint32_t>
    lookup(uint32_t dim) const {
        auto dim_it = dim_map_.find(dim);
        if (dim_it == dim_map_.cend()) {
            return std::nullopt;
        }
        return dim_it->second;
    }

    [[nodiscard]] uint32_t
    size() const {
        return static_cast<uint32_t>(dim_map_.size());
    }

    [[nodiscard]] size_t
    byte_size() const {
        return dim_map_.size() * sizeof(typename decltype(dim_map_)::value_type);
    }

    std::vector<uint32_t>
    materialize_reverse() const {
        auto reverse = std::vector<uint32_t>(dim_map_.size());
        for (const auto& [dim, inner_dim] : dim_map_) {
            reverse[inner_dim] = dim;
        }
        return reverse;
    }

 private:
    std::unordered_map<uint32_t, uint32_t> dim_map_;
};

}  // namespace knowhere::sparse::inverted
