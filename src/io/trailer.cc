// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include "io/trailer.h"

#include <cstring>
#include <fstream>

namespace {
static constexpr size_t kBlockSize = 4096;
uint32_t
CalculateCheckSum(const uint8_t* data, int64_t size) {
    uint32_t checksum = 0;
    for (auto i = 0; i < size; i++) {
        checksum ^= data[i];  // xor
    }
    return checksum;
}

uint32_t
GetFilesCheckSum(std::vector<std::string> files) {
    uint32_t checksum = 0;
    auto buffer = std::unique_ptr<uint8_t[]>(new uint8_t[kBlockSize]);
    for (auto& file_name : files) {
        std::ifstream reader(file_name.c_str(), std::ios::binary);
        if (!reader) {
            LOG_KNOWHERE_WARNING_ << file_name << "not exist, skip calculate check sum for this file.";
            continue;
        }
        while (reader.read((char*)buffer.get(), kBlockSize)) {
            std::streamsize read_size = reader.gcount();
            checksum ^= CalculateCheckSum(buffer.get(), read_size);
        }
    }
    return checksum;
}

knowhere::Status
FillTrailer(knowhere::TrailerPtr& trailer_ptr, const std::string& name, const uint32_t checksum,
            const knowhere::Version& version) {
    bool fill_success = true;
    fill_success &= trailer_ptr->SetCheckSum(checksum);
    fill_success &= trailer_ptr->SetVersion(version.VersionNumber());
    fill_success &= trailer_ptr->SetIndexName(name);
    if (!fill_success) {
        return knowhere::Status::invalid_trailer;
    }
    return knowhere::Status::success;
}

knowhere::Status
TrailerCheck(const knowhere::TrailerPtr& trailer_ptr, const std::string& name, const uint32_t checksum) {
    if (!trailer_ptr->TrailerValidCheck()) {
        LOG_KNOWHERE_WARNING_ << "Trailer flag check failed.";
        return knowhere::Status::invalid_trailer;
    }

    auto version = knowhere::Version(trailer_ptr->GetVersion());
    if (!knowhere::Version::VersionSupport(version)) {
        LOG_KNOWHERE_ERROR_ << "Index version(" << version.VersionNumber() << ") is not supported, Trailer check fail.";
        return knowhere::Status::invalid_trailer;
    }

    if (trailer_ptr->GetIndexName() != name) {
        LOG_KNOWHERE_ERROR_ << "Index type or data type is not correct(" << name << ").";
        return knowhere::Status::invalid_trailer;
    }

    if (trailer_ptr->GetCheckSum() != checksum) {
        LOG_KNOWHERE_ERROR_ << "Checksum check fail.";
        return knowhere::Status::invalid_trailer;
    }
    return knowhere::Status::success;
}
}  // namespace

namespace knowhere {
Status
AddTrailerForMemoryIO(MemoryIOWriter& writer, const std::string& name, const Version& version) {
    auto trailer_ptr = std::make_unique<Trailer>();
    auto size = writer.tellg();
    auto status = FillTrailer(trailer_ptr, name, CalculateCheckSum(writer.data(), size), version);
    if (status != Status::success) {
        return status;
    }
    writer.write(reinterpret_cast<char*>(trailer_ptr.get()), KNOWHERE_TRAILER_SIZE);
    return Status::success;
}

Status
CheckTrailerForMemoryIO(MemoryIOReader& reader, const std::string& name) {
    // check trailer exist
    if (reader.size() == reader.tellg()) {
        LOG_KNOWHERE_WARNING_ << "Trailer not exist.";
        return Status::success;
    }
    // check trailer sizes
    uint64_t bin_size = TRAILER_OFFSET(reader.size());
    if (bin_size < 0) {
        LOG_KNOWHERE_ERROR_ << "Trailer size is not correct.";
        return Status::invalid_trailer;
    }
    // check trailer meta
    auto trailer_ptr = std::make_unique<Trailer>();
    auto pre_rp = reader.tellg();
    reader.seekg(bin_size);
    reader.read(trailer_ptr.get(), KNOWHERE_TRAILER_SIZE);
    reader.seekg(pre_rp);
    // check trailer meta
    auto checksum = CalculateCheckSum(reader.data(), bin_size);
    return TrailerCheck(trailer_ptr, name, checksum);
}

Status
AddTrailerForFiles(const std::vector<std::string>& files, const std::string& trailer_file, const std::string& name,
                   const Version& version) {
    auto trailer_ptr = std::make_unique<Trailer>();
    auto status = FillTrailer(trailer_ptr, name, GetFilesCheckSum(files), version);
    if (status != Status::success) {
        return status;
    }
    std::ofstream writer(trailer_file.c_str(), std::ios::binary);
    writer.write(reinterpret_cast<char*>(trailer_ptr.get()), KNOWHERE_TRAILER_SIZE);
    writer.close();
    return Status::success;
}

Status
CheckTrailerForFiles(const std::vector<std::string>& files, const std::string& trailer_file, const std::string& name) {
    std::ifstream reader(trailer_file.c_str(), std::ios::binary);
    // check trailer file exists
    if (!reader) {
        LOG_KNOWHERE_WARNING_ << "Trailer file not exist.";
        return Status::success;
    }

    reader.seekg(0, std::ios::end);
    auto fsize = reader.tellg();
    reader.seekg(0, std::ios::beg);
    // check trailer size
    if (fsize != KNOWHERE_TRAILER_SIZE) {
        LOG_KNOWHERE_ERROR_ << "Trailer size (" << fsize << ")not correct.";
        return Status::invalid_trailer;
    }
    // trailer meta check
    auto trailer_ptr = std::make_unique<Trailer>();
    reader.read(reinterpret_cast<char*>(trailer_ptr.get()), KNOWHERE_TRAILER_SIZE);
    auto checksum = GetFilesCheckSum(files);
    return TrailerCheck(trailer_ptr, name, checksum);
}

}  // namespace knowhere
