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

#pragma once
#include "io/memory_io.h"
#include "knowhere/expected.h"
#include "knowhere/version.h"

#define KNOWHERE_TRAILER_SIZE 512  // bytes
#define MAX_INDEX_NAME_SIZE 63     // bytes
#define FLAG_TYPE uint32_t
#define FLAG_MAGIC_NUMBER UINT32_C(0x1234ABCD)
#define BEG_FLAG_OFFEST 0
#define END_FLAG_OFFEST KNOWHERE_TRAILER_SIZE / sizeof(FLAG_TYPE) - 1
#define TRAILER_OFFSET(size) size - KNOWHERE_TRAILER_SIZE

namespace knowhere {
/**
 * @brief Trailer is a struct checks the loaded file or binary.
 * The size of trailer is 512 bytes.
 * The format is like:
 *  Begin of a flag(4 bytes): 0x1234ABCD;
 *      Meta info:
 *      version(4 bytes);
 *      checksum(4 bytes);
 *      index_name(64 bytes);
 *      ...
 *  End of a flag(4 bytes): 0x1234ABCD;
 */

#pragma pack(1)
struct Trailer {
 public:
    Trailer() {
        static_assert((sizeof(meta) + sizeof(unused) + 2 * sizeof(FLAG_TYPE)) == KNOWHERE_TRAILER_SIZE,
                      "no more meta information can be added into trailer.");
        beg_flag = FLAG_MAGIC_NUMBER;
        end_flag = FLAG_MAGIC_NUMBER;
    };
    bool
    TrailerValidCheck() {
        return beg_flag == FLAG_MAGIC_NUMBER && end_flag == FLAG_MAGIC_NUMBER;
    }
    std::string
    GetIndexName() {
        return std::string(meta.index_name);
    }
    IndexVersion
    GetVersion() {
        return meta.version;
    }
    uint32_t
    GetCheckSum() {
        return meta.checksum;
    }
    bool
    SetIndexName(std::string index_name) {
        if (index_name.size() > MAX_INDEX_NAME_SIZE) {
            return false;
        } else {
            memcpy((char*)meta.index_name, index_name.data(), index_name.size());
            meta.index_name[index_name.size()] = '\0';
            return true;
        }
    }
    bool
    SetCheckSum(uint32_t value) {
        meta.checksum = value;
        return true;
    }
    bool
    SetVersion(int32_t version) {
        meta.version = version;
        return true;
    }

 private:
    uint32_t beg_flag;
    struct Meta {
        IndexVersion version;
        uint32_t checksum;
        char index_name[MAX_INDEX_NAME_SIZE + 1];
    } meta;
    uint8_t unused[KNOWHERE_TRAILER_SIZE - 2 * sizeof(FLAG_TYPE) - sizeof(Meta)];
    uint32_t end_flag;
};
#pragma pack()

using TrailerPtr = std::unique_ptr<Trailer>;
Status
AddTrailerForMemoryIO(MemoryIOWriter& writer, const std::string& name, const Version& version);
Status
AddTrailerForFiles(const std::vector<std::string>& files, const std::string& trailer_file, const std::string& name,
                   const Version& version);
Status
CheckTrailerForMemoryIO(MemoryIOReader& reader, const std::string& name);
Status
CheckTrailerForFiles(const std::vector<std::string>& files, const std::string& trailer_file, const std::string& name);
}  // namespace knowhere
