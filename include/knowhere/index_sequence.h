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

#ifndef INDEX_SEQUENCE_H
#define INDEX_SEQUENCE_H
#include <iostream>
#include <memory>
namespace knowhere {
class IndexSequence {
 public:
    IndexSequence(std::unique_ptr<uint8_t[]>&& obj = nullptr, size_t seq_size = 0)
        : seq(std::make_unique<Sequence_Impl<std::default_delete<uint8_t[]>>>(
              std::forward<std::unique_ptr<uint8_t[], std::default_delete<uint8_t[]>>>(obj))),
          size(seq_size) {
    }

    template <class DEL_T = std::default_delete<uint8_t[]>>
    IndexSequence(std::unique_ptr<uint8_t[], DEL_T>&& obj, size_t seq_size = 0)
        : seq(std::make_unique<Sequence_Impl<DEL_T>>(std::forward<std::unique_ptr<uint8_t[], DEL_T>>(obj))),
          size(seq_size) {
    }

    IndexSequence(IndexSequence&& index_seq) {
        *this = std::move(index_seq);
    }

    IndexSequence(const IndexSequence&) = delete;

    IndexSequence&
    operator=(IndexSequence&& index_seq) {
        seq = std::move(index_seq.seq);
        size = index_seq.size;
        index_seq.size = 0;
        return *this;
    }

    IndexSequence&
    operator=(const IndexSequence&) = delete;

    size_t
    GetSize() const {
        return size;
    }

    uint8_t*
    GetSeq() const {
        return seq.get()->get_seq();
    }

    bool
    Empty() const {
        return (seq == nullptr) || (size == 0);
    }

    struct Sequence {
        virtual ~Sequence() {
        }
        virtual uint8_t*
        get_seq() = 0;
    };

    template <typename DEL_T>
    struct Sequence_Impl : Sequence {
        Sequence_Impl(std::unique_ptr<uint8_t[], DEL_T>&& obj) : seq(std::move(obj)) {
        }
        uint8_t*
        get_seq() {
            return seq.get();
        };
        std::unique_ptr<uint8_t[], DEL_T>
        steal() {
            return std::move(seq);
        };

     private:
        std::unique_ptr<uint8_t[], DEL_T> seq;
    };

 private:
    std::unique_ptr<Sequence> seq;
    size_t size = 0;
};
}  // namespace knowhere
#endif /* INDEX_SEQUENCE_H */
