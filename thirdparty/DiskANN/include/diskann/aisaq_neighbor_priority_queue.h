// Copyright (c) KIOXIA Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef AISAQ_NEIGHBOR_PRIORITY_QUEUE_H
#define AISAQ_NEIGHBOR_PRIORITY_QUEUE_H

#pragma once
#include <stdint.h>
#include "neighbor.h"

namespace diskann {

    class NeighborPriorityQueue {
    public:

        NeighborPriorityQueue() : _size(0), _capacity(0), _cur(0) {
        }

        explicit NeighborPriorityQueue(size_t capacity) : _size(0), _capacity(capacity), _cur(0), _data(capacity + 1) {
        }

        // Inserts the item ordered into the set up to the sets capacity.
        // The item will be dropped if it is the same id as an exiting
        // set item or it has a greated distance than the final
        // item in the set. The set cursor that is used to pop() the
        // next item will be set to the lowest index of an uncheck item

        void insert(const diskann::Neighbor &nbr) {
            if (_size == _capacity && _data[_size - 1] < nbr) {
                return;
            }

            size_t lo = 0, hi = _size;
            while (lo < hi) {
                size_t mid = (lo + hi) >> 1;
                if (nbr < _data[mid]) {
                    hi = mid;
                    // Make sure the same id isn't inserted into the set
                } else if (_data[mid].id == nbr.id) {
                    return;
                } else {
                    lo = mid + 1;
                }
            }

            if (lo < _capacity) {
                std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof (diskann::Neighbor));
            }
            _data[lo] = {nbr.id, nbr.distance};
            if (_size < _capacity) {
                _size++;
            }
            if (lo < _cur) {
                _cur = lo;
            }
        }

        void insert_with_rem_info(const diskann::Neighbor &nbr, bool &rem, uint32_t &id) {
            if (_size == _capacity && _data[_size - 1] < nbr) {
                rem = true;
                id = nbr.id;
                return;
            }

            size_t lo = 0, hi = _size;
            while (lo < hi) {
                size_t mid = (lo + hi) >> 1;
                if (nbr < _data[mid]) {
                    hi = mid;
                    // Make sure the same id isn't inserted into the set
                } else if (_data[mid].id == nbr.id) {
                    rem = false;
                    return;
                } else {
                    lo = mid + 1;
                }
            }

            if (_size == _capacity) {
                rem = true;
                id = _data[_size - 1].id;
            } else {
                rem = false;
            }

            if (lo < _capacity) {
                std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof (diskann::Neighbor));
            }
            _data[lo] = {nbr.id, nbr.distance};
            if (_size < _capacity) {
                _size++;
            }
            if (lo < _cur) {
                _cur = lo;
            }
        }

        diskann::Neighbor closest_unexpanded() {
            _data[_cur].flag = true;
            size_t pre = _cur;
            while (_cur < _size && _data[_cur].flag) {
                _cur++;
            }
            return _data[pre];
        }

        bool has_unexpanded_node() const {
            return _cur < _size;
        }

        bool get_first_unexpanded_position(size_t &position, bool expand) {
            position = _cur;
            if (position < _size) {
                if (expand) {
                    closest_unexpanded();
                }
                return true;
            }
            return false;
        }

        bool get_next_unexpanded_position(size_t &position, bool expand) {
            position++;
            while (position < _size && _data[position].flag) {
                position++;
            }
            if (position < _size) {
                if (expand) {
                    if (position == _cur) {
                        closest_unexpanded();
                    } else {
                        _data[position].flag = true;
                    }
                }
                return true;
            }
            return false;
        }

        size_t size() const {
            return _size;
        }

        size_t capacity() const {
            return _capacity;
        }

        void reserve(size_t capacity) {
            if (capacity + 1 > _data.size()) {
                _data.resize(capacity + 1);
            }
            _capacity = capacity;
        }

        diskann::Neighbor &operator[](size_t i) {
            return _data[i];
        }

        diskann::Neighbor operator[](size_t i) const {
            return _data[i];
        }

        void remove(size_t position) {
        	if(position >= _size){
        		return;
        	}
            std::memmove(&_data[position], &_data[position + 1],
                          (_size - position - 1) * sizeof(Neighbor));
            _size--;
            if(_cur == position){
                while (_cur < _size && _data[_cur].flag) {
                	_cur++;
                }
            } else if(_cur > position){
            	_cur --;
            }
        }
        void clear() {
            _size = 0;
            _cur = 0;
        }

    private:
        size_t _size, _capacity, _cur;
        std::vector<diskann::Neighbor> _data;
    };

}

#endif /* AISAQ_NEIGHBOR_PRIORITY_QUEUE_H */
