#ifndef KNOWHERE_BITSETVIEW_C
#define KNOWHERE_BITSETVIEW_C

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "comp/materialized_view_c.h"

#ifdef __cplusplus
extern "C" {
#endif

struct KnowhereBitsetView {
    uint8_t* bits;
    size_t num_bits;
};

#ifdef __cplusplus
}
#endif

#endif