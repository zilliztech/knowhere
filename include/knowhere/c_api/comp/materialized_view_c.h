#ifndef KNOWHERE_MATERIALIZED_VIEW_C
#define KNOWHERE_MATERIALIZED_VIEW_C

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//
struct KnowhereMaterializedViewSearchInfo {
    int64_t* keys;
    uint64_t* values;
    size_t count;

    bool is_pure_and;
    bool has_not;
};


#ifdef __cplusplus
}
#endif

#endif
