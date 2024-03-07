#ifndef KNOWHERE_INDEX_NODE_C
#define KNOWHERE_INDEX_NODE_C

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "comp/materialized_view_c.h"
#include "bitsetview_c.h"
#include "config_c.h"
#include "dataset_c.h"

#ifdef __cplusplus
extern "C" {
#endif

//
typedef void* knowhere_index_node_handle;

//
int knowhere_index_node_build(
    knowhere_index_node_handle handle,
    knowhere_dataset_handle dataset_handle,
    knowhere_config_handle config_handle
);

int knowhere_index_node_train(
    knowhere_index_node_handle handle,
    knowhere_dataset_handle dataset_handle,
    knowhere_config_handle config_handle
);

int knowhere_index_node_search(
    knowhere_index_node_handle handle,
    knowhere_dataset_handle dataset_handle,
    knowhere_config_handle config_handle,
    struct KnowhereBitsetView* bitset_view,
    knowhere_dataset_handle result
);

int knowhere_index_node_populate_config(
    knowhere_index_node_handle handle,
    knowhere_config_handle config_handle
);

#ifdef __cplusplus
}
#endif

#endif
