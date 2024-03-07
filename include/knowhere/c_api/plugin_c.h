#ifndef KNOWHERE_PLUGIN_C
#define KNOWHERE_PLUGIN_C

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "config_c.h"
#include "dataset_c.h"

#include "plugin_manager_c.h"

// these are the function that are expected to be provided by a plugin

#ifdef __cplusplus
extern "C" {
#endif

//
enum KnowhereIndexDataType {
    KNOWHERE_INDEX_DATATYPE_FLOAT32,
    KNOWHERE_INDEX_DATATYPE_FP16,
    KNOWHERE_INDEX_DATATYPE_BF16,
};

//
typedef void* plugin_index_node_handle;

struct KnowhereIndexRegistration {
    const char* name,
    enum KnowhereIndexDataType index_data_type,
    plugin_index_node_handle template_handle
};

// upon start.
int plugin_initialize(
    struct KnowhereIndexRegistration* registration_slots,
    size_t slots_count
    size_t* slots_used
);
// // // the typical plugin code is the following
// // int plugin_initialize(
// //     struct KnowhereIndexRegistration* registration_slots,
// //     size_t slots_count
// //     size_t* slots_used
// // ) {
// //     slots_used = 0;
// // 
// //     CustomIdx<float>* myCustomIdxFloat = new CustomIdx<float>(...);
// //     registration_slots[slots_used] = KnowhereIndexRegistration {
// //             .name = "CustomIdx", 
// //             .index_data_type = KNOWHERE_INDEX_DATATYPE_FLOAT32,
// //             .plugin_index_node_handle = reinterpret_cast<plugin_index_node_handle>(myCustomIdxFloat)
// //         };
// //     slots_used += 1;
// // 
// //     CustomIdx<fp16>* myCustomIdxFp16 = new CustomIdx<fp16>(...);
// //     registration_slots[slots_used] = KnowhereIndexRegistration {
// //             .name = "CustomIdx", 
// //             .index_data_type = KNOWHERE_INDEX_DATATYPE_FLOAT16,
// //             .plugin_index_node_handle = reinterpret_cast<plugin_index_node_handle>(myCustomIdxFp16)
// //         };
// //     slots_used += 1;
// //     ...
// //
// //     return success;
// // }
//
// // // the typical knowhere code is the following
// // {
// //     // no more that 32 indices
// //     const size_t n_slots = 32;
// //     std::array<KnowhereIndexRegistration, n_slots> slots;

// //     size_t n_slots_used = 0;
// //     auto err = plugin->plugin_initialize(slots.data(), n_slots, n_slots_used);
// //     CHECK_ERROR(err);

// //     for (size_t i = 0; i < std::min(n_slots_used, n_slots); i++) {
// //         IndexFactory::register(slots[i]);
// //     }
// // }

// clean up the resources. 
// for example, closing the logging files
int plugin_finalize();

// maybe, replace with a single plugin_reset() call.


// instantiate a new index based on a template and a config
int plugin_index_node_create_from_template(
    plugin_index_node_handle* handle,
    plugin_index_node_handle template_handle,
    knowhere_config_handle config_handle
);

//
int plugin_index_node_destroy(
    plugin_index_node_handle* handle
);

//
int plugin_index_node_add(
    plugin_index_node_handle handle,
    knowhere_dataset_handle dataset_handle,
    knowhere_config_handle config_handle
);
// // // the typical client code is the following
// // int plugin_index_node_add(
// //     plugin_index_node_handle handle,
// //     knowhere_dataset_handle dataset_handle,
// //     knowhere_config_handle config_handle
// // ) {
// //     BaseCustomIdx* index = reinterpret_cast<BaseCustomIdx*>(handle);
// //
// //     // we'll provide a C++ code for this, which will be compiled into
// //     //   the client code and just calls functions from dataset_c.h
// //     KnowhereDatasetHelper dataset(dataset_handle);
// // 
// //     // we'll provide a C++ code for this, which will be compiled into
// //     //   the client code and just calls functions from config_c.h
// //     KnowhereConfigHelper config(config_handle);
// // 
// //     index->Add(dataset, config);
// // 
// //     return success;
// // }


int plugin_index_node_train(
    plugin_index_node_handle handle,
    knowhere_dataset_handle dataset_handle,
    knowhere_config_handle config_handle
);

int plugin_index_node_search(
    plugin_index_node_handle handle,
    knowhere_dataset_handle dataset_handle,
    knowhere_config_handle config_handle,
    struct KnowhereBitsetView* bitset_view,
    knowhere_dataset_handle result
);

#ifdef __cplusplus
}
#endif

#endif
