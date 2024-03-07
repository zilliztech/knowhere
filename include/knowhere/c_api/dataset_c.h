#ifndef KNOWHERE_DATASET_C
#define KNOWHERE_DATASET_C

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//
typedef void* knowhere_dataset_handle;

// int knowhere_dataset_create(
//     knowhere_dataset_handle* handle
// );

// int knowhere_dataset_destroy(
//     knowhere_dataset_handle handle
// );

int knowhere_dataset_set_rows(
    knowhere_dataset_handle handle,
    const int64_t rows
);

int knowhere_dataset_set_dim(
    knowhere_dataset_handle handle,
    const int64_t dim
);

int knowhere_dataset_set_tensor(
    knowhere_dataset_handle handle,
    const void* tensor
);

int knowhere_dataset_set_ids(
    knowhere_dataset_handle handle,
    const int64_t* ids
);

int knowhere_dataset_get_dim(
    knowhere_dataset_handle handle,
    int64_t* dim
);

int knowhere_dataset_get_rows(
    knowhere_dataset_handle handle,
    int64_t* rows
);

int knowhere_dataset_get_tensor(
    knowhere_dataset_handle handle,
    void* tensor
);

int knowhere_dataset_get_ids(
    knowhere_dataset_handle handle,
    int64_t* ids
);

#ifdef __cplusplus
}
#endif

#endif
