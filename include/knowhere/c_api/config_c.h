#ifndef KNOWHERE_CONFIG_C
#define KNOWHERE_CONFIG_C

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "comp/materialized_view_c.h"

#ifdef __cplusplus
extern "C" {
#endif

//
enum KnowhereConfigFieldDescriptionType {
    KNOWHERE_CONFIG_FOR_TRAIN,
    KNOWHERE_CONFIG_FOR_SEARCH,
    KNOWHERE_CONFIG_FOR_RANGE_SEARCH,
    KNOWHERE_CONFIG_FOR_ITERATOR,
    KNOWHERE_CONFIG_FOR_FEDER,
    KNOWHERE_CONFIG_FOR_DESERIALIZE,
    KNOWHERE_CONFIG_FOR_DESERIALIZE_FROM_FILE
};

//
struct KnowhereConfigIntFieldDescription {
    char* name;

    enum KnowhereConfigFieldDescriptionType type;
    int* default_value;
    char* description;
    bool allow_empty_without_default;
    int* range_min;
    int* range_max;
};

//
struct KnowhereConfigFloatFieldDescription {
    char* name;

    enum KnowhereConfigFieldDescriptionType type;
    float* default_value;
    char* description;
    bool allow_empty_without_default;
    float* range_min;
    float* range_max;
};

//
struct KnowhereConfigBoolFieldDescription {
    char* name;

    enum KnowhereConfigFieldDescriptionType type;
    bool* default_value;
    char* description;
    bool allow_empty_without_default;
};

//
struct KnowhereConfigStringFieldDescription {
    char* name;

    enum KnowhereConfigFieldDescriptionType type;
    char* default_value;
    char* description;
    bool allow_empty_without_default;
};

//
struct KnowhereConfigMaterializedViewSearchInfoFieldDescription {
    char* name;

    enum KnowhereConfigFieldDescriptionType type;
    char* description;
    bool allow_empty_without_default;
};

//
typedef void* knowhere_config_handle;

//
int knowhere_get_config_field_string(
    knowhere_config_handle handle,
    const char* name, 
    char* string_value, 
    const size_t string_value_buf_size,
    struct KnowhereConfigStringFieldDescription* field_description
);

int knowhere_get_config_field_bool(
    knowhere_config_handle handle,
    const char* name, 
    bool* bool_value,
    struct KnowhereConfigBoolFieldDescription* field_description
);

int knowhere_get_config_field_int(
    knowhere_config_handle handle,
    const char* name, 
    int* int_value, 
    struct KnowhereConfigIntFieldDescription* field_description
);

int knowhere_get_config_field_float(
    knowhere_config_handle handle,
    const char* name, 
    float* float_value,
    struct KnowhereConfigFloatFieldDescription* field_description
);

int knowhere_get_config_field_materialized_view_search_info(
    knowhere_config_handle handle,
    const char* name, 
    struct KnowhereMaterializedViewSearchInfo* materialized_view_search_info_value,
    struct KnowhereConfigMaterializedViewSearchInfoFieldDescription* field_description
);

#ifdef __cplusplus
}
#endif

#endif