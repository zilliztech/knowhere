I'm assuming that a plugin is a completely independent library that never uses `knowhere` binary (shared or static) and never references it.

Knowhere should create a separate library called `knowhere_structs`. This library contains implementations of `Dataset`, `Config` and all other primitives used in Knowhere, and also the provides the C API for the functions defined in `config_c.h`, `dataset_c.h`. This library is compiled with the same compiler as the knowhere and this library has to be completely independent of `knowhere` host library. Knowhere library dynamically links to `knowhere_structs`. The plugin will reference this dynamic library as well.

So, the knowhere starts and at certain moment it loads a plugin dynamic library.
Immediately after that, knowhere invokes 
``` C
int plugin_initialize(
    struct KnowhereIndexRegistration* registration_slots,
    size_t slots_count
    size_t* slots_used
);
```
from `plugin_c.h` and populated the index template it wants to register.

Alternatively - and I think that it is *more preferreble* - we keep `Dataset`, `Config`, etc. as is, but in this case we'll need to provide a struct with callbacks to functions defined in `config_c.h`, `dataset_c.h` right inside `int plugin_initialize()` call:
```C
struct CallbackTable {
    int (*knowhere_dataset_set_rows)(
        knowhere_dataset_handle handle,
        const int64_t rows
    );
    ...
};

int plugin_initialize(
    struct CallbackTable callback_table,
    struct KnowhereIndexRegistration* registration_slots,
    size_t slots_count
    size_t* slots_used
);

```
so that a plugin would use it.

Things like `knowhere_dataset_handle` and `knowhere_dataset_handle` are just pointer recasts to `void*` of internal knowhere objects.
