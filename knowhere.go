package knowhere

/*
#cgo pkg-config: knowhere
#include <stdlib.h>
int CheckConfig(int index_type, char const* str, int n, int param_type);
*/
import "C"

import (
	"encoding/json"
	"errors"
	"fmt"
	"unsafe"
)

func CheckConfig(name string, params map[string]string, ptype string) error {

	mJson, err := json.Marshal(params)
	if err != nil {
		fmt.Println(err.Error())
		return err
	}
	jsonStr := string(mJson)

	var indexType int
	switch name {
	case "FLAT":
		indexType = 0
	case "DISKANN":
		indexType = 1
	case "HNSW":
		indexType = 2
	case "IVF_FLAT":
		indexType = 3
	case "IVF_PQ":
		indexType = 4
	case "GPU_CAGRA":
		indexType = 5
	case "GPU_IVF_PQ":
		indexType = 6
	case "GPU_IVF_FLAT":
		indexType = 7
	case "GPU_BRUTE_FORCE":
		indexType = 8
	default:
		return errors.New("index is not supported.")
	}

	var paramType int
	switch ptype {
	case "TRAIN":
		paramType = 1 << 0
	case "SEARCH":
		paramType = 1 << 1
	case "RANGE_SEARCH":
		paramType = 1 << 2
	case "FEDER":
		paramType = 1 << 3
	case "DESERIALIZE":
		paramType = 1 << 4
	case "DESERIALIZE_FROM_FILE":
		paramType = 1 << 5
	case "ITERATOR":
		paramType = 1 << 6
	case "CLUSTER":
		paramType = 1 << 7
	default:
		return errors.New("param type is not supported.")
	}

	c_str := C.CString(jsonStr)
	ret, err := C.CheckConfig(C.int(indexType), c_str, C.int(len(jsonStr)), C.int(paramType))
	C.free(unsafe.Pointer(c_str))
	if err != nil {
		return err
	}
	if ret == 0 {
		return nil
	}

	return errors.New("Config check failed.")
}
