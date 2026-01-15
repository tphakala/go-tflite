package xnnpack

/*
#ifndef GO_XNNPACK_H
#include "xnnpack.go.h"
#endif
#cgo LDFLAGS: -ltensorflowlite_c

// Go 1.24+ CGO optimizations
#cgo noescape TfLiteXNNPackDelegateCreate
#cgo nocallback TfLiteXNNPackDelegateOptionsDefault
#cgo nocallback TfLiteXNNPackDelegateCreate
#cgo nocallback TfLiteXNNPackDelegateDelete
*/
import "C"
import (
	"runtime"
	"unsafe"

	"github.com/tphakala/go-tflite/delegates"
)

type DelegateOptions struct {
	NumThreads int32
}

// Delegate is the tflite delegate
type Delegate struct {
	d       *C.TfLiteDelegate
	cleanup runtime.Cleanup
}

func New(options DelegateOptions) delegates.Delegater {
	var d *C.TfLiteDelegate
	coptions := C.TfLiteXNNPackDelegateOptionsDefault()
	coptions.num_threads = C.int32_t(options.NumThreads)
	d = C.TfLiteXNNPackDelegateCreate(&coptions)
	if d == nil {
		return nil
	}
	del := &Delegate{d: d}
	del.cleanup = runtime.AddCleanup(del, func(ptr *C.TfLiteDelegate) {
		C.TfLiteXNNPackDelegateDelete(ptr)
	}, d)
	return del
}

// Return a pointer
func (d *Delegate) Ptr() unsafe.Pointer {
	return unsafe.Pointer(d.d)
}
