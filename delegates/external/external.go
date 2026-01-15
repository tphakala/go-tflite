package external

/*
#ifndef GO_EXTERNAL_H
#include "external.go.h"
#endif
#cgo CFLAGS: -std=c99
#cgo CXXFLAGS: -std=c99
#cgo LDFLAGS: -ltensorflowlite-delegate_external

// Go 1.24+ CGO optimizations
#cgo noescape TfLiteExternalDelegateOptionsDefault
#cgo noescape TfLiteExternalDelegateOptionsInsert
#cgo noescape TfLiteExternalDelegateCreate
#cgo nocallback TfLiteExternalDelegateOptionsDefault
#cgo nocallback TfLiteExternalDelegateOptionsInsert
#cgo nocallback TfLiteExternalDelegateCreate
#cgo nocallback TfLiteExternalDelegateDelete
*/
import "C"
import (
	"errors"
	"unsafe"

	"github.com/tphakala/go-tflite/delegates"
)

type DelegateOptions struct {
	o       C.TfLiteExternalDelegateOptions
	LibPath string
}

func (o *DelegateOptions) Insert(key, value string) error {
	ckey := C.CString(key)
	cvalue := C.CString(value)
	defer C.free(unsafe.Pointer(ckey))
	defer C.free(unsafe.Pointer(cvalue))
	if C.TfLiteExternalDelegateOptionsInsert(&o.o, ckey, cvalue) == C.kTfLiteError {
		return errors.New("Max options")
	}
	return nil
}

// Delegate is the tflite delegate
type Delegate struct {
	d *C.TfLiteDelegate
}

func New(options DelegateOptions) delegates.Delegater {
	var d *C.TfLiteDelegate
	cpath := C.CString(options.LibPath)
	defer C.free(unsafe.Pointer(cpath))
	coptions := C.TfLiteExternalDelegateOptionsDefault(cpath)
	d = C.TfLiteExternalDelegateCreate(&coptions)
	if d == nil {
		return nil
	}
	return &Delegate{
		d: d,
	}
}

// Delete the delegate
func (d *Delegate) Delete() {
	C.TfLiteExternalDelegateDelete(d.d)
}

// Return a pointer
func (d *Delegate) Ptr() unsafe.Pointer {
	return unsafe.Pointer(d.d)
}
