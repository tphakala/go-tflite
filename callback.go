package tflite

import "C"
import (
	"runtime/cgo"
	"unsafe"
)

type callbackInfo struct {
	user_data any
	f         func(msg string, user_data any)
}

//export _go_error_reporter
func _go_error_reporter(user_data unsafe.Pointer, msg *C.char) {
	cb := cgo.Handle(user_data).Value().(*callbackInfo)
	cb.f(C.GoString(msg), cb.user_data)
}
