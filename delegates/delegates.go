package delegates

import (
	"unsafe"
)

type ModifyGraphWithDelegater interface {
	ModifyGraphWithDelegate(Delegater)
}

type Delegater interface {
	Ptr() unsafe.Pointer
}
