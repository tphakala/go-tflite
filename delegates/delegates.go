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

// Deleter is an optional interface that delegates can implement
// for deterministic native memory cleanup.
type Deleter interface {
	Delete()
}
