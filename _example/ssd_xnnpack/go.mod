module github.com/mattn/go-tflite/_example/ssd_xnnpack

go 1.13

replace github.com/mattn/go-tflite => ../..

replace github.com/mattn/go-tflite/delegates/xnnpack => ../../delegates/xnnpack

require (
	github.com/mattn/go-tflite v0.0.0-00010101000000-000000000000
	github.com/tphakala/go-tflite v0.0.0-20231013114437-e78004b1b843 // indirect
	gocv.io/x/gocv v0.29.0
	golang.org/x/image v0.10.0
)
