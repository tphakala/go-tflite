module github.com/mattn/go-tflite/_example/yolov3

go 1.13

replace github.com/mattn/go-tflite => ../..

require (
	github.com/mattn/go-tflite v0.0.0-00010101000000-000000000000
	github.com/tphakala/go-tflite v0.0.0-20231013114437-e78004b1b843 // indirect
	gocv.io/x/gocv v0.29.0
	golang.org/x/image v0.5.0
)
