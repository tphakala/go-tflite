module github.com/tphakala/go-tflite/_example/yolov3

go 1.25

replace github.com/tphakala/go-tflite => ../..

require (
	github.com/tphakala/go-tflite v0.0.0-00010101000000-000000000000
	gocv.io/x/gocv v0.29.0
	golang.org/x/image v0.18.0
)

require github.com/mattn/go-pointer v0.0.1 // indirect
