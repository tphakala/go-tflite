module github.com/mattn/go-tflite/_example/webcam

go 1.13

replace github.com/mattn/go-tflite => ../..

require (
	github.com/faiface/pixel v0.10.0
	github.com/mattn/go-tflite v0.0.0-00010101000000-000000000000
	github.com/nfnt/resize v0.0.0-20180221191011-83c6a9932646
	github.com/tphakala/go-tflite v0.0.0-20231013114437-e78004b1b843 // indirect
	gocv.io/x/gocv v0.29.0
	golang.org/x/image v0.5.0
)
