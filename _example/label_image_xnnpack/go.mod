module github.com/tphakala/go-tflite/_example/label_image_xnnpack

go 1.13

replace github.com/tphakala/go-tflite => ../..

replace github.com/tphakala/go-tflite/delegates/xnnpack => ../../delegates/xnnpack

require (
	github.com/tphakala/go-tflite v0.0.0-00010101000000-000000000000
	github.com/nfnt/resize v0.0.0-20180221191011-83c6a9932646
)
