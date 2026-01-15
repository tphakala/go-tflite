module github.com/tphakala/go-tflite/_example/fizzbuzz_edgetpu

go 1.13

replace github.com/tphakala/go-tflite => ../..

replace github.com/tphakala/go-tflite/delegates/edgetpu => ../../delegates/edgetpu

require github.com/tphakala/go-tflite v0.0.0-00010101000000-000000000000
