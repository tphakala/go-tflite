#ifndef GO_TFLITE_H
#define GO_TFLITE_H

#define _GNU_SOURCE
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdint.h>
#include <tensorflow/lite/c/c_api.h>

extern void _go_error_reporter(uintptr_t, char*);

static void
_error_reporter(void *user_data, const char* format, va_list args) {
  char *ptr;
  if (asprintf(&ptr, format, args)) {}
  _go_error_reporter((uintptr_t)user_data, ptr);
  free(ptr);
}

static void
_TfLiteInterpreterOptionsSetErrorReporter(TfLiteInterpreterOptions* options, uintptr_t user_data) {
  TfLiteInterpreterOptionsSetErrorReporter(options, _error_reporter, (void*)user_data);
}
#endif
