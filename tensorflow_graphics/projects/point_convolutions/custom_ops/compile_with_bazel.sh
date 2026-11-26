#!/usr/bin/env bash
# to test the compilation of a single op
if ['$2' == 'clean']; then
  bazel clean
fi
bazel build 'tfg_custom_ops/'$1":python/ops/_"$1"_ops.so" --verbose_failures
bazel test 'tfg_custom_ops/'$1:$1'_ops_py_test'