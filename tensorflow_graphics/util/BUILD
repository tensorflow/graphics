# Utility functionalities for tf-graphics.

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cleanup",
    hdrs = ["cleanup.h"],
)

cc_test(
    name = "cleanup_test",
    size = "small",
    srcs = ["tests/cleanup_test.cc"],
    deps = [
        ":cleanup",
        "//testing/base/public:gunit_main",
    ],
)
