licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured", "if_cuda")

config_setting(
    name = "windows",
    constraint_values = ["@bazel_tools//platforms:windows"],
)

cc_library(
    name = "cuda",
    data = [
        "@local_config_cuda//cuda:cudart",
    ],
    linkopts = select({
        "@local_config_cuda//cuda:darwin": [
            "-Wl,-rpath,../local_config_cuda/cuda/lib",
            "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib",
        ],
        ":windows": [],
        "//conditions:default": [
            "-Wl,-rpath,../local_config_cuda/cuda/lib64",
            "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib64",
        ],
    }),
    deps = [
        "@local_config_cuda//cuda:cudart",
    ],
)

cc_library(
    name = "pointcloud_common_kernels",
    srcs = [
        "cc/kernels/cuda_kernel_utils.h",
        "cc/kernels/defines.hpp",
        "cc/kernels/gpu_device.hpp",
        "cc/kernels/tf_utils.hpp",
        "cc/kernels/tf_gpu_device.hpp",
        "cc/kernels/tf_gpu_device.cpp",
        "cc/kernels/math_helper.h",
        "cc/kernels/grid_utils.h",
        "cc/kernels/rnd_utils.h",
        "cc/kernels/nn_utils.h",
    ],
    deps =  [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ] + if_cuda_is_configured([":cuda",  "@local_config_cuda//cuda:cuda_headers"]),
    # visibility = ["//tfg_custom_ops/compute_keys:__pkg__"],
)
