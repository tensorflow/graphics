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
    name = "compute_keys_ops_gpu",
    srcs = ["cc/kernels/compute_keys.h", "cc/kernels/compute_keys_kernels.cu.cc"],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "//tfg_custom_ops/shared:pointcloud_common_kernels",
    ] + if_cuda_is_configured([":cuda",  "@local_config_cuda//cuda:cuda_headers"]),
    alwayslink = 1,
    linkopts = [],
    copts = select({
        ":windows": ["/D__CLANG_SUPPORT_DYN_ANNOTATION__", "/DEIGEN_MPL2_ONLY", "/DEIGEN_MAX_ALIGN_BYTES=64", "/DEIGEN_HAS_TYPE_TRAITS=0", "/DTF_USE_SNAPPY", "/showIncludes", "/MD", "/O2", "/DNDEBUG", "/w", "-DWIN32_LEAN_AND_MEAN", "-DNOGDI", "/d2ReducedOptimizeHugeFunctions", "/arch:AVX", "/std:c++14", "-DTENSORFLOW_MONOLITHIC_BUILD", "/DPLATFORM_WINDOWS", "/DEIGEN_HAS_C99_MATH", "/DTENSORFLOW_USE_EIGEN_THREADPOOL", "/DEIGEN_AVOID_STL_ARRAY", "/Iexternal/gemmlowp", "/wd4018", "/wd4577", "/DNOGDI", "/UTF_COMPILE_LIBRARY"],
        "//conditions:default": ["-pthread", "-std=c++11"],
    }) + if_cuda_is_configured(["-DTENSORFLOW_USE_NVCC=1", "-DGOOGLE_CUDA=1", "-x cuda", "-nvcc_options=relaxed-constexpr", "-nvcc_options=ftz=true"]),
    visibility = ["//tfg_custom_ops/find_neighbors:__pkg__"],
)

cc_binary(
    name = 'python/ops/_compute_keys_ops.so',
    srcs = [
        "cc/ops/compute_keys_ops.cpp",
    ],
    linkshared = 1,
    features = select({
        ":windows": ["windows_export_all_symbols"],
        "//conditions:default": [],
    }),    
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ] + if_cuda_is_configured([":compute_keys_ops_gpu"]),
    copts = select({
        ":windows": ["/D__CLANG_SUPPORT_DYN_ANNOTATION__", "/DEIGEN_MPL2_ONLY", "/DEIGEN_MAX_ALIGN_BYTES=64", "/DEIGEN_HAS_TYPE_TRAITS=0", "/DTF_USE_SNAPPY", "/showIncludes", "/MD", "/O2", "/DNDEBUG", "/w", "-DWIN32_LEAN_AND_MEAN", "-DNOGDI", "/d2ReducedOptimizeHugeFunctions", "/arch:AVX", "/std:c++14", "-DTENSORFLOW_MONOLITHIC_BUILD", "/DPLATFORM_WINDOWS", "/DEIGEN_HAS_C99_MATH", "/DTENSORFLOW_USE_EIGEN_THREADPOOL", "/DEIGEN_AVOID_STL_ARRAY", "/Iexternal/gemmlowp", "/wd4018", "/wd4577", "/DNOGDI", "/UTF_COMPILE_LIBRARY"],
        "//conditions:default": ["-pthread", "-std=c++11"],
    }) + if_cuda_is_configured(["-DTENSORFLOW_USE_NVCC=1", "-DGOOGLE_CUDA=1", "-x cuda", "-nvcc_options=relaxed-constexpr", "-nvcc_options=ftz=true"]),
)

py_library(
    name = "compute_keys_ops_py",
    srcs = ([
        "python/ops/compute_keys_ops.py",
    ]),
    data = [
        ":python/ops/_compute_keys_ops.so"
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "compute_keys_ops_py_test",
    srcs = [
        "python/ops/compute_keys_ops_test.py"
    ],
    main = "python/ops/compute_keys_ops_test.py",
    deps = [
        ":compute_keys_ops_py",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "compute_keys_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":compute_keys_ops_py"
    ],
    srcs_version = "PY2AND3",
)
