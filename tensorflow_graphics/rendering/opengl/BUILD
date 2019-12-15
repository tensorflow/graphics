#Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Math functionalities for tf-graphics.

# google internal package dependency 8)
load("//third_party/tensorflow:tensorflow.bzl", "tf_custom_op_library", "tf_gen_op_wrapper_py")
load("//third_party/tensorflow:tensorflow.google.bzl", "tf_custom_op_py_library")
# google internal package dependency 5

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "opengl",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    # google internal rule 1
    visibility = ["//visibility:public"],
    deps = [
        ":math",
        "//tensorflow_graphics/util:export_api",
    ],
)

cc_library(
    name = "EGL_headers",
    hdrs = [
        "EGL/egl.h",
        "EGL/eglext.h",
        "EGL/eglplatform.h",
        "KHR/khrplatform.h",
    ],
    defines = ["USE_OZONE"],
    features = [
        "-parse_headers",
    ],
    includes = ["."],
)

cc_library(
    name = "egl_offscreen_context",
    srcs = ["egl_offscreen_context.cc"],
    hdrs = ["egl_offscreen_context.h"],
    deps = [
        ":EGL_headers",
        ":egl_util",
        ":macros",
        "//tensorflow_graphics/util:cleanup",
        "//third_party/tensorflow/core:lib",
    ],
)

cc_test(
    name = "egl_offscreen_context_test",
    srcs = ["tests/egl_offscreen_context_test.cc"],
    tags = [
        "nomsan",  # Un-initialized values in libEGL.
        "requires-gpu-sm35",
    ],
    deps = [
        ":egl_offscreen_context",
        ":gl_native_libs",
        "//third_party/tensorflow/core:test",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "egl_util",
    srcs = ["egl_util.cc"],
    hdrs = ["egl_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":EGL_headers",
    ],
)

cc_test(
    name = "egl_util_test",
    srcs = ["egl_util_test.cc"],
    deps = [
        ":EGL_headers",
        ":egl_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":EGL_headers",
        "//third_party/tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gl_native_libs",
    linkopts = [
        "-lEGL",
        "-lGLESv2",
    ],
    alwayslink = 1,
)

cc_library(
    name = "gl_program",
    srcs = ["gl_program.cc"],
    hdrs = ["gl_program.h"],
    deps = [
        ":macros",
        "//tensorflow_graphics/util:cleanup",
        "//third_party/tensorflow/core:lib",
    ],
)

cc_test(
    name = "gl_program_test",
    srcs = ["tests/gl_program_test.cc"],
    tags = [
        "nomsan",  # Un-initialized values in libEGL.
        "requires-gpu-sm35",
    ],
    deps = [
        ":egl_offscreen_context",
        ":gl_native_libs",
        ":gl_program",
        "//third_party/tensorflow/core:test",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gl_render_targets",
    srcs = ["gl_render_targets.cc"],
    hdrs = ["gl_render_targets.h"],
    deps = [
        ":macros",
        "//tensorflow_graphics/util:cleanup",
        "//third_party/tensorflow/core:lib",
    ],
)

cc_test(
    name = "gl_render_targets_test",
    srcs = ["tests/gl_render_targets_test.cc"],
    tags = [
        "nomsan",  # Un-initialized values in libEGL.
        "requires-gpu-sm35",
    ],
    deps = [
        ":egl_offscreen_context",
        ":gl_native_libs",
        ":gl_render_targets",
        ":macros",
        "//third_party/tensorflow/core:lib",
        "//third_party/tensorflow/core:test",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gl_shader_storage_buffer",
    srcs = ["gl_shader_storage_buffer.cc"],
    hdrs = ["gl_shader_storage_buffer.h"],
    deps = [
        ":macros",
        "//tensorflow_graphics/util:cleanup",
        "//third_party/tensorflow/core:lib",
    ],
)

cc_test(
    name = "gl_shader_storage_buffer_test",
    srcs = ["tests/gl_shader_storage_buffer_test.cc"],
    tags = [
        "nomsan",  # Un-initialized values in libEGL.
        "requires-gpu-sm35",
    ],
    deps = [
        ":gl_native_libs",
        ":gl_shader_storage_buffer",
        "//third_party/tensorflow/core:test",
        "@com_google_googletest//:gtest_main",
    ],
)

# This library provides stub implementation of the OpenGL libraries.
cc_library(
    name = "gl_stub_libs",
    deps = [
        "//third_party/GL/stub:EGL",  # buildcleaner: keep
        "//third_party/GL/stub:GLESv2",  # buildcleaner: keep
    ],
    alwayslink = 1,
)

py_library(
    name = "math",
    srcs = ["math.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        # google internal package dependency 1,
        "//tensorflow_graphics/math:vector",
        "//tensorflow_graphics/math/interpolation:weighted",
        "//tensorflow_graphics/util:asserts",
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:shape",
    ],
)

py_test(
    name = "math_test",
    timeout = "long",
    srcs = ["tests/math_test.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    # google internal rule 2
    # google internal rule 3
    # google internal rule 4
    # google internal rule 5
    # google internal rule 6
    deps = [
        ":math",
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)

cc_library(
    name = "rasterizer",
    srcs = ["rasterizer.cc"],
    hdrs = ["rasterizer.h"],
    deps = [
        ":gl_program",
        ":gl_render_targets",
        ":gl_shader_storage_buffer",
        "//tensorflow_graphics/util:cleanup",
        "//third_party/tensorflow/core:lib",
    ],
)

cc_test(
    name = "rasterizer_test",
    srcs = ["tests/rasterizer_test.cc"],
    tags = [
        "nomsan",  # Un-initialized values in libEGL.
        "requires-gpu-sm35",
    ],
    deps = [
        ":egl_offscreen_context",
        ":gl_native_libs",
        ":rasterizer",
        "//third_party/tensorflow/core:test",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "rasterizer_op_lib",
    srcs = ["rasterizer_op.cc"],
    deps = [
        ":macros",
        ":rasterizer_with_context",
        ":thread_safe_resource_pool",
        "//third_party/tensorflow/core:framework",
        "//third_party/tensorflow/core:lib",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = 1,
)

tf_custom_op_library(
    name = "rasterizer_op.so",
    srcs = [
        "egl_offscreen_context.cc",
        "egl_offscreen_context.h",
        "egl_util.cc",
        "egl_util.h",
        "gl_program.cc",
        "gl_program.h",
        "gl_render_targets.cc",
        "gl_render_targets.h",
        "gl_shader_storage_buffer.cc",
        "gl_shader_storage_buffer.h",
        "macros.h",
        "rasterizer.cc",
        "rasterizer.h",
        "rasterizer_op.cc",
        "rasterizer_with_context.cc",
        "rasterizer_with_context.h",
        "thread_safe_resource_pool.h",
    ],
    deps = [
        ":EGL_headers",
        "//tensorflow_graphics/util:cleanup",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_rasterizer_op",
    out = "gen_rasterizer_op.py",
    deps = [
        ":gl_stub_libs",
        ":rasterizer_op_lib",
    ],
)

tf_custom_op_py_library(
    name = "rasterizer_op_py",
    dso = [":rasterizer_op.so"],
    kernels = [
        ":rasterizer_op_lib",
    ],
    deps = [":gen_rasterizer_op"],
)

py_test(
    name = "rasterizer_op_test",
    srcs = [
        "tests/rasterizer_op_test.py",
    ],
    srcs_version = "PY2AND3",
    # google internal rule 1
    tags = [
        "nomsan",  # Un-initialized values in libEGL.
    ],
    test_on_cpu = False,
    # google internal rule 3
    # google internal rule 4
    # google internal rule 5
    # google internal rule 6
    deps = [
        ":gl_native_libs",
        ":math",
        ":rasterizer_op_py",
        # google internal package dependency 2
        # google internal package dependency 6
        "//third_party/py/six",
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)

cc_library(
    name = "rasterizer_with_context",
    srcs = ["rasterizer_with_context.cc"],
    hdrs = ["rasterizer_with_context.h"],
    deps = [
        ":egl_offscreen_context",
        ":rasterizer",
        "//tensorflow_graphics/util:cleanup",
        "//third_party/tensorflow/core:lib",
    ],
)

cc_test(
    name = "rasterizer_with_context_test",
    srcs = ["tests/rasterizer_with_context_test.cc"],
    tags = [
        "nomsan",  # Un-initialized values in libEGL.
        "requires-gpu-sm35",
    ],
    deps = [
        ":gl_native_libs",
        ":rasterizer_with_context",
        ":thread_safe_resource_pool",
        "//third_party/tensorflow/core:lib",
        "//third_party/tensorflow/core:test",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "thread_safe_resource_pool",
    hdrs = ["thread_safe_resource_pool.h"],
    deps = [
        ":macros",
        "//third_party/tensorflow/core:lib",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "thread_safe_resource_pool_test",
    srcs = ["tests/thread_safe_resource_pool_test.cc"],
    deps = [
        ":thread_safe_resource_pool",
        "//third_party/tensorflow/core:lib",
        "//third_party/tensorflow/core:test",
        "@com_google_googletest//:gtest_main",
    ],
)
