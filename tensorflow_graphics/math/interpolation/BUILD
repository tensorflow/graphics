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
# Interpolation functionalities for tf-graphics.

# google internal package dependency 8
# google internal package dependency 10

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "interpolation",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    # google internal rule 1
    visibility = ["//visibility:public"],
    deps = [
        ":bspline",
        ":slerp",
        ":weighted",
        "//tensorflow_graphics/util:export_api",
    ],
)

py_library(
    name = "slerp",
    srcs = ["slerp.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        # google internal package dependency 1,
        "//tensorflow_graphics/math:vector",
        "//tensorflow_graphics/util:asserts",
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:safe_ops",
        "//tensorflow_graphics/util:shape",
    ],
)

py_library(
    name = "bspline",
    srcs = ["bspline.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        # google internal package dependency 1,
        "//tensorflow_graphics/util:asserts",
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:shape",
    ],
)

py_library(
    name = "weighted",
    srcs = ["weighted.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        # google internal package dependency 1,
        "//tensorflow_graphics/math:vector",
        "//tensorflow_graphics/util:asserts",
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:safe_ops",
        "//tensorflow_graphics/util:shape",
    ],
)

py_library(
    name = "trilinear",
    srcs = [
        "trilinear.py",
    ],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        # google internal package dependency 1,
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:shape",
    ],
)

py_test(
    name = "slerp_test",
    srcs = ["tests/slerp_test.py"],
    srcs_version = "PY2AND3",
    # google internal rule 2
    # google internal rule 3
    # google internal rule 4
    # google internal rule 7
    # google internal rule 8
    # google internal rule 9
    # google internal rule 10
    # google internal rule 11
    deps = [
        ":slerp",
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "bspline_test",
    srcs = ["tests/bspline_test.py"],
    srcs_version = "PY2AND3",
    # google internal rule 2
    # google internal rule 3
    # google internal rule 4
    # google internal rule 7
    # google internal rule 8
    # google internal rule 9
    # google internal rule 10
    # google internal rule 11
    deps = [
        ":bspline",
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "weighted_test",
    srcs = ["tests/weighted_test.py"],
    srcs_version = "PY2AND3",
    # google internal rule 2
    # google internal rule 3
    # google internal rule 4
    # google internal rule 7
    # google internal rule 8
    # google internal rule 9
    # google internal rule 10
    # google internal rule 11
    deps = [
        ":weighted",
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "trilinear_test",
    srcs = [
        "tests/trilinear_test.py",
    ],
    srcs_version = "PY2AND3",
    # google internal rule 2
    # google internal rule 3
    # google internal rule 4
    # google internal rule 7
    # google internal rule 8
    # google internal rule 9
    # google internal rule 10
    # google internal rule 11
    deps = [
        ":trilinear",
        # google internal package dependency 3
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/geometry/representation:grid",
        "//tensorflow_graphics/geometry/transformation:rotation_matrix_3d",
        "//tensorflow_graphics/util:test_case",
    ],
)
