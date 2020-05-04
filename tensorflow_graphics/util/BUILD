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
# Utility functionalities for tf-graphics.

# google internal package dependency 8
# google internal package dependency 10

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
        "@com_google_googletest//:gtest_main",
    ],
)

py_library(
    name = "export_api",
    srcs = ["export_api.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
)

py_library(
    name = "doc",
    srcs = ["doc.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
)

py_library(
    name = "test_case",
    srcs = ["test_case.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        ":tfg_flags",
        # google internal package dependency 4,
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
    ],
)

py_library(
    name = "tfg_flags",
    srcs = ["tfg_flags.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        # google internal package dependency 4,
    ],
)

py_library(
    name = "asserts",
    srcs = ["asserts.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        # google internal package dependency 4,
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:tfg_flags",
    ],
)

py_library(
    name = "safe_ops",
    srcs = ["safe_ops.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        ":asserts",
        # google internal package dependency 6
        # google internal package dependency 1,
    ],
)

py_library(
    name = "shape",
    srcs = ["shape.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "util",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        ":asserts",
        ":doc",
        ":export_api",
        ":safe_ops",
        ":shape",
        ":test_case",
        ":tfg_flags",
        "//tensorflow_graphics/util/data_formats",
    ],
)

py_test(
    name = "asserts_test",
    srcs = ["tests/asserts_test.py"],
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
        ":asserts",
        ":test_case",
        # google internal package dependency 3
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
    ],
)

py_test(
    name = "export_api_test",
    srcs = ["tests/export_api_test.py"],
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
        ":export_api",
        ":test_case",
    ],
)

py_test(
    name = "safe_ops_test",
    srcs = ["tests/safe_ops_test.py"],
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
        ":asserts",
        ":safe_ops",
        ":test_case",
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
    ],
)

py_test(
    name = "shape_test",
    srcs = ["tests/shape_test.py"],
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
        ":shape",
        ":test_case",
        # google internal package dependency 2
        # google internal package dependency 1,
    ],
)
