#Copyright 2018 Google LLC
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

# google internal library 4)
# google internal library 2

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "export_api",
    srcs = ["export_api.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
    ],
)

py_library(
    name = "test_case",
    srcs = ["test_case.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        ":tfg_flags",
        # package dep,
        # package dep
        # google internal import 1
        # google internal,
    ],
)

py_library(
    name = "tfg_flags",
    srcs = ["tfg_flags.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        # package dep,
    ],
)

py_library(
    name = "asserts",
    srcs = ["asserts.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        # package dep,
        # google internal import 1
        # google internal,
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
        # google internal import 1
        # google internal,
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
    # google internal rule 1
    # google internal rule 2
    # google internal rule 3
    # google internal rule 4
    # google internal rule 5
    # google internal rule 6
    deps = [
        ":asserts",
        ":test_case",
        # package dep
        # package dep
        # google internal import 1
        # google internal,
    ],
)

py_test(
    name = "export_api_test",
    srcs = ["tests/export_api_test.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    # google internal rule 2
    # google internal rule 3
    # google internal rule 4
    # google internal rule 5
    # google internal rule 6
    deps = [
        ":export_api",
        ":test_case",
    ],
)

py_test(
    name = "safe_ops_test",
    srcs = ["tests/safe_ops_test.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    # google internal rule 2
    # google internal rule 3
    # google internal rule 4
    # google internal rule 5
    # google internal rule 6
    deps = [
        ":asserts",
        ":safe_ops",
        ":test_case",
        # package dep
        # google internal import 1
        # google internal,
    ],
)

py_test(
    name = "shape_test",
    srcs = ["tests/shape_test.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    # google internal rule 2
    # google internal rule 3
    # google internal rule 4
    # google internal rule 5
    # google internal rule 6
    deps = [
        ":shape",
        ":test_case",
        # package dep
        # google internal,
    ],
)
