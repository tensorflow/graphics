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
# Convolutions and related tools for tf-graphics.

# google internal)

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "convolution",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_convolution",
        ":utils",
        "//tensorflow_graphics/util:export_api",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        # google internal,
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:shape",
    ],
)

py_library(
    name = "graph_convolution",
    srcs = ["graph_convolution.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        # google internal,
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:shape",
    ],
)

py_library(
    name = "graph_pooling",
    srcs = ["graph_pooling.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        # google internal,
        "//tensorflow_graphics/util:export_api",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["tests/utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        # google internal,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "graph_convolution_test",
    srcs = ["tests/graph_convolution_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_convolution",
        # package dep

        # google internal,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "graph_pooling_test",
    srcs = [
        "tests/graph_pooling_test.py",
        "tests/utils_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_pooling",
        ":utils",
        # google internal,
        "//tensorflow_graphics/util:test_case",
    ],
)
