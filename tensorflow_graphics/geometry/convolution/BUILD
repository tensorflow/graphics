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
# Convolutions and related tools for tf-graphics.

# google internal package dependency 8
# google internal package dependency 10

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "convolution",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        ":graph_convolution",
        ":graph_pooling",
        ":utils",
        "//tensorflow_graphics/util:export_api",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        # google internal package dependency 1,
        "//tensorflow_graphics/util:shape",
    ],
)

py_library(
    name = "graph_convolution",
    srcs = ["graph_convolution.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        ":utils",
        # google internal package dependency 1,
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:shape",
    ],
)

py_library(
    name = "graph_pooling",
    srcs = ["graph_pooling.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        ":utils",
        # google internal package dependency 1,
        "//tensorflow_graphics/util:export_api",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["tests/utils_test.py"],
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
        ":utils",
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "graph_convolution_test",
    srcs = ["tests/graph_convolution_test.py"],
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
        ":graph_convolution",
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "graph_pooling_test",
    srcs = [
        "tests/graph_pooling_test.py",
        "tests/utils_test.py",
    ],
    shard_count = 4,
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
        ":graph_pooling",
        ":utils",
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)
