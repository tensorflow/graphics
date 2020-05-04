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
# Layers tf-graphics.

# google internal package dependency 8
# google internal package dependency 10

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "layer",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    # google internal rule 1
    visibility = ["//visibility:public"],
    deps = [
        ":graph_convolution",
        "//tensorflow_graphics/util:export_api",
    ],
)

py_library(
    name = "graph_convolution",
    srcs = ["graph_convolution.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        # google internal package dependency 1,
        "//tensorflow_graphics/geometry/convolution:graph_convolution",
        "//tensorflow_graphics/geometry/convolution:utils",
        "//tensorflow_graphics/util:export_api",
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

py_library(
    name = "pointnet",
    srcs = ["pointnet.py"],
    srcs_version = "PY2AND3",
    deps = [
        # google internal package dependency 1,
    ],
)

py_test(
    name = "pointnet_test",
    size = "small",
    timeout = "long",
    srcs = ["tests/pointnet_test.py"],
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
        ":pointnet",
        # google internal package dependency 2
        "//tensorflow_graphics/util:test_case",
    ],
)
