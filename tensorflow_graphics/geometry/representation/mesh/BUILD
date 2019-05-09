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
# Mesh functionalities for tf-graphics.

# google internal)

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "mesh",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":normals",
        ":utils",
        "//tensorflow_graphics/util:export_api",
    ],
)

py_library(
    name = "normals",
    srcs = ["normals.py"],
    srcs_version = "PY2AND3",
    deps = [

        # google internal,
        "//tensorflow_graphics/geometry/representation:triangle",
        "//tensorflow_graphics/math:vector",
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:safe_ops",
        "//tensorflow_graphics/util:shape",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_graphics/util:export_api",
    ],
)

py_test(
    name = "normals_test",
    srcs = ["tests/normals_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":normals",
        # package dep

        # google internal,
        "//tensorflow_graphics/math:vector",
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["tests/utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        # package dep
        "//tensorflow_graphics/util:test_case",
    ],
)
