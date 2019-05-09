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
# Optimizer functionalities for tf-graphics.

# google internal)

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "optimizer",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":levenberg_marquardt",
        "//tensorflow_graphics/util:export_api",
    ],
)

py_library(
    name = "levenberg_marquardt",
    srcs = ["levenberg_marquardt.py"],
    srcs_version = "PY2AND3",
    deps = [

        # google internal,
        "//tensorflow_graphics/util:export_api",
    ],
)

py_test(
    name = "levenberg_marquardt_test",
    timeout = "long",
    srcs = ["tests/levenberg_marquardt_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":levenberg_marquardt",
        # package dep

        # google internal,
        "//tensorflow_graphics/util:test_case",
    ],
)
