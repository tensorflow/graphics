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
# Data format utilities for tf-graphics.

# google internal package dependency 8
# google internal package dependency 13

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "data_formats",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        ":exr",
    ],
)

py_library(
    name = "exr",
    srcs = ["exr.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "exr_test",
    srcs = ["tests/exr_test.py"],
    deps = [
        ":exr",
        # google internal package dependency 2
        # google internal package dependency 6
        "//tensorflow_graphics/util:test_case",
    ],
)
