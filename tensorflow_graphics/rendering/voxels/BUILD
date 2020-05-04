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
# Voxel rendering functionalities for tf-graphics.

# google internal package dependency 8
# google internal package dependency 10

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "visual_hull",
    srcs = ["visual_hull.py"],
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
    name = "absorption",
    srcs = ["absorption.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        # google internal package dependency 1,
        "//tensorflow_graphics/math:vector",
        "//tensorflow_graphics/util:asserts",
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:shape",
    ],
)

py_library(
    name = "emission_absorption",
    srcs = ["emission_absorption.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        # google internal package dependency 1,
        "//tensorflow_graphics/math:vector",
        "//tensorflow_graphics/util:asserts",
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:shape",
    ],
)

py_library(
    name = "voxels",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        ":absorption",
        ":emission_absorption",
        ":visual_hull",
        "//tensorflow_graphics/util:export_api",
    ],
)

py_library(
    name = "test_helpers",
    srcs = ["tests/test_helpers.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        # google internal package dependency 6
    ],
)

py_test(
    name = "visual_hull_test",
    timeout = "short",
    srcs = ["tests/visual_hull_test.py"],
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
        ":test_helpers",
        ":visual_hull",
        # google internal package dependency 3
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "absorption_test",
    srcs = ["tests/absorption_test.py"],
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
        ":absorption",
        ":test_helpers",
        # google internal package dependency 3
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "emission_absorption_test",
    srcs = ["tests/emission_absorption_test.py"],
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
        ":emission_absorption",
        ":test_helpers",
        # google internal package dependency 3
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)
