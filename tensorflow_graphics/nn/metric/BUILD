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
# Metric tf-graphics.

# google internal package dependency 8
# google internal package dependency 10

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "metric",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    # google internal rule 1
    visibility = ["//visibility:public"],
    deps = [
        ":fscore",
        ":intersection_over_union",
        ":precision",
        ":recall",
        "//tensorflow_graphics/util:export_api",
    ],
)

py_library(
    name = "intersection_over_union",
    srcs = ["intersection_over_union.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        # google internal package dependency 1,
        "//tensorflow_graphics/util:asserts",
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:shape",
        "//tensorflow_graphics/util:tfg_flags",
    ],
)

py_test(
    name = "intersection_over_union_test",
    srcs = ["tests/intersection_over_union_test.py"],
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
        ":intersection_over_union",
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_library(
    name = "fscore",
    srcs = ["fscore.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        ":precision",
        ":recall",
        # google internal package dependency 1,
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:safe_ops",
        "//tensorflow_graphics/util:shape",
    ],
)

py_test(
    name = "fscore_test",
    srcs = ["tests/fscore_test.py"],
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
        ":fscore",
        ":precision",
        ":recall",
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_library(
    name = "precision",
    srcs = ["precision.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        # google internal package dependency 1,
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:safe_ops",
        "//tensorflow_graphics/util:shape",
    ],
)

py_test(
    name = "precision_test",
    srcs = ["tests/precision_test.py"],
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
        ":precision",
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_library(
    name = "recall",
    srcs = ["recall.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        # google internal package dependency 1,
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:safe_ops",
        "//tensorflow_graphics/util:shape",
        "//tensorflow_graphics/util:tfg_flags",
    ],
)

py_test(
    name = "recall_test",
    srcs = ["tests/recall_test.py"],
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
        ":recall",
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)
