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
# Geometric transformation functionalities for tf-graphics.

# google internal package dependency 8
# google internal package dependency 10

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "quaternion",
    srcs = ["quaternion.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        ":rotation_matrix_3d",
        # google internal package dependency 1,
        "//tensorflow_graphics/math:vector",
        "//tensorflow_graphics/util:asserts",
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:safe_ops",
        "//tensorflow_graphics/util:shape",
    ],
)

py_library(
    name = "dual_quaternion",
    srcs = ["dual_quaternion.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        ":quaternion",
        # google internal package dependency 1,
        "//tensorflow_graphics/util:asserts",
        "//tensorflow_graphics/util:shape",
    ],
)

py_library(
    name = "axis_angle",
    srcs = ["axis_angle.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        ":quaternion",
        ":rotation_matrix_3d",
        # google internal package dependency 1,
        "//tensorflow_graphics/math:vector",
        "//tensorflow_graphics/util:asserts",
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:safe_ops",
        "//tensorflow_graphics/util:shape",
    ],
)

py_library(
    name = "rotation_matrix_common",
    srcs = ["rotation_matrix_common.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        # google internal package dependency 1,
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:shape",
    ],
)

py_library(
    name = "rotation_matrix_3d",
    srcs = ["rotation_matrix_3d.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        ":rotation_matrix_common",
        # google internal package dependency 4,
        # google internal package dependency 1,
        "//tensorflow_graphics/util:asserts",
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:shape",
        "//tensorflow_graphics/util:tfg_flags",
    ],
)

py_library(
    name = "rotation_matrix_2d",
    srcs = ["rotation_matrix_2d.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        ":rotation_matrix_common",
        # google internal package dependency 1,
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:shape",
    ],
)

py_library(
    name = "linear_blend_skinning",
    srcs = ["linear_blend_skinning.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        ":rotation_matrix_3d",
        # google internal package dependency 1,
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:shape",
    ],
)

py_library(
    name = "euler",
    srcs = ["euler.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        ":quaternion",
        ":rotation_matrix_3d",
        # google internal package dependency 1,
        "//tensorflow_graphics/util:asserts",
        "//tensorflow_graphics/util:export_api",
        "//tensorflow_graphics/util:safe_ops",
        "//tensorflow_graphics/util:shape",
    ],
)

py_library(
    name = "transformation",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        ":axis_angle",
        ":euler",
        ":linear_blend_skinning",
        ":quaternion",
        ":rotation_matrix_2d",
        ":rotation_matrix_3d",
        ":rotation_matrix_common",
        "//tensorflow_graphics/util:export_api",
    ],
)

py_library(
    name = "test_data",
    srcs = ["tests/test_data.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        # google internal package dependency 6
        # google internal package dependency 1,
    ],
)

py_library(
    name = "test_helpers",
    srcs = ["tests/test_helpers.py"],
    srcs_version = "PY2AND3",
    # google internal rule 1
    deps = [
        ":axis_angle",
        ":quaternion",
        ":rotation_matrix_2d",
        ":rotation_matrix_3d",
        # google internal package dependency 1,
    ],
)

py_test(
    name = "quaternion_test",
    timeout = "long",
    srcs = ["tests/quaternion_test.py"],
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
        ":axis_angle",
        ":quaternion",
        ":rotation_matrix_3d",
        ":test_data",
        ":test_helpers",
        # google internal package dependency 3
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "dual_quaternion_test",
    timeout = "long",
    srcs = ["tests/dual_quaternion_test.py"],
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
        ":dual_quaternion",
        ":quaternion",
        ":test_helpers",
        # google internal package dependency 3
        # google internal package dependency 2
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "axis_angle_test",
    srcs = ["tests/axis_angle_test.py"],
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
        ":axis_angle",
        ":quaternion",
        ":rotation_matrix_3d",
        ":test_helpers",
        # google internal package dependency 3
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "rotation_matrix_common_test",
    srcs = ["tests/rotation_matrix_common_test.py"],
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
        ":rotation_matrix_common",
        # google internal package dependency 2
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "rotation_matrix_3d_test",
    timeout = "long",
    srcs = ["tests/rotation_matrix_3d_test.py"],
    shard_count = 3,
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
        ":axis_angle",
        ":quaternion",
        ":rotation_matrix_3d",
        ":test_data",
        ":test_helpers",
        # google internal package dependency 3
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "rotation_matrix_2d_test",
    srcs = ["tests/rotation_matrix_2d_test.py"],
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
        ":rotation_matrix_2d",
        ":test_data",
        ":test_helpers",
        # google internal package dependency 3
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "linear_blend_skinning_test",
    srcs = ["tests/linear_blend_skinning_test.py"],
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
        ":linear_blend_skinning",
        ":test_helpers",
        # google internal package dependency 3
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)

py_test(
    name = "euler_test",
    srcs = ["tests/euler_test.py"],
    shard_count = 16,
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
        ":axis_angle",
        ":euler",
        ":quaternion",
        ":rotation_matrix_3d",
        ":test_data",
        ":test_helpers",
        # google internal package dependency 3
        # google internal package dependency 2
        # google internal package dependency 6
        # google internal package dependency 1,
        "//tensorflow_graphics/util:test_case",
    ],
)
